"""CRISP-DM Phase 3: Data Preparation.

Cleans the raw trip records and engineers a feature set restricted to what the
deployed app can actually know *before* a trip happens: pickup/dropoff zone
and requested time. `trip_distance` (the taxi meter reading) is deliberately
excluded as a model feature — it isn't known at estimation time, so training
on it would create train/serve skew and leak the answer. In its place we use
haversine distance between zone centroids (computable identically online and
offline) plus a historical average-distance-by-OD-pair lookup, which *is*
legitimately available at serving time (it's a static table, not telemetry).
"""

import json
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

RUSH_HOURS = set(range(7, 10)) | set(range(16, 19))


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def main():
    print("Loading raw data...")
    df = pd.read_parquet(RAW / "yellow_tripdata_2024-01.parquet")
    n_raw = len(df)

    with open(RAW / "taxi_zone_centroids.json") as f:
        centroids = json.load(f)
    zone_lookup = pd.read_csv(RAW / "taxi_zone_lookup.csv")

    # --- Cleaning (rules justified by 01_eda.py findings) ---
    df["duration_min"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60

    mask = (
        (df["tpep_pickup_datetime"] >= "2024-01-01")
        & (df["tpep_pickup_datetime"] < "2024-02-01")
        & (df["duration_min"] > 1)
        & (df["duration_min"] <= 180)
        & (df["fare_amount"] > 0)
        & (df["fare_amount"] <= 250)
        & (df["trip_distance"] > 0)
        & (df["trip_distance"] <= 100)
        & (~df["PULocationID"].isin([264, 265]))
        & (~df["DOLocationID"].isin([264, 265]))
        & (df["PULocationID"].astype(str).isin(centroids))
        & (df["DOLocationID"].astype(str).isin(centroids))
    )
    df = df.loc[mask].copy()
    n_clean = len(df)
    df["passenger_count"] = df["passenger_count"].fillna(1).clip(lower=1, upper=6).astype(int)

    # --- Feature engineering (pre-trip-only features) ---
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_dow"] = df["tpep_pickup_datetime"].dt.dayofweek  # 0=Mon
    df["pickup_month_day"] = df["tpep_pickup_datetime"].dt.day
    df["is_weekend"] = (df["pickup_dow"] >= 5).astype(int)
    df["is_rush_hour"] = df["pickup_hour"].isin(RUSH_HOURS).astype(int)

    pu_lat = df["PULocationID"].astype(str).map(lambda z: centroids[z]["lat"])
    pu_lon = df["PULocationID"].astype(str).map(lambda z: centroids[z]["lon"])
    do_lat = df["DOLocationID"].astype(str).map(lambda z: centroids[z]["lat"])
    do_lon = df["DOLocationID"].astype(str).map(lambda z: centroids[z]["lon"])
    df["haversine_km"] = haversine_km(pu_lat, pu_lon, do_lat, do_lon)

    borough_by_id = zone_lookup.set_index("LocationID")["Borough"].to_dict()
    df["pickup_borough"] = df["PULocationID"].map(borough_by_id)
    df["dropoff_borough"] = df["DOLocationID"].map(borough_by_id)
    df["same_borough"] = (df["pickup_borough"] == df["dropoff_borough"]).astype(int)
    df["is_airport_ratecode"] = df["RatecodeID"].isin([2, 3]).astype(int)  # JFK / Newark flat fare

    df["PULocationID"] = df["PULocationID"].astype(int)
    df["DOLocationID"] = df["DOLocationID"].astype(int)

    feature_cols = [
        "PULocationID",
        "DOLocationID",
        "pickup_borough",
        "dropoff_borough",
        "same_borough",
        "haversine_km",
        "pickup_hour",
        "pickup_dow",
        "is_weekend",
        "is_rush_hour",
        "passenger_count",
        "is_airport_ratecode",
    ]
    target_cols = ["duration_min", "fare_amount"]

    model_df = df[feature_cols + target_cols].copy()
    for col in ["pickup_borough", "dropoff_borough", "PULocationID", "DOLocationID"]:
        model_df[col] = model_df[col].astype("category")

    # Historical avg distance/duration per OD pair -> served as a static lookup
    # at inference time (legitimate: it's precomputed, not per-request telemetry).
    od_stats = (
        df.groupby(["PULocationID", "DOLocationID"])
        .agg(
            avg_trip_distance_mi=("trip_distance", "mean"),
            avg_duration_min=("duration_min", "mean"),
            avg_fare_amount=("fare_amount", "mean"),
            n_trips=("duration_min", "size"),
        )
        .reset_index()
    )
    od_stats = od_stats[od_stats["n_trips"] >= 5]  # drop noisy low-sample OD pairs

    # --- Train/test split (time-based: last 20% of days held out, avoids
    # leaking future OD-pair patterns backward into training) ---
    cutoff_day = int(df["pickup_month_day"].quantile(0.8))
    train_df = model_df[df["pickup_month_day"] <= cutoff_day]
    test_df = model_df[df["pickup_month_day"] > cutoff_day]

    train_df.to_parquet(PROCESSED / "train.parquet", index=False)
    test_df.to_parquet(PROCESSED / "test.parquet", index=False)
    od_stats.to_parquet(PROCESSED / "od_stats.parquet", index=False)

    summary = {
        "n_rows_raw": n_raw,
        "n_rows_after_cleaning": n_clean,
        "pct_dropped": round((1 - n_clean / n_raw) * 100, 2),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "cutoff_day_of_month": cutoff_day,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "n_od_pairs_with_history": len(od_stats),
    }
    with open(ROOT / "docs" / "eda" / "prep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
