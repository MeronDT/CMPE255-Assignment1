"""CRISP-DM Phase 2: Data Understanding.

Profiles the raw January 2024 Yellow Taxi trip records: schema, missingness,
target distributions (duration, fare), and obvious data quality issues that
Phase 3 (Data Preparation) needs to handle. Writes findings as JSON (consumed
by the admin dashboard) and a handful of PNG charts.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DOCS = ROOT / "docs" / "eda"
DOCS.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_parquet(RAW / "yellow_tripdata_2024-01.parquet")
    n_raw = len(df)

    duration_min = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60

    findings = {
        "n_rows_raw": int(n_raw),
        "n_columns": int(df.shape[1]),
        "date_range_claimed": "2024-01-01 to 2024-01-31",
        "date_range_actual": {
            "min_pickup": str(df["tpep_pickup_datetime"].min()),
            "max_pickup": str(df["tpep_pickup_datetime"].max()),
        },
        "missing_values_pct": {
            col: round(float(df[col].isna().mean() * 100), 3)
            for col in df.columns
            if df[col].isna().any()
        },
        "data_quality_issues": [
            {
                "issue": "Pickup timestamps outside claimed month",
                "count": int(((df["tpep_pickup_datetime"] < "2024-01-01") | (df["tpep_pickup_datetime"] >= "2024-02-01")).sum()),
            },
            {
                "issue": "Non-positive or extreme trip duration (<=0 min or >240 min)",
                "count": int(((duration_min <= 0) | (duration_min > 240)).sum()),
            },
            {
                "issue": "Non-positive or extreme fare_amount (<=0 or >250)",
                "count": int(((df["fare_amount"] <= 0) | (df["fare_amount"] > 250)).sum()),
            },
            {
                "issue": "Non-positive or extreme trip_distance (<=0 or >100 miles)",
                "count": int(((df["trip_distance"] <= 0) | (df["trip_distance"] > 100)).sum()),
            },
            {
                "issue": "Unknown/N-A pickup or dropoff zone (LocationID 264/265)",
                "count": int((df["PULocationID"].isin([264, 265]) | df["DOLocationID"].isin([264, 265])).sum()),
            },
            {
                "issue": "Missing passenger_count",
                "count": int(df["passenger_count"].isna().sum()),
            },
        ],
        "target_stats_raw": {
            "duration_min": {
                "mean": float(duration_min.mean()),
                "median": float(duration_min.median()),
                "std": float(duration_min.std()),
                "p99": float(duration_min.quantile(0.99)),
                "max": float(duration_min.max()),
                "min": float(duration_min.min()),
            },
            "fare_amount": {
                "mean": float(df["fare_amount"].mean()),
                "median": float(df["fare_amount"].median()),
                "std": float(df["fare_amount"].std()),
                "p99": float(df["fare_amount"].quantile(0.99)),
                "max": float(df["fare_amount"].max()),
                "min": float(df["fare_amount"].min()),
            },
        },
        "categorical_cardinality": {
            "PULocationID": int(df["PULocationID"].nunique()),
            "DOLocationID": int(df["DOLocationID"].nunique()),
            "VendorID": int(df["VendorID"].nunique()),
            "payment_type": int(df["payment_type"].nunique()),
        },
    }

    with open(DOCS / "eda_findings.json", "w") as f:
        json.dump(findings, f, indent=2)

    # --- Charts ---
    clean_mask = (duration_min > 0) & (duration_min <= 120) & (df["fare_amount"] > 0) & (df["fare_amount"] <= 150)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].hist(duration_min[clean_mask], bins=60, color="#6366f1", edgecolor="none")
    axes[0, 0].set_title("Trip Duration (minutes, clipped 0-120)")
    axes[0, 0].set_xlabel("minutes")

    axes[0, 1].hist(df.loc[clean_mask, "fare_amount"], bins=60, color="#22c55e", edgecolor="none")
    axes[0, 1].set_title("Fare Amount ($, clipped 0-150)")
    axes[0, 1].set_xlabel("USD")

    hourly = df["tpep_pickup_datetime"].dt.hour.value_counts().sort_index()
    axes[1, 0].bar(hourly.index, hourly.values, color="#f59e0b")
    axes[1, 0].set_title("Trips by Pickup Hour")
    axes[1, 0].set_xlabel("hour of day")

    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow = df["tpep_pickup_datetime"].dt.dayofweek.value_counts().sort_index()
    axes[1, 1].bar([dow_labels[i] for i in dow.index], dow.values, color="#ec4899")
    axes[1, 1].set_title("Trips by Day of Week")

    plt.tight_layout()
    plt.savefig(DOCS / "eda_overview.png", dpi=130)
    plt.close(fig)

    print(f"Raw rows: {n_raw:,}")
    print(json.dumps(findings["data_quality_issues"], indent=2))
    print(f"\nWrote findings -> {DOCS / 'eda_findings.json'}")
    print(f"Wrote chart -> {DOCS / 'eda_overview.png'}")


if __name__ == "__main__":
    main()
