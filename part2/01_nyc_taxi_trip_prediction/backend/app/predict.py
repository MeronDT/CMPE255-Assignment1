"""Loads the trained LightGBM boosters and rebuilds the exact feature pipeline
used in scripts/02_prepare_data.py, so serving-time features match training-time
features byte-for-byte (including categorical dtype codes, which LightGBM's
Booster.predict requires to line up with what it was trained on).
"""

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from math import radians, sin, cos, asin, sqrt

import lightgbm as lgb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

RUSH_HOURS = set(range(7, 10)) | set(range(16, 19))
CAT_COLS = ["pickup_borough", "dropoff_borough", "PULocationID", "DOLocationID"]


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(a))


class TaxiModelService:
    def __init__(self):
        self.duration_model = lgb.Booster(model_file=str(MODELS / "duration_model.txt"))
        self.fare_model = lgb.Booster(model_file=str(MODELS / "fare_model.txt"))

        with open(MODELS / "category_maps.json") as f:
            self.category_maps: dict[str, list[str]] = json.load(f)
        with open(MODELS / "feature_cols.json") as f:
            self.feature_cols: list[str] = json.load(f)

        with open(RAW / "taxi_zone_centroids.json") as f:
            self.centroids: dict[str, dict] = json.load(f)

        zone_df = pd.read_csv(RAW / "taxi_zone_lookup.csv")
        self.zones = zone_df.set_index("LocationID").to_dict(orient="index")
        self.airport_zone_ids = set(
            zone_df.loc[zone_df["service_zone"].isin(["Airports", "EWR"]), "LocationID"].tolist()
        )

        od_df = pd.read_parquet(PROCESSED / "od_stats.parquet")
        self.od_stats = {
            (int(r.PULocationID), int(r.DOLocationID)): {
                "n_trips": int(r.n_trips),
                "avg_duration_min": float(r.avg_duration_min),
                "avg_fare_amount": float(r.avg_fare_amount),
                "avg_trip_distance_mi": float(r.avg_trip_distance_mi),
            }
            for r in od_df.itertuples(index=False)
        }

    def zone_meta(self, location_id: int) -> dict:
        if location_id not in self.zones or str(location_id) not in self.centroids:
            raise ValueError(f"Unknown or unsupported LocationID: {location_id}")
        z = self.zones[location_id]
        c = self.centroids[str(location_id)]
        return {"zone": z["Zone"], "borough": z["Borough"], "lat": c["lat"], "lon": c["lon"]}

    def build_features(self, pickup_id: int, dropoff_id: int, pickup_dt: datetime, passenger_count: int) -> pd.DataFrame:
        pu = self.zone_meta(pickup_id)
        do = self.zone_meta(dropoff_id)

        row = {
            "PULocationID": pickup_id,
            "DOLocationID": dropoff_id,
            "pickup_borough": pu["borough"],
            "dropoff_borough": do["borough"],
            "same_borough": int(pu["borough"] == do["borough"]),
            "haversine_km": haversine_km(pu["lat"], pu["lon"], do["lat"], do["lon"]),
            "pickup_hour": pickup_dt.hour,
            "pickup_dow": pickup_dt.weekday(),
            "is_weekend": int(pickup_dt.weekday() >= 5),
            "is_rush_hour": int(pickup_dt.hour in RUSH_HOURS),
            "passenger_count": passenger_count,
            "is_airport_ratecode": int(pickup_id in self.airport_zone_ids or dropoff_id in self.airport_zone_ids),
        }
        df = pd.DataFrame([row])[self.feature_cols]
        for col in CAT_COLS:
            df[col] = pd.Categorical([str(row[col])], categories=self.category_maps[col])
        return df

    def predict(self, pickup_id: int, dropoff_id: int, pickup_dt: datetime, passenger_count: int) -> dict:
        X = self.build_features(pickup_id, dropoff_id, pickup_dt, passenger_count)
        duration = max(1.0, float(self.duration_model.predict(X)[0]))
        fare = max(2.5, float(self.fare_model.predict(X)[0]))

        pu = self.zone_meta(pickup_id)
        do = self.zone_meta(dropoff_id)
        historical = self.od_stats.get((pickup_id, dropoff_id))

        return {
            "predicted_duration_min": round(duration, 1),
            "predicted_fare_amount": round(fare, 2),
            "haversine_km": round(X["haversine_km"].iloc[0], 2),
            "pickup_zone": pu["zone"],
            "dropoff_zone": do["zone"],
            "pickup_borough": pu["borough"],
            "dropoff_borough": do["borough"],
            "is_rush_hour": bool(X["is_rush_hour"].iloc[0]),
            "is_airport_trip": bool(pickup_id in self.airport_zone_ids or dropoff_id in self.airport_zone_ids),
            "historical": historical,
        }

    def list_zones(self) -> list[dict]:
        out = []
        for location_id, meta in self.zones.items():
            c = self.centroids.get(str(location_id))
            if not c:
                continue
            out.append(
                {
                    "location_id": location_id,
                    "zone": meta["Zone"],
                    "borough": meta["Borough"],
                    "service_zone": meta["service_zone"],
                    "lat": c["lat"],
                    "lon": c["lon"],
                }
            )
        return sorted(out, key=lambda z: (z["borough"], z["zone"]))


@lru_cache(maxsize=1)
def get_service() -> "TaxiModelService":
    return TaxiModelService()
