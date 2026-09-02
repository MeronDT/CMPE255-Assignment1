"""CRISP-DM Phase 2 (data collection step): download raw source data.

Pulls directly from NYC TLC's public CloudFront bucket — the same underlying
source as the Kaggle NYC Taxi competitions, no API key or account required.
"""

import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

FILES = {
    "yellow_tripdata_2024-01.parquet": "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet",
    "taxi_zone_lookup.csv": "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv",
    "taxi_zones.zip": "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip",
}


def main():
    for filename, url in FILES.items():
        dest = RAW / filename
        if dest.exists():
            print(f"Already have {filename}, skipping")
            continue
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, dest)

    zones_dir = RAW / "taxi_zones_shp"
    if not zones_dir.exists():
        print("Extracting taxi_zones.zip...")
        with zipfile.ZipFile(RAW / "taxi_zones.zip") as zf:
            zf.extractall(zones_dir)

    print("Done. Next: python scripts/convert_zones_to_geojson.py && python scripts/simplify_geojson.py")


if __name__ == "__main__":
    main()
