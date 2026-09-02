"""Convert the TLC taxi_zones shapefile (NY State Plane ft) to WGS84 GeoJSON,
merged with the zone lookup CSV (borough/zone names) and per-zone centroids.
Output is consumed directly by the frontend map and by the feature-engineering
pipeline (for haversine distance between pickup/dropoff zone centroids).
"""

import csv
import json
from pathlib import Path

import shapefile  # pyshp
from pyproj import Transformer

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
SHP = RAW / "taxi_zones_shp" / "taxi_zones" / "taxi_zones.shp"
LOOKUP = RAW / "taxi_zone_lookup.csv"
OUT_GEOJSON = RAW / "taxi_zones.geojson"
OUT_CENTROIDS = RAW / "taxi_zone_centroids.json"

transformer = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)


def transform_ring(ring):
    return [list(transformer.transform(x, y)) for x, y in ring]


def polygon_centroid(rings):
    # Centroid of the largest ring by |signed area| (shoelace formula).
    best_ring, best_area = None, 0.0
    for ring in rings:
        area = 0.0
        for i in range(len(ring) - 1):
            x1, y1 = ring[i]
            x2, y2 = ring[i + 1]
            area += x1 * y2 - x2 * y1
        area = abs(area) / 2
        if area > best_area:
            best_area, best_ring = area, ring
    if not best_ring:
        return None
    cx = sum(p[0] for p in best_ring) / len(best_ring)
    cy = sum(p[1] for p in best_ring) / len(best_ring)
    return [cx, cy]


def main():
    lookup = {}
    with open(LOOKUP, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lookup[int(row["LocationID"])] = {
                "borough": row["Borough"],
                "zone": row["Zone"],
                "service_zone": row["service_zone"],
            }

    sf = shapefile.Reader(str(SHP))
    fields = [f[0] for f in sf.fields[1:]]

    features = []
    centroids = {}

    for sr in sf.shapeRecords():
        rec = dict(zip(fields, sr.record))
        location_id = int(rec.get("LocationID"))
        geom = sr.shape.__geo_interface__
        if geom["type"] == "Polygon":
            rings_ll = [transform_ring(r) for r in geom["coordinates"]]
            geometry = {"type": "Polygon", "coordinates": rings_ll}
        else:  # MultiPolygon
            polys_ll = [[transform_ring(r) for r in poly] for poly in geom["coordinates"]]
            geometry = {"type": "MultiPolygon", "coordinates": polys_ll}
            rings_ll = [ring for poly in polys_ll for ring in poly]

        meta = lookup.get(location_id, {"borough": "Unknown", "zone": rec.get("zone", ""), "service_zone": ""})
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "location_id": location_id,
                    "zone": meta["zone"],
                    "borough": meta["borough"],
                    "service_zone": meta["service_zone"],
                },
                "geometry": geometry,
            }
        )

        c = polygon_centroid(rings_ll)
        if c:
            centroids[str(location_id)] = {
                "lon": c[0],
                "lat": c[1],
                "zone": meta["zone"],
                "borough": meta["borough"],
            }

    with open(OUT_GEOJSON, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)

    with open(OUT_CENTROIDS, "w", encoding="utf-8") as f:
        json.dump(centroids, f, indent=2)

    print(f"Wrote {len(features)} zone polygons -> {OUT_GEOJSON}")
    print(f"Wrote {len(centroids)} zone centroids -> {OUT_CENTROIDS}")


if __name__ == "__main__":
    main()
