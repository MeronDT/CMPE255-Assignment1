"""Simplify taxi_zones.geojson (Douglas-Peucker) and round coordinate precision
so the frontend can ship it without a multi-MB payload. Zone shapes only need to
look right at borough/city zoom levels, not survey-grade precision.
"""

import json
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
CLIENT_PUBLIC = Path(__file__).resolve().parents[1] / "frontend" / "public"

TOLERANCE_DEG = 0.00008  # ~9m at NYC's latitude
PRECISION = 5


def perpendicular_distance(pt, start, end):
    if start == end:
        return ((pt[0] - start[0]) ** 2 + (pt[1] - start[1]) ** 2) ** 0.5
    x, y = pt
    x1, y1 = start
    x2, y2 = end
    num = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    den = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
    return num / den


def douglas_peucker(points, tolerance):
    if len(points) < 3:
        return points
    start, end = points[0], points[-1]
    max_dist, index = 0, 0
    for i in range(1, len(points) - 1):
        d = perpendicular_distance(points[i], start, end)
        if d > max_dist:
            max_dist, index = d, i
    if max_dist > tolerance:
        left = douglas_peucker(points[: index + 1], tolerance)
        right = douglas_peucker(points[index:], tolerance)
        return left[:-1] + right
    return [start, end]


def simplify_ring(ring):
    simplified = douglas_peucker(ring, TOLERANCE_DEG)
    return [[round(x, PRECISION), round(y, PRECISION)] for x, y in simplified]


def simplify_geometry(geom):
    if geom["type"] == "Polygon":
        return {"type": "Polygon", "coordinates": [simplify_ring(r) for r in geom["coordinates"]]}
    polys = [[simplify_ring(r) for r in poly] for poly in geom["coordinates"]]
    return {"type": "MultiPolygon", "coordinates": polys}


def main():
    with open(RAW / "taxi_zones.geojson", encoding="utf-8") as f:
        data = json.load(f)

    for feature in data["features"]:
        feature["geometry"] = simplify_geometry(feature["geometry"])

    CLIENT_PUBLIC.mkdir(parents=True, exist_ok=True)
    out_path = CLIENT_PUBLIC / "taxi_zones.geojson"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))

    before = (RAW / "taxi_zones.geojson").stat().st_size
    after = out_path.stat().st_size
    print(f"Simplified {before/1024:.0f}KB -> {after/1024:.0f}KB, written to {out_path}")


if __name__ == "__main__":
    main()
