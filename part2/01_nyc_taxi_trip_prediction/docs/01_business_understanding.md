# CRISP-DM Phase 1: Business Understanding

## Business Objectives

A rider opening a taxi app wants two numbers before requesting a trip: **how long will it take** and **what will it cost**. Both need to be available *before* the trip starts — from just a pickup location, a dropoff location, and a requested time. Getting these estimates right (and being transparent about their uncertainty) is core to trust in any ride-hailing product.

## Data Mining Goals

Translate the business objective into two supervised regression problems, trained on NYC TLC's public Yellow Taxi trip records (January 2024):

1. **Trip duration** (minutes) — regression target `duration_min`, derived from `tpep_dropoff_datetime - tpep_pickup_datetime`.
2. **Fare amount** (USD) — regression target `fare_amount`, the metered base fare (excludes tips, which are rider-discretionary and not predictable from trip characteristics).

## Success Criteria

- **Statistical**: beat a naive mean-baseline by a wide margin on held-out data, report RMSE / MAE / R² per target (see [`docs/eda/model_search_log.json`](./eda/model_search_log.json) and the Model Insights dashboard).
- **Product-realistic**: only use features genuinely available at estimation time — pickup zone, dropoff zone, and requested pickup time. The raw dataset's `trip_distance` (the taxi meter reading) is deliberately **excluded** as a model feature, since a rider requesting an estimate hasn't taken the trip yet and the meter hasn't run. Training on it would leak the answer and silently fail in production when that column isn't available. This decision is documented in [`DESIGN_DOC.md`](./../DESIGN_DOC.md#5-key-technical-decisions).
- **Deployable**: served live via a REST API behind an interactive map UI, not just a notebook metric.

## Constraints

- **Compute**: local laptop CPU (no GPU needed — gradient-boosted trees on ~2.4M rows train in under a minute per LightGBM configuration).
- **Data availability at inference time**: no telemetry, no live traffic feed, no routing engine — only pickup/dropoff zone identity and requested time, plus static historical lookups (e.g., average distance/fare/duration per zone-pair, precomputed offline).
- **Privacy**: NYC TLC data has anonymized pickup/dropoff to 263 named zones (not exact lat/lon) since 2016 — the whole pipeline, including the interactive map, works at zone granularity, not street-address granularity.

## Data Source

[NYC TLC Yellow Taxi Trip Records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), January 2024 (~2.96M trips), fetched directly from TLC's public CloudFront bucket — the same underlying data source as the Kaggle NYC Taxi competitions, with no API key or account required. Taxi zone shapes and lookup table from the same source, used both for the interactive map and for zone-centroid-based distance features.
