# Research Report: NYC Taxi Trip Duration & Fare Prediction

**Methodology**: CRISP-DM · **Data**: NYC TLC Yellow Taxi, January 2024 (2,964,624 raw trips) · **Models**: LightGBM (duration, fare) selected via a 4-phase AutoResearch hill-climbing search

---

## Abstract

We build two regression models — trip duration and fare amount — restricted to features genuinely available *before* a trip starts (pickup/dropoff zone, requested time), deployed behind a live estimation API. An AutoResearch pipeline (multi-backbone tournament → feature-transform search → hyperparameter hill-climbing → blending) was run to validate the modeling choices against alternatives, with full search telemetry logged and surfaced in an admin dashboard. Final held-out test performance: **duration RMSE 5.16 min (R²=0.788)**, **fare RMSE $3.94 (R²=0.929)**, evaluated on a time-based split (no future data leaking into training). Both figures are discussed against comparable published work in §6; the comparison is not apples-to-apples, and we explain why.

---

## 1. Business Understanding

A rider needs a duration and fare estimate *before* requesting a trip, from only pickup location, dropoff location, and requested time — see [`docs/01_business_understanding.md`](./docs/01_business_understanding.md) for full objectives, constraints, and success criteria. The central design constraint that shapes every later phase: **no feature may depend on the trip having already happened** (e.g., the taxi meter's actual distance reading is unusable, since it doesn't exist yet at estimation time).

## 2. Data Understanding

Source: [NYC TLC Yellow Taxi Trip Records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), January 2024, fetched directly from TLC's public bucket (no Kaggle account needed — same underlying data). Full profiling in [`docs/eda/eda_findings.json`](./docs/eda/eda_findings.json); headline findings:

- 2,964,624 raw trips, 19 raw columns.
- Real data quality issues found and quantified: 18 trips with pickup timestamps outside the claimed month, 2,748 trips with non-positive/extreme duration, 38,882 with non-positive/extreme fare, 60,430 with non-positive/extreme distance, 31,527 with unknown pickup/dropoff zone, 140,162 with missing passenger count.
- **A concrete signal, not noise**: a sharp spike at ~$70 in the fare-amount distribution — NYC's flat-rate JFK Airport fare (RatecodeID=2). This later shows up correctly in both the trained model's predictions and its feature importances (`is_airport_ratecode` is a top-8 feature for the fare model).

## 3. Data Preparation

Cleaning rules (each directly justified by a Data Understanding finding — see [`docs/eda/prep_summary.json`](./docs/eda/prep_summary.json)) removed 4.5% of rows (2,964,624 → 2,831,174), split 2,360,809 train / 470,365 test **by day-of-month** (not randomly — a random split would leak future OD-pair demand patterns backward into training).

**Feature set — 12 pre-trip-only features**: `PULocationID`, `DOLocationID`, `pickup_borough`, `dropoff_borough`, `same_borough`, `haversine_km` (zone-centroid distance, not the taxi meter's `trip_distance`), `pickup_hour`, `pickup_dow`, `is_weekend`, `is_rush_hour`, `passenger_count`, `is_airport_ratecode`. The taxi meter's actual `trip_distance` — by far the strongest possible predictor — is **deliberately excluded**; see §7 for why this matters when comparing to other published results.

## 4. Modeling

### 4.1 Initial search
A first LightGBM hyperparameter sweep (5 configurations, full 2.36M-row training set) selected `num_leaves=127, learning_rate=0.05, n_estimators=500, min_child_samples=80` for both targets — logged in [`docs/eda/model_search_log.json`](./docs/eda/model_search_log.json).

### 4.2 AutoResearch: 4-phase hill-climbing search
To validate that choice rather than assume it, we ran the `nyc-taxi-autoresearch` methodology end to end (full telemetry: [`docs/eda/autoresearch_history.json`](./docs/eda/autoresearch_history.json), also live at `/api/autoresearch` and rendered in the Model Insights dashboard):

1. **Multi-backbone tournament** (Ridge, Random Forest, LightGBM, XGBoost) on a 300K-row search sample: **LightGBM won both targets** (duration RMSE 5.49 vs. 5.63–7.03 for alternatives; fare RMSE 4.22 vs. 4.33–5.46), confirming gradient-boosted trees are the right model family for this structured, mixed categorical/numeric tabular problem.
2. **Feature-transform search**: tried `log1p(haversine_km)`, cyclical hour encoding (sin/cos), and a distance×rush-hour interaction term against the untransformed baseline. Cyclical hour encoding won for duration; the distance×rush interaction won for fare — both retained.
3. **Hyperparameter hill-climbing** (literal greedy local search — not random/grid search): starting from the Phase 4.1 seed config, every run evaluates all one-step neighbors across `num_leaves`, `learning_rate`, `n_estimators`, and `min_child_samples`, and moves to the best-improving neighbor until none improves (a local optimum). Duration converged after 5 accepted moves (RMSE 5.47→5.44 on the search sample); fare after 2 moves (4.22→4.20).
4. **Blending**: weighted-averaging the hill-climbed model with the seed model gave a further small improvement on the search sample for both targets.
5. **Full-data finalization**: the AutoResearch-winning configuration was retrained on the *full* 2.36M-row training set and compared against the production model on the real held-out test set ([`docs/eda/autoresearch_finalize.json`](./docs/eda/autoresearch_finalize.json)). Result: **the search-sample-tuned config did not beat the model already tuned on the full training set** (duration: 5.1628 vs. 5.1629; fare: 3.9363 vs. 3.9460) — so the original Phase 4.1 model was kept in production. This is a genuine, useful negative result: hyperparameter search on a 300K-row subsample can converge to a config that's locally better *for that subsample* without generalizing better on the full 2.36M-row distribution.

## 5. Evaluation

| Target | Test RMSE | Test MAE | Test R² | vs. naive mean baseline |
|---|---:|---:|---:|---:|
| Trip duration (min) | 5.16 | 3.22 | 0.788 | 56% lower RMSE |
| Fare amount ($) | 3.94 | 2.42 | 0.929 | 74% lower RMSE |

Feature importance (both models): `PULocationID`/`DOLocationID` dominate, followed by `pickup_hour`, `haversine_km`, and `pickup_dow` — sensible: *where* and *when* a trip happens carries almost all the predictable signal available before the trip starts. Full residual/predicted-vs-actual plots: [`docs/eda/evaluation_duration_min.png`](./docs/eda/evaluation_duration_min.png), [`evaluation_fare_amount.png`](./docs/eda/evaluation_fare_amount.png).

**Real-world sanity check**: JFK Airport → Midtown Manhattan predicts a ~$69 fare — matching NYC's actual ~$70 JFK flat-rate fare policy, which our model was never told about explicitly; it inferred it from `is_airport_ratecode` and zone identity alone.

## 6. Deployment

FastAPI backend (`/api/predict`, `/api/zones`, `/api/model-info`, `/api/autoresearch`) + a React/Leaflet frontend with an interactive taxi-zone map for live estimation and a Model Insights dashboard exposing every artifact referenced above. See [`DESIGN_DOC.md`](./DESIGN_DOC.md) for architecture and [`README.md`](./README.md) for setup.

## 7. Related Work & Benchmark Comparison

- **Ye et al., "New York City taxi trip duration prediction using MLP and XGBoost"** (ResearchGate, 2021) report **R²=0.82** (MLP) / **0.81** (bagging random forest) for duration prediction. Our duration model: **R²=0.788** — close, but not directly comparable: their feature set includes exact pickup/dropoff coordinates and (per standard practice on this task) the completed trip's characteristics, giving strictly more signal than our pre-trip-only constraint permits.
- **arXiv:2507.20008, "Robust Taxi Fare Prediction under Noisy Conditions: GAT, TimesNet, XGBoost"** (2025) benchmarks XGBoost on a 55M-row dataset with full completed-trip features, reporting very low fare RMSE under clean conditions. Again not directly comparable — their setup has access to the actual metered distance, which is the single strongest fare predictor and is exactly what we exclude by design (§3).

**The honest takeaway**: our R²/RMSE figures are somewhat below full-feature literature benchmarks, and that gap is *expected and correct* — it's the cost of solving the actual product problem (estimate before the trip, not evaluate after it), not a modeling deficiency. A model matching those benchmarks would necessarily be leaking post-trip information and would fail in production.

## 8. Limitations & Future Work

- **Single month of data** (Jan 2024) — no seasonal (summer vs. winter demand/traffic) validation yet; a natural next step is multi-month training with month-of-year as a feature.
- **Zone-level, not address-level** — inherent to the anonymized public data source, not a modeling choice we can undo.
- **AutoResearch search sample (300K rows)** — Phase 5's finalization step shows this can mildly overfit to the sample; a larger search sample (or full-data hill-climbing, at higher compute cost) is the natural follow-up.
- **No traffic/weather features** — both are known drivers of real-world duration variance and aren't in the TLC dataset; incorporating a weather API would be the highest-leverage next feature addition.
