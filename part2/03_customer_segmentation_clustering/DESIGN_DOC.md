# Design Doc — Customer Intelligence & Segmentation Clustering

## 1. Overview

Unsupervised RFM-based customer segmentation on the UCI/Kaggle Online Retail dataset
(541,909 transactions, Dec 2010–Dec 2011). Full CRISP-DM pipeline as executable scripts,
an AutoResearch hill-climbing search, and a FastAPI + React admin dashboard. See
[docs/01_business_understanding.md](./docs/01_business_understanding.md) for scope and
objectives.

## 2. Data Pipeline

| Script | CRISP-DM Phase | Output |
|---|---|---|
| `01_eda.py` | Data Understanding | `docs/eda/eda_findings.json` |
| `02_prepare_data.py` | Data Preparation | `data/processed/customer_rfm.csv` (4,371 customers × 9 features) |
| `03_train_models.py` | Modeling (baseline) | `docs/eda/modeling_baseline.json`, `models/baseline_kmeans.joblib` |
| `04_autoresearch.py` | AutoResearch search | `docs/eda/autoresearch.json`, `models/production_kmeans.joblib` |
| `05_evaluate.py` | Evaluation | `docs/eda/cluster_profiles.json` |

## 3. Feature Engineering

Transaction log → customer-level table via `groupby("CustomerID")`:

- **Recency**: days since last purchase (relative to a snapshot date = max invoice date + 1 day)
- **Frequency**: distinct invoice count
- **Monetary**: net revenue (`Quantity × UnitPrice`, summed — cancellations net negative)
- **AvgBasketValue**: Monetary / Frequency
- **DistinctProducts**: distinct StockCode count
- **TotalItems**, **TenureDays** (first-to-last purchase span), **CancellationRate**
- **PrimaryCountry**: modal country (for display only, not a clustering feature — country
  is 91% UK-dominated per EDA, so including it as a clustering dimension would just
  re-derive the UK/non-UK split rather than add behavioral signal)

## 4. Modeling & AutoResearch

**Baseline** (`03_train_models.py`): K-Means/Agglomerative(Ward)/GaussianMixture/DBSCAN
tournament, k chosen by silhouette sweep (k=2..10). Pure silhouette maximization picks
**k=2** — statistically best-separated but not business-actionable (see §5 below).

**AutoResearch** (`04_autoresearch.py`), following the pattern from Projects 01–02:
1. Report the unconstrained optimum honestly (k=2) before applying any constraint.
2. Feature-transform search: 2 feature sets × 3 scalers (standard/robust/minmax) at a
   representative k=5.
3. Greedy hill-climb over k ∈ [3, 8] — a business-actionable range grounded in the RFM
   segmentation literature (Hughes 1994; Christy et al. 2021) — using the winning
   transform config from step 2.
4. Finalize: retrain the winner, deploy if it's a genuine improvement.

No proxy-subsampling step is needed (unlike the taxi/NanoLlama AutoResearch runs) — the
full dataset is only 4,371 rows, so every search step already runs on 100% of the data.

## 5. Key Decision: the k=2 vs. k=3 Trade-off

This is the project's central honest finding, not a footnote. Pure silhouette
maximization over an unconstrained k range finds k=2 (silhouette 0.355) — but a 2-segment
split ("big spenders" vs. "everyone else") gives a marketing team nothing to act on,
failing the actual business objective in `docs/01_business_understanding.md`. Applying a
business-motivated constraint (k ≥ 3, from the RFM literature's typical 4–6-segment
range) and re-searching finds k=3 with silhouette **0.409** — genuinely *better separated*
than the unconstrained optimum, not a worse-but-more-interpretable compromise. This is
reported transparently in the AutoResearch dashboard tab rather than only showing the
final winner, so a reviewer can see the constraint's effect for themselves.

## 6. Deployment

FastAPI backend (`backend/app/main.py`, port 8003) serves every artifact above as JSON.
React + TypeScript + Recharts frontend (port 5176) — six tabs: Overview, Segments
(scatter + profile table), Explorer (searchable customer table), Modeling (algorithm
tournament), AutoResearch (search history + honest k=2-vs-k=3 discussion), Data & EDA.

Verified via Playwright: 0 console errors, 0 failed requests across all six tabs;
screenshots in `docs/screenshots/`.
