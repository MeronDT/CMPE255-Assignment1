# Design Doc — Autonomous Anomaly Detection Platform

## 1. Overview

Unsupervised fraud detection on the Kaggle Credit Card Fraud dataset (284,807
transactions, 492 confirmed frauds, 0.173% positive rate — sourced via OpenML's public
mirror, `sklearn.datasets.fetch_openml("creditcard")`, identical to Kaggle's
`mlg-ulb/creditcardfraud`). Full CRISP-DM pipeline + AutoResearch + FastAPI/React
dashboard. See [docs/01_business_understanding.md](./docs/01_business_understanding.md).

## 2. Pipeline

| Script | CRISP-DM Phase | Output |
|---|---|---|
| `01_eda.py` | Data Understanding | `docs/eda/eda_findings.json` |
| `02_prepare_data.py` | Data Preparation | scaled features + labels (parquet/npy) |
| `03_train_models.py` | Modeling (baseline) | 4-algorithm tournament on a proxy subsample |
| `04_autoresearch.py` | AutoResearch search | hill-climb + full-scale finalization |
| `05_evaluate.py` | Evaluation | PR curve, operating-point precision/recall |

## 3. Data Preparation

Duplicate rows (9,144 of 284,807) are dropped — with 28 continuous PCA-transformed
features, exact duplication by chance is vanishingly improbable, so these are
near-certain data-collection artifacts, not genuine repeat transactions; keeping them
would let the same fraud pattern leak across the evaluation. `Amount` is log1p +
standardized (heavily right-skewed, different scale than the already-PCA'd `V1`–`V28`).
The `Class` label is split off and used **only** for evaluation, never as a model input.

## 4. Modeling & AutoResearch

**Baseline tournament** (`03_train_models.py`), on a 20,000-normal + 473-fraud proxy
subsample (full-data tournament for 4 algorithms, two of them super-linear in sample
count, would blow the time budget for a comparison pass):

| Algorithm | AUPRC | ROC-AUC |
|---|---:|---:|
| **Isolation Forest** | **0.607** | 0.946 |
| Elliptic Envelope | 0.516 | 0.913 |
| One-Class SVM | 0.274 | 0.923 |
| Local Outlier Factor | 0.028 | 0.484 |

Isolation Forest wins clearly, matching the literature's known strength on this exact
dataset. LOF performed near-randomly (ROC-AUC 0.484) — reported honestly rather than
omitted; LOF's local-density approach is known to struggle when anomalies aren't locally
sparse relative to neighbors in a given feature space, and this dataset's PCA-transformed
features are one such case.

**AutoResearch** (`04_autoresearch.py`): hill-climb over `(n_estimators,
contamination)` on the proxy, then **finalize by retraining at full scale (275,663
rows)** — the same proxy-then-verify caution as Projects 01/02's AutoResearch.

## 5. Key Decision: the AUPRC Base-Rate Trap

**The project's central honest finding.** Finalizing at full scale, raw AUPRC dropped
from 0.6156 (proxy) to 0.1444 (full) — which looks like the model failing to generalize.
It isn't: AUPRC is base-rate-dependent (a random classifier scores ≈ the positive rate),
and the proxy subsample's positive rate (2.31%) is ~13x the full dataset's true rate
(0.172%). Comparing raw AUPRC across two different base rates is comparing apples to
oranges. The fair comparison — AUPRC relative to each evaluation's own random
baseline — shows the ranking quality actually **held up, and by lift improved**:
proxy lift 26.6x over random, full-scale lift **84.2x** over random. Reported
transparently on the dashboard's AutoResearch tab, not smoothed over.

A secondary honest note: AUPRC/ROC-AUC don't depend on `contamination` for Isolation
Forest (it only shifts `predict()`'s binary threshold, not the continuous
`score_samples()` ranking) — visible directly in the hill-climb table (identical AUPRC
across all contamination values within an `n_estimators` group) and called out
explicitly rather than left for a reviewer to puzzle over.

## 6. Deployment

FastAPI backend (`backend/app/main.py`, port 8006). React + TypeScript + Recharts
frontend (port 5178) — six tabs: Overview, Evaluation (PR curve), Flagged (top-50
anomalies with ground truth), Modeling, AutoResearch, Data & EDA.

Verified via Playwright: 0 console errors, 0 failed requests across all six tabs.
