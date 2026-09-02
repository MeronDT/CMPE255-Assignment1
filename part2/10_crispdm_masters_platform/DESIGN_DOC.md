# Design Doc — CRISP-DM Master's Data Science Platform

## 1. Overview

One dataset (UCI/Kaggle Online Retail, reused deliberately from Projects 03/04) taken
through five distinct data mining paradigms — clustering, anomaly detection, supervised
learning, association rule mining, and LSH sub-linear search — under one shared CRISP-DM
lifecycle. See [docs/01_business_understanding.md](./docs/01_business_understanding.md)
for why one dataset, and the leakage/labeling caveats stated up front.

## 2. Pipeline

| Script | Produces |
|---|---|
| `01_eda_and_prepare.py` | Shared cleaning + 3 feature tables (full-history RFM+, early-window supervised features, basket matrix) |
| `02_clustering.py` | Business-constrained K-Means segmentation |
| `03_anomaly_detection.py` | Isolation Forest + synthetic-outlier validation |
| `04_supervised_learning.py` | 3-model comparison predicting future high-value customers |
| `05_association_rules.py` | Business-constrained FP-Growth rule mining |
| `06_lsh_search.py` | MinHash LSH approximate similar-customer search |
| `07_synthesis.py` | Cross-technique conclusion, pulling every prior script's key result together |

## 3. Key Decisions Per Technique

- **Clustering**: applies the k∈[3,8] business constraint from Project 03's finding
  *from the start*, rather than rediscovering the k=2 trap from scratch.
- **Anomaly detection**: this dataset has **no natural anomaly labels** (unlike
  Project 06's Credit Card Fraud data) — addressed by injecting 15 synthetic extreme
  outliers and checking they rank in the top 5% by anomaly score (mean percentile 98.9,
  all passed), explicitly scoped as a sanity check, not a precision/recall claim.
- **Supervised learning**: predicts "becomes high-value" from only the first 90 days
  of a customer's behavior — a genuine early-prediction task. The feature/label overlap
  (41.5% of lifetime spend occurs within that 90-day window on average) is quantified
  and discussed explicitly rather than left as an implicit leakage risk.
- **Association rules**: applies Project 04's business-constrained rule-count selection
  (5–50 reviewable rules, ranked by avg lift) from the start.
- **LSH**: the one genuinely new technique. An initial threshold (0.25) gave only 28%
  recall@5 — diagnosed (not assumed correct) by inspecting actual brute-force
  similarity values, which showed true top-5 neighbors mostly scoring 0.05–0.25 Jaccard
  in this sparse product-set data. Recalibrated to threshold=0.08, fixing recall@5 to
  97% while keeping a 177.6x speedup over brute-force — a real empirical calibration
  process shown, not a lucky first guess.

## 4. The Cross-Cutting Finding

**Every one of the five techniques hit the same class of pitfall at least once**: an
unconstrained or default-parameter objective finding a technically-valid but
practically-useless or miscalibrated answer. Clustering's k=2 trap, association rules'
7,799-rule trap (both inherited from Projects 03/04's findings and designed around
here), and LSH's threshold miscalibration (discovered fresh in this project) are the
same underlying failure mode wearing different clothes. The fix was identical every
time: state an explicit constraint grounded in the business objective or the data's
actual distribution, apply it *before* optimizing, then verify the result holds up.
This discipline — not any single technique's headline number — is this capstone's
actual thesis, stated explicitly in `07_synthesis.py`'s output and the dashboard's
Conclusion tab.

## 5. Deployment

FastAPI backend (`backend/app/main.py`, port 8010). React + TypeScript + Recharts
frontend (port 5182) — 8 tabs: Overview, Data & EDA, and one tab per technique (each
ending in a quiz), plus a closing Conclusion/Synthesis tab.

Verified via Playwright: 0 console errors, 0 failed requests across all 8 tabs.
