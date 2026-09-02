# CRISP-DM Master's Data Science Platform

A textbook-quality, end-to-end walkthrough of every major data mining paradigm — on
**one dataset** (UCI/Kaggle Online Retail, reused from Projects 03/04): clustering,
anomaly detection, supervised learning, association rule mining, and LSH sub-linear
search — all under one CRISP-DM lifecycle, with quizzes throughout and a closing
synthesis phase.

## Results at a Glance

| Technique | Result |
|---|---|
| Clustering | k=3 segments, silhouette 0.409 (business-constrained) |
| Anomaly Detection | 15 synthetic outliers, mean percentile rank 98.9 (validation, not a fraud-label evaluation — this data has none) |
| Supervised Learning | ROC-AUC 0.905 predicting future high-value customers from first-90-day behavior |
| Association Rules | 31 business-constrained rules, avg lift 13.4x |
| LSH Search | 177.6x faster than brute-force, 97% recall@5 (after recalibrating threshold — see `RESEARCH_REPORT.md`) |

**The cross-cutting finding**: every technique hit the same class of pitfall at least
once — an unconstrained/default objective finding a technically-valid but
practically-useless answer — and every fix was the same: apply an explicit,
data-or-business-grounded constraint before optimizing, then verify. See
`RESEARCH_REPORT.md §5` and the dashboard's Conclusion tab.

## Running it

```bash
cd scripts
python 01_eda_and_prepare.py
python 02_clustering.py && python 03_anomaly_detection.py
python 04_supervised_learning.py && python 05_association_rules.py
python 06_lsh_search.py && python 07_synthesis.py

cd ../backend && python -m uvicorn app.main:app --port 8010
cd ../frontend && npm install && npm run dev -- --port 5182
```

Full design decisions in [DESIGN_DOC.md](./DESIGN_DOC.md); the complete research
writeup is in [RESEARCH_REPORT.md](./RESEARCH_REPORT.md).
