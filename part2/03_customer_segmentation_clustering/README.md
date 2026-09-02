# Customer Intelligence & Segmentation Clustering

Unsupervised customer segmentation on the UCI/Kaggle **Online Retail** dataset (541,909
transactions from a UK-based online gift retailer, Dec 2010–Dec 2011) via RFM feature
engineering + K-Means, following the full CRISP-DM lifecycle with an AutoResearch
hill-climbing search and a data-scientist-facing admin dashboard.

## Result

3 customer segments found (AutoResearch-selected, silhouette 0.409):

| Segment | % Customers | % Revenue | Avg Recency | Avg Frequency |
|---|---:|---:|---:|---:|
| **Loyal High-Value** | 40.8% | 81.7% | 34d | 9.5 orders |
| **New / Occasional** | 37.0% | 13.6% | 56d | 2.2 orders |
| **At-Risk / Dormant** | 22.2% | 4.8% | 258d | 1.7 orders |

A classic Pareto concentration (top segment: 41% of customers, 82% of revenue) — the kind
of pattern a marketing team can act on directly (retention campaigns for At-Risk, VIP
treatment for Loyal High-Value), unlike the statistically "best-separated" but
uninterpretable k=2 split that naive silhouette maximization finds (see
`RESEARCH_REPORT.md §4` for the full honest comparison).

## Running it

```bash
# Data pipeline (run once)
cd scripts
python 01_eda.py && python 02_prepare_data.py && python 03_train_models.py
python 04_autoresearch.py && python 05_evaluate.py

# Backend (port 8003)
cd ../backend && python -m uvicorn app.main:app --port 8003

# Frontend (port 5176)
cd ../frontend && npm install && npm run dev -- --port 5176
```

Full design decisions in [DESIGN_DOC.md](./DESIGN_DOC.md); the complete research writeup —
literature grounding, AutoResearch findings, and the honest k=2-vs-k=3 discussion — is in
[RESEARCH_REPORT.md](./RESEARCH_REPORT.md).
