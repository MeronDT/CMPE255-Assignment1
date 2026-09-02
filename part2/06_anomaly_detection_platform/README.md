# Autonomous Anomaly Detection Platform

Unsupervised fraud detection on the Kaggle Credit Card Fraud dataset (275,663 clean
transactions, 473 known frauds, 0.17% positive rate) via Isolation Forest, following the
full CRISP-DM lifecycle with an AutoResearch hill-climbing search and a
data-scientist-facing admin dashboard.

## Result

**Isolation Forest** wins a 4-algorithm tournament (AUPRC 0.607 vs. LOF's near-random
0.028). At full scale: **84.2x lift over random baseline**, catching 114/473 (24.1%)
of actual frauds at 24.1% precision when flagging the top 473 most-anomalous
transactions — realistic numbers for a purely unsupervised method on 0.17%-imbalanced
data, not inflated. See `RESEARCH_REPORT.md §4` for the project's central finding: why
raw AUPRC dropping from proxy (0.62) to full scale (0.14) looks like a regression but
isn't (base-rate dependence of the metric, not model degradation).

## Running it

```bash
cd scripts
python 01_eda.py && python 02_prepare_data.py && python 03_train_models.py
python 04_autoresearch.py && python 05_evaluate.py

cd ../backend && python -m uvicorn app.main:app --port 8006
cd ../frontend && npm install && npm run dev -- --port 5178
```

Full design decisions in [DESIGN_DOC.md](./DESIGN_DOC.md); the complete research
writeup is in [RESEARCH_REPORT.md](./RESEARCH_REPORT.md).
