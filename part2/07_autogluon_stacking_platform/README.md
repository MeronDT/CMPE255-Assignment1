# AutoGluon Multi-Layer Stacking Platform

Illustrates AutoML via AutoGluon Tabular across two data science tasks — regression
(California Housing) and binary classification (Adult Census Income) — comparing
AutoGluon's multi-layer stacking ensemble against a hand-tuned single-model baseline,
following the full CRISP-DM lifecycle with a data-scientist-facing admin dashboard.

## Result

AutoGluon's `best_quality` preset beat a hand-tuned LightGBM baseline by **2.09%**
(RMSE) on regression and essentially tied (**+0.01%**, ROC-AUC) on classification, at
a capped 120-second-per-task time budget.

**Central finding**: on both tasks, AutoGluon's internally-selected best model (always
the top-level `WeightedEnsemble_L3`, chosen by validation score) was **not** the model
that actually scored best on the true held-out test set — a genuine, 2-for-2 consistent
methodology finding, not a bug. See `RESEARCH_REPORT.md §4` for the full breakdown.

## Running it

```bash
cd scripts
python 01_prepare_data.py && python 02_train_models.py && python 03_evaluate.py

cd ../backend && python -m uvicorn app.main:app --port 8007
cd ../frontend && npm install && npm run dev -- --port 5179
```

Full design decisions in [DESIGN_DOC.md](./DESIGN_DOC.md); the complete research
writeup is in [RESEARCH_REPORT.md](./RESEARCH_REPORT.md).
