# Design Doc — AutoGluon Multi-Layer Stacking Platform

## 1. Overview

Illustrates AutoML via AutoGluon Tabular across two distinct data science tasks —
regression (California Housing) and binary classification (Adult Census Income) —
comparing AutoGluon's `best_quality` multi-layer stacking preset against a hand-tuned
single-model LightGBM baseline. See
[docs/01_business_understanding.md](./docs/01_business_understanding.md).

## 2. Pipeline

| Script | CRISP-DM Phase | Output |
|---|---|---|
| `01_prepare_data.py` | Data Understanding + Preparation | train/test splits for both tasks |
| `02_train_models.py` | Modeling | AutoGluon `best_quality` + LightGBM baseline, both tasks |
| `03_evaluate.py` | Evaluation | improvement %, validation-vs-test selection integrity check |

## 3. Datasets

- **California Housing** (regression): sklearn's built-in copy of Pace & Barry (1997),
  20,640 rows, 8 features, no missing values.
- **Adult Census Income** (binary classification): via `fetch_openml("adult")`,
  OpenML's mirror of the Kaggle/UCI dataset, 48,842 rows, 14 features (mixed
  numeric/categorical, 6,465 missing values handled natively by both AutoGluon and
  LightGBM's categorical support — no manual imputation needed for either model).

## 4. Modeling

Both AutoGluon runs use `presets="best_quality"` with `time_limit=120` seconds
(deliberately capped, stated as a real constraint in `01_business_understanding.md`).
The baseline is a LightGBM model hill-climbed over a small hyperparameter grid — the
same greedy-search AutoResearch pattern used in every prior project, here serving as
AutoGluon's control group rather than AutoGluon replacing it.

**Note on installed base learners**: CatBoost was installed specifically for this run
(AutoGluon's ensemble uses it as one of its stacked base models). FastAI (neural-net
base learner) was NOT installed — a heavier torch-based dependency not worth adding for
this project's time budget — so the stacking ensemble here is tree-based-only
(LightGBM, XGBoost, CatBoost, Random Forest, Extra Trees variants across two stack
levels), not the full model-family diversity AutoGluon supports with every optional
extra installed. Stated explicitly rather than silently working around the gap.

| Task | Baseline | AutoGluon | Improvement |
|---|---:|---:|---:|
| Housing (RMSE, lower better) | 0.4393 | 0.4301 | +2.09% |
| Adult (ROC-AUC, higher better) | 0.9301 | 0.9302 | +0.01% |

## 5. Key Decision: Validation Selection vs. Test-Set Reality

**The project's central finding.** On **both** tasks, AutoGluon's internally-selected
best model (`WeightedEnsemble_L3` in both cases, chosen by out-of-fold validation
score) was **not** the model that actually scored best on the true held-out test set:

- Housing: selected `WeightedEnsemble_L3` (test RMSE 0.4301); the level-1
  `LightGBM_BAG_L1` single model actually scored better (test RMSE 0.4281).
- Adult: selected `WeightedEnsemble_L3` (test ROC-AUC 0.9302); `LightGBMXT_BAG_L2`
  actually scored better (test ROC-AUC 0.9311).

This is a genuine, consistent (2-for-2) methodology finding, not a bug — reported
transparently on the dashboard's per-task tabs rather than hidden behind the headline
"AutoGluon wins" framing. It's a real, useful lesson for a data science team: a
stacking ensemble's validation-based selection is not a guarantee of best test
performance, especially at a modest sample size and a capped compute budget — a
production deployment should hold out its own final test set and verify the selected
model there before shipping, not trust the AutoML tool's internal ranking blindly.

## 6. Deployment

FastAPI backend (`backend/app/main.py`, port 8007). React + TypeScript + Recharts
frontend (port 5179) — four tabs: Overview, Housing (Regression), Adult
(Classification), Data & EDA.

Verified via Playwright: 0 console errors, 0 failed requests across all four tabs.
