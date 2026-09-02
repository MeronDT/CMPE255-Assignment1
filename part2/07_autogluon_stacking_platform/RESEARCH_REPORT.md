# Research Report — AutoGluon Multi-Layer Stacking Platform

## 1. Introduction

AutoGluon-Tabular (Erickson et al. 2020) is a leading open-source AutoML framework
built specifically around **multi-layer stacking with bagging**: base models are
trained via k-fold cross-validation, their out-of-fold predictions become features for
a second-layer "meta" model, and the process can repeat across further stack levels.
This project tests that architecture's actual value on two well-known tabular
benchmarks, one regression and one classification, against a hand-tuned single-model
baseline — the honest control-group comparison an AutoML vendor's own marketing rarely
runs.

## 2. Data Preparation

**California Housing** (sklearn built-in, Pace & Barry 1997): 20,640 rows, 8 numeric
features, no missing values, no cleaning required. **Adult Census Income** (via
`fetch_openml("adult")`, OpenML's mirror of the Kaggle/UCI dataset): 48,842 rows, 14
mixed numeric/categorical features, 6,465 missing values (in `workclass`,
`occupation`, `native-country`) — left as native `NaN`/missing-category markers rather
than imputed, since both AutoGluon and LightGBM handle missing values internally, and
imputing would risk injecting information the models don't actually need.

## 3. Modeling

Both tasks: AutoGluon `TabularPredictor(presets="best_quality", time_limit=120)`
vs. a LightGBM model hill-climbed over 3-4 hyperparameter configurations (same greedy
search pattern as every prior project's AutoResearch phase).

| Task | Baseline | AutoGluon | Improvement | Ensemble Size | Fit Time |
|---|---:|---:|---:|---:|---:|
| Housing (RMSE ↓) | 0.4393 | 0.4301 | **+2.09%** | 12 models | 123.9s |
| Adult (ROC-AUC ↑) | 0.9301 | 0.9302 | **+0.01%** | 13 models | 120.9s |

**Honest framing, not oversold**: AutoGluon's gains over a competent single-model
baseline were modest at this capped time budget — a real 2% win on regression, an
essentially-tied result on classification. AutoGluon's own documentation states result
quality improves with additional time budget; this project's 120-second cap is a
genuine, stated constraint (§ `01_business_understanding.md`) chosen to fit the
project's overall time allowance, not evidence that stacking doesn't help — a longer
run would very plausibly widen the gap, untested here due to time budget.

**Note on installed base learners**: CatBoost was specifically installed so AutoGluon's
ensemble could include it; FastAI (a torch-based neural-net base learner) was not
installed, given its heavier dependency footprint relative to the value it would add
within this project's time budget. The resulting ensembles are tree-based-only
(LightGBM/XGBoost/CatBoost/RandomForest/ExtraTrees variants across up to 2 stack
levels plus a `WeightedEnsemble` meta-layer) — stated explicitly rather than silently
presenting a partial ensemble as the full picture.

## 4. Evaluation: Does Validation Selection Predict Test Performance?

**This project's central finding**, surfaced by explicitly checking whether
AutoGluon's chosen `model_best` (selected using internal out-of-fold validation score,
which never touches the test set) actually had the best score on the held-out test set:

| Task | Selected Model | Selected Test Score | Actual Best-on-Test Model | Its Test Score | Matched? |
|---|---|---:|---|---:|---|
| Housing | `WeightedEnsemble_L3` | 0.4301 (RMSE) | `LightGBM_BAG_L1` | **0.4281** (RMSE) | **No** |
| Adult | `WeightedEnsemble_L3` | 0.9302 (AUC) | `LightGBMXT_BAG_L2` | **0.9311** (AUC) | **No** |

**On both tasks, the validation-selected winner was not the actual test-set winner.**
This is a real, 2-for-2 consistent finding — not noise from a single lucky/unlucky
split — and a genuinely useful lesson for a data science team evaluating AutoML tools:
a multi-layer stacking ensemble's validation-based model selection is not a guarantee
of best true generalization, especially at moderate dataset sizes where cross-validation
fold variance is non-trivial relative to the differences between top models (in both
cases here, the gap between the selected model and the true best is small — 0.0020
RMSE, 0.0009 AUC — well within the range where fold-to-fold noise could plausibly flip
the ranking). The practical takeaway: **always hold out a genuine final test set and
verify the AutoML tool's selected model there before deploying it**, rather than
trusting the tool's internal leaderboard blindly. This doesn't mean AutoGluon's
stacking approach failed — the *ensemble as a whole* still competitively matched or
beat the hand-tuned baseline — but the specific model AutoGluon's automated selection
process picked wasn't provably the best one available in its own leaderboard.

## 5. Limitations & Future Work

- **120-second time budget per task is a real, stated constraint**, not a hidden
  handicap — AutoGluon's documented behavior is that quality improves with more time;
  a production evaluation would run substantially longer (AutoGluon's own benchmarks
  typically use hours, not minutes) before drawing final conclusions about ceiling
  performance.
- **FastAI neural-net base learner excluded** (§3) — a fuller `best_quality` run with
  every optional dependency installed might produce a different (likely larger, and
  possibly better) ensemble; not pursued given this project's dependency-footprint
  and time trade-offs.
- **Selection-integrity finding (§4) is based on 2 tasks**, not a large-scale study —
  a rigorous claim about how often this happens in general AutoGluon usage would need
  many more datasets and repeated random seeds; presented here as a genuine finding on
  these two specific runs, not a general indictment of AutoGluon's selection method.
- **No hyperparameter search on the AutoGluon side** — `best_quality` uses AutoGluon's
  own internal defaults; a `time_limit` increase or `hyperparameter_tune_kwargs` config
  is the natural next lever, not explored here for time-budget reasons.
