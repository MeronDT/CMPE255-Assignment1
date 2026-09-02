# CRISP-DM Phase 1: Business Understanding

## Project

**AutoGluon Multi-Layer Stacking Platform** — illustrates AutoML via
[AutoGluon Tabular](https://auto.gluon.ai/) (Erickson et al., 2020, "AutoGluon-Tabular:
Robust and Accurate AutoML for Structured Data") across **two distinct data science
tasks** — regression and binary classification — to demonstrate AutoGluon's
task-agnostic multi-layer stacking-ensemble approach, not just a single dataset.

## Datasets

1. **California Housing** (regression) — sklearn's built-in copy of the classic Pace &
   Barry (1997) dataset, also widely mirrored on Kaggle (`camnugent/california-housing-prices`).
   20,640 census-block records; predict median house value.
2. **Adult Census Income** (binary classification) — the UCI/Kaggle "Adult" dataset
   (Kohavi 1996), one of the most-used binary classification benchmarks in the ML
   literature and a common Kaggle dataset (`uciml/adult-census-income`). ~48,842
   records; predict whether income exceeds $50K/year. Sourced via
   `sklearn.datasets.fetch_openml("adult")`, OpenML's public mirror — no Kaggle
   credentials required, same substitution pattern as every prior project here.

## Business Objective

AutoML platforms like AutoGluon promise "good performance with minimal manual tuning"
across *any* tabular task. The business case for a data science team: is that promise
real, and by how much does AutoGluon's headline technique — multi-layer model
stacking, where a second-layer meta-model learns to combine first-layer base models'
out-of-fold predictions — actually beat a single well-configured baseline model?

## Data Mining Objective

For each task, compare **AutoGluon's `best_quality` preset** (multi-layer stacking +
bagging ensemble of LightGBM, CatBoost, XGBoost, Random Forest, and neural-net base
learners) against a **single-model baseline** (LightGBM, hand-tuned via the same
greedy hill-climbing search pattern used in Projects 01/03/04/06's AutoResearch phases)
on held-out test data. Regression is scored by RMSE and R²; classification by ROC-AUC
and accuracy — the standard metrics for each task type.

## Scope & Constraints

- AutoGluon's `time_limit` is deliberately capped (120 seconds per task) to fit this
  project's time budget — AutoGluon's own documentation is explicit that quality
  improves with more time budget, so this is a real, stated constraint on the results,
  not a hidden one.
- Following the pattern from Projects 01–06: full CRISP-DM lifecycle as executable
  scripts, an "AutoResearch" phase (here, AutoGluon's own internal
  stacking/bagging search *is* the AutoResearch, compared against a manual
  hill-climbed baseline as the control group), and a data-scientist-facing admin
  dashboard.
