# CRISP-DM Phase 1: Business Understanding

## Project

**Autonomous Anomaly Detection Platform** — unsupervised fraud detection on the
[Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(Machine Learning Group, ULB — 284,807 European-cardholder transactions, Sept. 2013,
492 confirmed frauds, 0.173% positive rate). This is *the* canonical anomaly-detection
dataset in the field — nearly every anomaly-detection tutorial, Kaggle kernel, and
several published papers (e.g. Dal Pozzolo et al. 2015, "Calibrating Probability with
Undersampling for Unbalanced Classification") use this exact data. Sourced here via
`sklearn.datasets.fetch_openml("creditcard")`, OpenML's public mirror of the identical
Kaggle dataset — no Kaggle API credentials required, same substitution pattern used in
every prior project in this repo.

## Business Objective

A payment processor needs to flag fraudulent transactions in near-real-time. Labels
exist here (it's a benchmark dataset) but a production fraud system cannot assume
labels will keep arriving fast enough, or that fraud patterns won't drift — so the
actual deployed technique must be **unsupervised or semi-supervised anomaly detection**,
scored against the *known* labels here as a validation exercise (this is standard
practice in the anomaly-detection literature: train unsupervised, evaluate against
held-out ground truth when it's available, exactly as Dal Pozzolo et al. do).

## Data Mining Objective

Compare the field's standard unsupervised anomaly-detection methods — **Isolation
Forest** (Liu, Ting & Zhou 2008), **Local Outlier Factor** (Breunig et al. 2000),
**One-Class SVM** (Schölkopf et al. 2001), and **Elliptic Envelope** (a Gaussian-based
baseline) — on this dataset's known extreme class imbalance (0.173% fraud). Because
labels exist, success is measured properly: **AUPRC (area under precision-recall
curve)**, not accuracy or plain ROC-AUC, which are both known to be misleadingly
optimistic under this level of class imbalance (a trivial "always predict normal"
classifier gets 99.83% accuracy and can score deceptively well on ROC-AUC too) — this
is the exact evaluation-metric caution the original paper itself raises.

## Scope & Constraints

- Features `V1`–`V28` are already PCA-transformed by the dataset's original authors
  for confidentiality — the underlying raw features are not available, and no attempt
  is made to reverse-engineer them. `Amount` is the only non-anonymized feature column
  in the OpenML mirror used here (the original Kaggle listing also includes a `Time`
  column, seconds-since-first-transaction; OpenML's `creditcard` version omits it —
  noted here rather than silently working around the gap, since it means no
  time-of-day/velocity features are available in this analysis).
- The dataset's `Class` label is used **only for evaluation**, never as a training
  feature — using it during model fitting would defeat the entire point of testing
  unsupervised methods and silently turn this into supervised classification.
- Following the pattern from Projects 01–04: full CRISP-DM lifecycle as executable
  scripts, an AutoResearch hill-climbing phase (contamination-rate + algorithm search,
  matched against the founding papers' reported operating points), and a
  data-scientist-facing admin dashboard.
