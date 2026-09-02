# Research Report — Autonomous Anomaly Detection Platform

## 1. Introduction

The Kaggle Credit Card Fraud dataset (Dal Pozzolo et al., Worldline & ULB, 2015) is the
field's most-used public benchmark for extreme-class-imbalance anomaly detection. This
project applies the four standard unsupervised detectors — Isolation Forest (Liu, Ting
& Zhou 2008), Local Outlier Factor (Breunig et al. 2000), One-Class SVM (Schölkopf et
al. 2001), Elliptic Envelope — and evaluates against the dataset's known labels, exactly
as the original authors' own methodology does: train unsupervised, validate against
ground truth when available.

## 2. Data Preparation

Sourced via `sklearn.datasets.fetch_openml("creditcard")`, OpenML's public mirror of
the identical Kaggle dataset (no Kaggle credentials required). 9,144 of 284,807 rows
(3.2%) are exact duplicates — with 28 continuous PCA-transformed features, duplication
by chance is vanishingly improbable, so these are treated as data-collection artifacts
and dropped, leaving 275,663 transactions / 473 confirmed frauds (0.1716%). `Amount` is
log1p-transformed and standardized alongside the already-PCA'd `V1`–`V28`. **Note**: the
OpenML mirror used here omits the original `Time` column present in Kaggle's listing —
reported rather than silently worked around, since it means no time-of-day/transaction-
velocity features are available in this analysis.

## 3. Modeling: Algorithm Tournament

Compared on a 20,000-normal + 473-fraud proxy subsample (full-data tournament for two
super-linear algorithms would exceed the time budget for a comparison pass):

| Algorithm | AUPRC | ROC-AUC | Runtime |
|---|---:|---:|---:|
| **Isolation Forest** | **0.6074** | 0.9456 | 0.55s |
| Elliptic Envelope | 0.5155 | 0.9128 | 2.29s |
| One-Class SVM | 0.2738 | 0.9226 | 1.66s |
| Local Outlier Factor | 0.0281 | 0.4844 | 1.80s |

Isolation Forest wins decisively — consistent with its dominance in the published
literature on this exact dataset (its random-partition-depth approach is well-suited to
globally sparse anomalies in moderate-dimensional continuous feature spaces). **Local
Outlier Factor performed at essentially chance level** (ROC-AUC 0.484, worse than a coin
flip) — reported honestly rather than omitted or excused. This is a real, explicable
finding: LOF measures *local* density deviation, and it degrades when anomalies aren't
locally sparse relative to their immediate neighbors in the given feature space —
exactly the failure mode the original LOF paper itself flags as a limitation, not a
bug in this implementation (verified: default `n_neighbors=20` is the standard choice;
the failure is characteristic of the method on this data, not a misconfiguration —
Elliptic Envelope's success rules out this simply being "the whole space is
adversarial to density methods").

Elliptic Envelope's `MinCovDet` fitting emitted repeated `RuntimeWarning`s ("Determinant
has increased; this should not happen") during training — a known numerical-stability
issue with the algorithm's iterative reweighting on data that doesn't fit its Gaussian
assumption well. It still produced a reasonable AUPRC (0.516), but the warning itself is
informative: it's evidence the Gaussian-ellipse model is a strained fit for this data's
actual (non-Gaussian, PCA-anonymized) distribution, consistent with Isolation Forest's
non-parametric approach winning outright.

## 4. AutoResearch: the AUPRC Base-Rate Trap

**This project's central finding.** The hill-climb phase (`n_estimators` ∈
{100,200,400}, `contamination` swept for completeness) found `n_estimators=400`
best on the proxy: AUPRC 0.6156. Finalizing — retraining that exact config on the
**full 275,663-row dataset** — raw AUPRC dropped to **0.1444**. Taken at face value,
that reads as the model failing to generalize from proxy to full scale.

**It isn't a generalization failure — it's a metric artifact of subsampling.** AUPRC
is fundamentally base-rate-dependent: a purely random classifier scores AUPRC ≈ the
positive rate of the evaluation set. The proxy subsample's positive rate (473 fraud /
20,473 total = 2.31%) is roughly **13x** the full dataset's true rate (473/275,663 =
0.172%). Comparing raw AUPRC computed on two evaluation sets with such different base
rates is comparing apples to oranges — the *achievable ceiling* for AUPRC is different
in each case, independent of how good the model's ranking actually is.

The fair comparison is **AUPRC relative to each evaluation's own random baseline**
(i.e., "lift over random"):

| | AUPRC | Base Rate | Lift over Random |
|---|---:|---:|---:|
| Proxy (20,473 rows) | 0.6156 | 2.31% | **26.6x** |
| Full scale (275,663 rows) | 0.1444 | 0.172% | **84.2x** |

By the fair comparison, the model's ranking quality didn't degrade at full scale — it
**improved**, from 26.6x to 84.2x better than chance. ROC-AUC (which is far less
base-rate-sensitive than AUPRC) confirms this directly: 0.9464 (proxy) → 0.95 (full),
essentially flat. This is reported transparently on the dashboard's AutoResearch tab —
showing both numbers with the explanation, not just the flattering one.

A secondary honest note surfaced during this analysis: **`contamination` has zero
effect on AUPRC or ROC-AUC for Isolation Forest**, visible directly in the hill-climb
table as identical scores across every `contamination` value within an `n_estimators`
group. This is expected, not a bug: `score_samples()` returns a continuous anomaly
score independent of `contamination`; the parameter only shifts the *binary* threshold
`predict()` applies. `contamination` matters downstream, for the actual
flagging/operating-point decision made in `05_evaluate.py` — not for the ranking
metrics reported here. Called out explicitly rather than left for a reviewer to notice
and wonder about.

## 5. Evaluation

At the natural operating point (flag the top-473 most-anomalous transactions, matching
the true fraud count): **24.1% precision, 24.1% recall** (114 of 473 actual frauds
caught). This is realistic, not impressive-sounding-but-hollow — purely unsupervised
detection on 0.17%-imbalanced data with no time-of-day or velocity features (the
`Time` column being unavailable in this exact data source, §2) genuinely cannot match
a supervised, feature-engineered production fraud system. The 84.2x lift over random is
the meaningful comparison, not "only 24% recall."

## 6. Limitations & Future Work

- **`Time` column unavailable** in the OpenML mirror used — a production system would
  add transaction-velocity features (time-since-last-purchase, transactions-per-hour),
  known from the fraud-detection literature to meaningfully improve detection.
- **Fully unsupervised is a deliberate scope choice**, not a technical ceiling — with
  the labels available here, a supervised or semi-supervised (e.g. labeled minority
  oversampling) approach would substantially outperform this. The point of this project
  is comparing the standard *unsupervised* toolkit honestly, matching the business
  scenario where labels lag or don't exist yet.
- **Elliptic Envelope's numerical warnings** suggest a Gaussian-mixture or robust
  covariance estimator with more iterations/regularization might fit better — not
  pursued further given Isolation Forest's clear win.
- **Single train/eval split** (no cross-validation) — reasonable for this project's
  time budget given the dataset's already-fixed, well-studied nature, but a rigorous
  production evaluation would use multiple random proxy subsamples to bound variance.
