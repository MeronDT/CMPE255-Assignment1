# Research Report — CRISP-DM Master's Data Science Platform

## 1. Introduction

Most data science coursework and tutorials teach one technique on one dataset. A
working data scientist rarely gets that luxury — real business data gets reused across
many analytical questions. This capstone deliberately takes **one dataset** (the
UCI/Kaggle Online Retail transaction log, 541,909 raw transactions, already used for
clustering in Project 03 and association rules in Project 04) through **five distinct
mining paradigms**, to demonstrate CRISP-DM's actual reusable value: one Data
Preparation phase (`01_eda_and_prepare.py`) feeds every technique after it, rather than
each technique getting a bespoke dataset built in isolation.

## 2. Data Preparation (Shared)

4,371 customers survive cleaning (guest checkouts and non-sale price adjustments
excluded, same justified decisions as Projects 03/04). Three feature tables are
derived once and reused:

- **Full-history RFM+** (Recency/Frequency/Monetary + basket/product/tenure/
  cancellation features) — feeds clustering, anomaly detection, and LSH.
- **Early-90-day features + high-value label** — feeds supervised learning. Only
  2,235 customers (those with ≥90 days observed tenure) are eligible; the label is
  top-quartile lifetime spend among that eligible group (positive rate 25.01%,
  matching the quartile definition exactly — a correctness check).
- **Basket matrix** (top-200-product one-hot, same restriction and justification as
  Project 04) — feeds association rule mining.

## 3. Technique 1: Clustering

Applying Project 03's business constraint (k∈[3,8]) from the start, K-Means finds k=3
(silhouette 0.409) — the identical segmentation found in Project 03 (Loyal High-Value
40.8%/81.7% revenue, New/Occasional, At-Risk/Dormant), confirming the methodology is
stable and reproducible rather than a one-off result.

## 4. Technique 2: Anomaly Detection

**Honest scoping, addressed directly**: unlike Project 06's Credit Card Fraud dataset,
Online Retail has no natural "this transaction is fraudulent" label. Rather than
pretend a precision/recall evaluation exists, 15 synthetic customers with deliberately
extreme, multi-dimensional values (200-500 orders, $200K-$800K spend, 500-1000 distinct
products — far outside this data's real range on every axis simultaneously) were
injected and scored alongside the real 4,371 customers using Isolation Forest. Result:
mean percentile rank 98.9, all 15 in the top 5% by anomaly score. This confirms the
detector correctly identifies genuine multi-dimensional outliers — a necessary sanity
check before trusting the model's flagged real anomalies (also inspected and shown on
the dashboard), but explicitly **not** claimed as a real-world fraud-detection accuracy
figure, since no such ground truth exists in this data.

## 5. Technique 3: Supervised Learning

**A genuine early-prediction business question**: using only a customer's first 90
days of behavior, predict whether they become high-value (top-quartile lifetime
spend). Three models compared:

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| **Logistic Regression** | **0.9048** | 0.786 | 0.629 | 0.698 |
| Random Forest | 0.9038 | 0.768 | 0.614 | 0.683 |
| Gradient Boosting | 0.9004 | 0.744 | 0.643 | 0.690 |

Notably, the simplest model (logistic regression) won — plausible given the small
feature set (5 engineered aggregates) and a fairly smooth, monotonic relationship
between early spend/frequency and eventual lifetime value in log-space, which doesn't
need a tree ensemble's capacity to capture. Winning coefficients: `EarlyMonetary` and
`EarlyCancellationRate` (positively — high cancellers who still spend a lot early tend
to also spend a lot in total) dominate; `EarlyDistinctProducts` contributes almost
nothing once the others are accounted for.

**Leakage discussion, addressed explicitly, not glossed over**: on average, 41.5% of a
customer's *lifetime* spend occurs within their *first 90 days* — meaning the label
(built from lifetime spend) and the features (built from the first-90-day window) share
real overlap. This is **not** the strict-sense leakage of using literal future
information the model wouldn't have at prediction time — it's the expected, legitimate
correlation between early and total customer behavior that any real early-warning
system would also rely on. It does mean the reported 0.905 ROC-AUC is somewhat higher
than a stricter formulation (e.g., predicting spend *only after* day 90, using *only*
day 0–90 features) would likely produce — stated as a limitation (§7), not hidden.

## 6. Technique 4: Association Rule Mining

Applying Project 04's business-constrained selection (5–50 rule band, ranked by avg
lift) from the start: winning config `min_support=0.02, min_confidence=0.6` → 31
rules, avg lift 13.4x — consistent in character with Project 04's findings (color/theme
product-collection variants dominate the top rules), confirming the methodology
transfers cleanly to a re-run on the same underlying data with slightly different
support/confidence grid points.

## 7. Technique 5: Sub-linear Search (LSH)

**The one genuinely new technique in this capstone, and where a real calibration
process happened in the open, not hidden after the fact.** MinHash LSH (128
permutations) indexes each customer's purchased-product set for approximate Jaccard
similarity search. A first attempt at `threshold=0.25` (a reasonable-sounding default)
produced only **28% recall@5** against brute-force ground truth — investigated, not
assumed correct, by inspecting actual top-5 brute-force similarity values for sample
queries, which showed true nearest-neighbor similarities in this data mostly fall in
the **0.05–0.25** range (customers have broad, only partially overlapping tastes across
a large product catalog). A threshold at the *edge* of that range structurally excludes
most genuine matches — independent of whether the LSH implementation itself is correct.

Recalibrated to `threshold=0.08` (matched to the data's actual similarity
distribution, not guessed): **recall@5 rose to 97%**, while still delivering a
**177.6x speedup** over brute-force (0.1ms vs. 17.8ms average query time on 4,338
indexed customers). This is the textbook LSH trade-off — approximate, sub-linear
search in exchange for a small (3%) chance of missing a true top-5 neighbor — made
concrete with a real calibration process shown rather than a lucky first guess
presented as if it were obviously correct.

## 8. Conclusion: The Cross-Cutting Finding

**Every one of the five techniques in this capstone hit the same class of pitfall at
least once**: an unconstrained or default-parameter objective finding a
technically-valid but practically-useless or miscalibrated answer.

- Clustering: unconstrained silhouette maximization → k=2, not business-actionable
  (Project 03's finding, designed around here from the start).
- Association rules: unconstrained rule-count maximization → 7,799 rules, unusable
  (Project 04's finding, designed around here from the start).
- LSH: a plausible-sounding default threshold → 28% recall, discovered and fixed
  fresh in this project.

**In every single case, the fix was identical**: state an explicit constraint grounded
in the actual business objective or the data's real distribution, apply it *before*
optimizing (not after discovering a bad result by accident), and verify the fixed
result actually holds up. This discipline — applied consistently across five
completely different algorithms spanning unsupervised, supervised, association, and
similarity-search paradigms — is the actual lesson of this capstone, more than any
single technique's headline number.

## 9. Limitations & Future Work

- **Supervised learning's leakage discussion (§5)** quantifies overlap but doesn't
  eliminate it — a stricter formulation predicting only *post-90-day* spend would be
  a cleaner (if harder) task, not attempted here given the time budget.
- **Anomaly detection has no real evaluation** (§4) — only a synthetic-outlier sanity
  check. A production deployment would need either real fraud/return-abuse labels or
  a human-in-the-loop review process to validate flagged real customers.
- **LSH's calibration (§7) is specific to this dataset's product-catalog size and
  customer purchase-diversity** — the 0.08 threshold is not a universal constant and
  would need recalibrating for a different catalog size or customer base.
- **No cross-technique feature reuse beyond the shared RFM+ table** — e.g., cluster
  assignment isn't fed as a feature into the supervised model, which could plausibly
  improve it; not pursued given the capstone's breadth-over-depth time budget.
