# Research Report — Customer Intelligence & Segmentation Clustering

## 1. Introduction

RFM (Recency, Frequency, Monetary) segmentation is the standard feature basis for
customer-value clustering in retail analytics, dating to Hughes (1994) and validated
repeatedly in the marketing-analytics literature since (e.g. Christy, A.J. et al. 2021,
"RFM ranking – An effective approach to customer segmentation using machine learning").
This project applies it to the UCI/Kaggle Online Retail dataset (541,909 transactions,
4,371 identifiable customers after cleaning) via K-Means, chosen through a 4-algorithm
tournament and an AutoResearch hill-climbing search over feature transforms and k.

## 2. Data Preparation

25% of raw transaction rows have no `CustomerID` (guest checkouts) and are excluded from
customer-level clustering — they remain visible in transaction-level EDA. Cancellations
(9,288 invoices, 1.71% of rows, `InvoiceNo` starting with "C") are **netted into** each
customer's totals rather than dropped: a customer with a high cancellation rate is
behaviorally different from one without, which is exactly the signal a segmentation model
should capture. 2,517 rows with `UnitPrice ≤ 0` (bank charges, manual adjustments, free
samples) are dropped as non-sale artifacts that would distort Monetary.

## 3. Modeling: Algorithm Tournament

K-Means, Agglomerative (Ward linkage), Gaussian Mixture, and DBSCAN were compared on
log-transformed (Monetary, AvgBasketValue, Frequency, DistinctProducts — all right-skewed),
standardized RFM+ features, with k chosen by silhouette sweep (k=2..10):

| Algorithm | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---:|---:|---:|
| **K-Means** | **0.355** | 1.075 | 2,922 |
| Agglomerative (Ward) | 0.333 | 1.148 | 2,589 |
| Gaussian Mixture | 0.302 | 1.242 | 2,082 |

(DBSCAN was swept over `eps` instead of k, since it has no k parameter — see
`docs/eda/modeling_baseline.json` for the full eps sweep. Best DBSCAN silhouette 0.515 at
eps=1.5, but at that setting it collapses to 2 clusters plus 22 noise points — not more
informative than K-Means at k=2, and DBSCAN's non-convex clusters are harder for a
marketing team to reason about than K-Means' centroid-based segments.)

K-Means wins the tournament and produces the most directly interpretable ("distance to
segment centroid") output, so it was carried forward into the AutoResearch phase.

## 4. AutoResearch: the k=2 vs. k=3 Finding

**This is the project's central result, not a footnote.** Pure silhouette maximization
over k=2..10 selects **k=2** (silhouette 0.355). Qualitatively, this splits customers into
"big spenders" and "everyone else" — statistically the most cleanly separated partition,
but useless to a marketing team, which needs multiple actionable tiers (e.g., "who do we
retain," "who do we upsell," "who do we win back"), not a single big/small split. This
directly fails the success criterion set in `docs/01_business_understanding.md`.

Rather than discard silhouette maximization and hand-pick a "nicer-looking" k, the
AutoResearch phase (`scripts/04_autoresearch.py`) applies an explicit, literature-grounded
constraint — k ∈ [3, 8], the range consistently reported as actionable in RFM segmentation
studies — *before* searching, then re-runs the same 4-phase process (feature-transform
search → greedy hill-climb over k → finalize):

1. **Phase 2 (feature/transform search)**: 2 feature sets × 3 scalers at a representative
   k=5. Winner: `rfm_core_plus_cancellation` (including `CancellationRate` as a feature) +
   MinMax scaling, silhouette 0.356 — beating every other combination, including the
   feature set that excludes cancellation rate (0.248–0.302 across scalers).
2. **Phase 3 (hill-climb over k)**: literal greedy search — every k in [3,8] evaluated,
   best silhouette wins. **k=3** wins at silhouette **0.409**.
3. **Phase 4 (finalize)**: retrain k=3 on the full dataset, confirm the result, deploy.

The genuinely interesting finding: **k=3 (business-constrained) scores *higher*
silhouette (0.409) than the unconstrained k=2 optimum (0.355)**. This isn't a
worse-but-more-useful trade-off — it's an outright improvement that the unconstrained
search never found, because plain silhouette-vs-k sweeps get stuck evaluating the wrong
feature/scaling configuration at each k. Searching the transform space *jointly* with k,
inside the business-relevant range, found a genuinely better clustering than either
the naive baseline or the naive constraint-then-pick-best-k approach would have alone.

## 5. Evaluation: Cluster Profiles

| Segment | Customers | % Revenue | Avg Recency | Avg Frequency | Avg Monetary | Cancellation Rate |
|---|---:|---:|---:|---:|---:|---:|
| Loyal High-Value | 1,785 (40.8%) | 81.7% | 34.2d | 9.5 | $3,802.93 | 2.9% |
| New / Occasional | 1,616 (37.0%) | 13.6% | 56.1d | 2.2 | $698.42 | 1.8% |
| At-Risk / Dormant | 970 (22.2%) | 4.8% | 258.3d | 1.7 | $408.69 | 6.1% |

Persona labels are rule-based (centroid position relative to the overall median on
Recency/Frequency/Monetary — see `scripts/05_evaluate.py`), not manually assigned after
the fact. The 82%-revenue-from-41%-of-customers concentration is a textbook Pareto
pattern in retail RFM analysis, not an artifact of clustering with too few segments —
it holds up as a genuine finding a retention/VIP program could act on directly.

## 6. Limitations & Future Work

- **Static snapshot, not time-aware.** RFM features are computed once against a single
  snapshot date; a production system would recompute this on a rolling basis and could
  track segment *migration* (customers moving from Loyal to At-Risk) as its own signal.
- **No product-level segmentation.** This clusters customers by behavior, not products by
  co-purchase pattern — that's Project 4 (Market Basket Pattern Mining)'s scope, not
  duplicated here.
- **Country feature excluded from clustering deliberately** (used for display only) since
  the dataset is 91% UK-dominated; including it would just re-derive UK/non-UK rather than
  add behavioral signal, as noted in DESIGN_DOC.md §3.
- **AutoResearch's k-range constraint (k∈[3,8]) is itself a modeling choice**, grounded in
  the cited literature but not exhaustively validated against alternative constraints
  (e.g., k∈[4,6]) — a reasonable next iteration given more time budget.
