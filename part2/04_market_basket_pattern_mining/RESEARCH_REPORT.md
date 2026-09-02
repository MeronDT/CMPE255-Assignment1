# Research Report — Market Basket Pattern Mining

## 1. Introduction

Association rule mining is the foundational unsupervised technique for discovering
co-purchase patterns in transaction data, formalized by Agrawal & Srikant's Apriori
algorithm (1994) and made more efficient by Han, Pei & Yin's FP-Growth (2000). This
project applies both to the UCI/Kaggle Online Retail dataset — reusing Project 3's
dataset deliberately, since it is analyzed at the basket level here rather than the
customer level, a genuinely different mining task.

## 2. Data Preparation

Transaction rows are grouped into baskets by `InvoiceNo`. Cancelled orders (`InvoiceNo`
starting with "C") are excluded entirely — a cancellation is the reversal of a purchase,
not a basket itself. Analysis is restricted to the top 200 most frequent products (of
~3,900 distinct `StockCode`s): below this cutoff, per-product basket-appearance counts
fall too low to support statistically meaningful thresholds at any reasonable operating
point. Single-item baskets are dropped (15,122 multi-item baskets remain, avg. basket
size 10.06 items) since no co-occurrence rule can be derived from them.

## 3. Modeling: Apriori vs. FP-Growth

Compared at `min_support=0.02` on the 200-product basket matrix:

| Algorithm | Runtime | Frequent Itemsets |
|---|---:|---:|
| Apriori | 0.548s | 456 |
| FP-Growth | 0.597s | 456 |

Both algorithms found **identical itemsets** (a correctness cross-check, not just a
runtime comparison — a mismatch would indicate a bug in one implementation or the other).
FP-Growth's textbook advantage is faster runtime via single-pass FP-tree construction
vs. Apriori's repeated candidate-generation database scans — but **at this dataset's
scale, that advantage did not materialize**; Apriori was marginally faster. This is
reported honestly rather than silently deploying FP-Growth on the papers' authority
alone. The likely explanation: FP-tree construction has fixed overhead that only pays
off once the itemset search space is significantly larger/denser than 200 products ×
15K baskets — consistent with the original FP-Growth paper's own benchmarks, which
demonstrate its advantage on datasets with hundreds of thousands to millions of
transactions, not tens of thousands.

## 4. AutoResearch: the Business-Constrained Selection Problem

**This project's central finding mirrors Project 3's k=2-vs-k=3 discussion in a
different mining task.** The first AutoResearch pass scored each (support, confidence)
grid point purely by count of high-lift (>2x) rules. This objective is monotonically
biased toward looser thresholds: the winning config was `min_support=0.01,
min_confidence=0.2`, yielding **7,799 rules** — every one nominally "high-lift," but
utterly unusable for a merchandising team to review, exactly the same class of failure
as Project 3's unconstrained silhouette-maximization picking k=2.

**Fix, applying the same pattern used in Project 3 §5**: introduce an explicit business
constraint *before* ranking — a merchandising team can meaningfully review roughly
5–50 rules in a sitting, not thousands. Among the grid configurations whose rule count
falls in that band, select the one maximizing **average lift** (not count), rewarding
genuinely surprising/valuable associations over sheer volume:

| Support | Confidence | Rules | Avg Lift |
|---:|---:|---:|---:|
| 0.03 | 0.5 | 47 | 8.548 |
| **0.03** | **0.6** | **27** | **9.307** ← winner |
| 0.05 | 0.4–0.6 | 3 | 8.918 |

`min_support=0.03, min_confidence=0.6` wins: 27 rules, average lift 9.31x, maximum
lift 14.13x — comfortably inside the reviewable band while surfacing the strongest
associations in the data, not diluted by thousands of marginal ones.

## 5. Evaluation: Rule Quality

Every one of the 27 final rules is qualitatively sensible on inspection — a genuine
sanity check, not just a numeric threshold pass:

- **Product-collection variants**: `PINK/GREEN/ROSES REGENCY TEACUP AND SAUCER` form a
  tightly interlinked cluster of rules (lift 10.9x–14.1x) — obviously the same product
  line in different colorways, exactly the kind of association a "frequently bought
  together" widget should surface.
- **Themed sets**: `GARDENERS KNEELING PAD CUP OF TEA` ↔ `KEEP CALM` (lift 12.1x),
  `ALARM CLOCK BAKELIKE RED` ↔ `GREEN` (lift 9.5x) — matching decor/gift-set patterns.
- **Hub products**: `GREEN REGENCY TEACUP AND SAUCER` appears in 8 of the 27 rules —
  a strong candidate for cross-sell placement, since it sits at the center of multiple
  independent high-lift associations rather than a single isolated pairing.

## 6. Limitations & Future Work

- **Top-200-product restriction is a scale trade-off**, not a hard limitation — the
  full ~3,900-product space could be mined given more compute budget, though EDA
  suggests most long-tail products wouldn't clear any reasonable support threshold
  regardless.
- **The 5–50 rule business constraint is itself a modeling choice**, not empirically
  derived from actual merchandiser feedback — a reasonable starting assumption stated
  explicitly in `01_business_understanding.md`, not validated against real usage.
- **No temporal/seasonal analysis** — rules are mined over the full Dec 2010–Dec 2011
  window; a production system might mine separate rule sets per season (this dataset
  has an obvious Christmas-gift skew visible in EDA) to catch time-varying associations.
- **Static rules, no online updating** — a production recommender would need to
  re-mine periodically as the catalog and purchase patterns shift.
