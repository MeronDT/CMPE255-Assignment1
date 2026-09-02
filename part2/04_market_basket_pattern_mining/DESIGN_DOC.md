# Design Doc — Market Basket Pattern Mining

## 1. Overview

Association rule mining on the same UCI/Kaggle Online Retail dataset used in Project 3,
this time at the basket (invoice) level. Full CRISP-DM pipeline + AutoResearch +
FastAPI/React dashboard. See
[docs/01_business_understanding.md](./docs/01_business_understanding.md).

## 2. Pipeline

| Script | CRISP-DM Phase | Output |
|---|---|---|
| `01_eda.py` | Data Understanding | `docs/eda/eda_findings.json` |
| `02_prepare_data.py` | Data Preparation | `data/processed/basket_matrix.pkl` (one-hot basket×product) |
| `03_train_models.py` | Modeling (baseline) | Apriori vs. FP-Growth comparison, baseline rules |
| `04_autoresearch.py` | AutoResearch search | grid search over (support, confidence) |
| `05_evaluate.py` | Evaluation | rule summary, product hub analysis, rule graph |

## 3. Feature Engineering

Transaction log → `groupby("InvoiceNo")` → set of distinct `StockCode`s per basket →
`mlxtend.preprocessing.TransactionEncoder` one-hot matrix. Restricted to the **top 200**
most frequent products (of ~3,900 distinct codes) — below that, per-product basket
co-occurrence counts drop too low for any threshold to yield statistically meaningful
support. Single-item baskets are dropped (no co-occurrence possible). Cancelled orders
excluded entirely (a cancellation isn't a purchase basket).

## 4. Modeling & AutoResearch

**Baseline** (`03_train_models.py`): Apriori vs. FP-Growth compared at `min_support=0.02`
— both found identical itemsets (456), confirming correctness; FP-Growth was NOT faster
at this scale (0.597s vs. Apriori's 0.548s), an honest divergence from the algorithms'
usual literature framing, reported rather than hidden.

**AutoResearch** (`04_autoresearch.py`): grid search over `min_support` × `min_confidence`
(7×5 = 35 configurations). **Key correction made during this run**: the first scoring
function (`n_high_lift_rules` alone) picked the loosest thresholds and returned 7,799
rules — unusable in practice, the association-rule-mining equivalent of Project 3's
unconstrained k=2 finding. Fixed with the same pattern used there: an explicit business
constraint (5–50 rules, a merchandiser-reviewable set) applied *before* ranking, then
selecting the highest-average-lift config among those satisfying it. Winner:
`min_support=0.03, min_confidence=0.6` → 27 rules, avg lift 9.31x, max lift 14.13x.

## 5. Key Decision: Business-Constrained Selection (mirrors Project 3 §5)

Same structural finding as Project 3's k=2-vs-k=3 discussion, in a different mining task:
an unconstrained objective (maximize raw high-lift rule count) finds a *statistically*
valid but *practically* useless answer. Reported transparently on the dashboard's
AutoResearch tab (full 35-row grid shown, winning row highlighted) rather than only
showing the final winner.

## 6. Deployment

FastAPI backend (`backend/app/main.py`, port 8004). React + TypeScript + Recharts
frontend (port 5177) — six tabs: Overview, Rules (searchable table), Products (hub
bar chart), Modeling, AutoResearch (grid scatter + full table), Data & EDA.

Verified via Playwright: 0 console errors, 0 failed requests across all six tabs.
