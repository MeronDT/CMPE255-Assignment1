# CRISP-DM Phase 1: Business Understanding

## Project

**Market Basket Pattern Mining** — association rule mining on the same
[UCI/Kaggle Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail)
used in Project 3, this time mined at the **basket** (invoice) level rather than the
customer level. This is deliberate reuse, not duplication: Project 3 asks "which
customers behave alike," Project 4 asks "which *products* tend to be bought together" —
a genuinely different mining task (unsupervised association-rule discovery vs.
clustering) on the same canonical retail dataset, which is itself one of the most common
datasets used to teach/demo Apriori and FP-Growth in the market-basket-analysis
literature.

## Business Objective

A retailer with tens of thousands of transactions has latent co-purchase patterns no
human could spot by eye ("customers who buy X also buy Y 73% of the time"). Surfacing
these patterns supports cross-sell recommendations, shelf/catalog placement, and bundle
promotions. The goal: mine statistically significant, actionable association rules from
the transaction log and expose them as an interactive admin dashboard a merchandising
team could act on directly.

## Data Mining Objective

Association rule mining via **Apriori** (Agrawal & Srikant, 1994) and **FP-Growth** (Han,
Pei & Yin, 2000) — the two foundational algorithms in this literature — compared on
runtime and rule yield at matched support thresholds. Rules are scored by **support**,
**confidence**, and **lift** (the standard three-metric basis since Agrawal & Srikant);
a rule is only interesting if lift > 1 (the itemset co-occurs more than chance would
predict), which is the primary filter applied before ranking.

## Scope & Constraints

- Basket-level analysis only considers `InvoiceNo` groupings with `CustomerID` present
  (same customer-attribution constraint as Project 3) is **not** required here — a
  basket is defined by the invoice itself, so guest-checkout transactions are legitimate
  baskets and are kept (unlike Project 3, which needed a stable customer identity across
  time and therefore excluded them).
- Cancelled orders (`InvoiceNo` starting with "C") are excluded from basket mining — a
  cancellation is not a "purchase basket," it's the reversal of one, and mixing the two
  would corrupt co-occurrence counts.
- Following the pattern established in Projects 01–03: full CRISP-DM lifecycle as
  executable scripts, an AutoResearch hill-climbing phase (support/confidence threshold
  + algorithm search, matched against the founding Apriori/FP-Growth papers), and a
  data-scientist-facing admin dashboard — timeboxed, not exhaustively tuned.
