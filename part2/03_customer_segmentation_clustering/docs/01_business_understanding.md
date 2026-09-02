# CRISP-DM Phase 1: Business Understanding

## Project

**Customer Intelligence & Segmentation Clustering** — unsupervised customer segmentation on the
[UCI/Kaggle Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail)
(541,909 transactions, Dec 2010–Dec 2011, a UK-based online gift retailer). This is one of the
most widely used public datasets for RFM-based customer segmentation research and tutorials —
matching the "popular Kaggle dataset" requirement (it is mirrored on Kaggle as
`vijayuv/onlineretail` and several similar listings; sourced here directly from UCI, its
canonical host, since no Kaggle API credentials are configured in this environment — same
substitution pattern as Project 01's NYC TLC bucket).

## Business Objective

A retailer with tens of thousands of one-off transactions has no native notion of "customer
segments." Marketing, retention, and inventory decisions are more effective when customers are
grouped by *behavior* (how recently, how often, and how much they buy) rather than treated
uniformly. The goal: cluster the ~4,300 identified customers into behaviorally distinct segments
a marketing team could act on (e.g., "at-risk high-value," "loyal frequent," "one-time
low-value"), and expose this as an interactive admin dashboard.

## Data Mining Objective

Unsupervised clustering (no labels exist) on customer-level features engineered from raw
transaction logs — primarily **RFM** (Recency, Frequency, Monetary), the standard feature basis
in the customer-segmentation literature since Hughes (1994), extended with a few behavioral
features (average basket size, product variety, customer tenure). Success is measured by
internal cluster-validity metrics (silhouette score, Davies-Bouldin index) since there is no
ground-truth label to score against — and by whether the resulting segments are interpretable
and separated in a way a marketer could act on (checked qualitatively via cluster profiling).

## Scope & Constraints

- Transactions with no `CustomerID` (guest checkouts, ~25% of rows) cannot be attributed to a
  customer and are excluded from the customer-level model — they remain visible in
  transaction-level EDA.
- Negative quantities are returns/cancellations (`InvoiceNo` starting with "C"); these are
  netted into each customer's totals rather than dropped, since a customer who buys and returns
  heavily *is* meaningfully different from one who doesn't.
- Single-country dominance (>90% UK) is expected and reported honestly, not corrected for — this
  is a UK-based gift retailer, and diluting that signal with a "balance the countries" heuristic
  would misrepresent the actual customer base.
- Following the pattern established in Projects 01–02: full CRISP-DM lifecycle as executable
  scripts, an AutoResearch hill-climbing phase (algorithm + hyperparameter + feature-transform
  search, matched against published RFM segmentation literature), and a data-scientist-facing
  admin dashboard — timeboxed per project, not exhaustively tuned.
