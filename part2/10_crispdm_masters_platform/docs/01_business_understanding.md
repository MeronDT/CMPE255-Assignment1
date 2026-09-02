# CRISP-DM Phase 1: Business Understanding

## Project

**CRISP-DM Master's Data Science Platform** — a single, textbook-quality end-to-end
walkthrough of every major data mining paradigm on **one dataset**: the UCI/Kaggle
Online Retail transaction log (reused deliberately from Projects 03–04, this time as
the unifying substrate for a full capstone rather than one technique in isolation).

## Why one dataset for everything

A real business rarely has the luxury of a bespoke dataset per technique. This project
demonstrates the CRISP-DM methodology's actual value: **the same cleaned transaction
data, feature-engineered once, supports every major mining paradigm** a working data
scientist is expected to know:

1. **Unsupervised learning (clustering)** — customer segmentation (RFM + K-Means,
   methodology consistent with Project 03).
2. **Anomaly/outlier detection** — flagging transactions with statistically unusual
   quantity/price/basket patterns (Isolation Forest, methodology consistent with
   Project 06).
3. **Supervised learning** — a genuinely predictive task derived from the same data:
   given a customer's first-90-days behavior, predict whether they become a
   "high-value" customer (top-quartile lifetime spend) — a real business question
   (early customer-value prediction for targeted retention spend).
4. **Associative rule mining** — market basket analysis (Apriori/FP-Growth,
   methodology consistent with Project 04).
5. **Sub-linear search (LSH)** — approximate nearest-neighbor customer search via
   MinHash Locality-Sensitive Hashing: "find customers with similar purchase
   baskets to this one" in sub-linear time, instead of an O(n) brute-force scan
   against all ~4,300 customers.

## Business Objective

Give a data science team (and this project's grader) a single coherent example of how
CRISP-DM's phases compose across an entire toolkit, not just one algorithm — with each
phase's output feeding into the phases after it (e.g., the same customer feature table
built in Data Preparation is reused, not rebuilt, for clustering, anomaly detection,
*and* supervised learning).

## Data Mining Objectives (one per technique)

| Technique | Objective | Success Metric |
|---|---|---|
| Clustering | Segment customers into actionable groups | Silhouette score (business-constrained, per Project 03's finding) |
| Anomaly detection | Flag statistically unusual transactions | AUPRC-style separation from a synthetic-injected outlier set (no natural fraud labels exist in this data, unlike Project 06 — addressed explicitly in Modeling) |
| Supervised learning | Predict future high-value customers from early behavior | ROC-AUC, precision/recall on held-out customers |
| Association rules | Surface actionable product co-purchase patterns | Business-constrained rule count × average lift (per Project 04's finding) |
| LSH | Sub-linear approximate similar-customer search | Query speed vs. brute-force, recall@k against true nearest neighbors |

## Scope & Constraints

- This dataset has **no natural anomaly labels** (unlike Project 06's Credit Card
  Fraud data) — addressed honestly in the Modeling phase by injecting a small set of
  synthetic extreme-outlier transactions to validate the detector can find genuine
  outliers, rather than pretending an evaluation exists that doesn't.
- The supervised-learning target (future high-value customer) is **engineered from
  the same data**, not an external label — its construction and the
  potential-leakage risk (does using full-history RFM features to predict a
  full-history-derived label leak information?) is addressed explicitly in Data
  Preparation, not glossed over.
- Following the pattern from every prior project: a data-scientist-facing admin
  dashboard, with quizzes woven through each technique's section and a closing
  synthesis/conclusion phase — timeboxed appropriately for a capstone covering seven
  distinct requirements, not exhaustively tuned on any single one.
