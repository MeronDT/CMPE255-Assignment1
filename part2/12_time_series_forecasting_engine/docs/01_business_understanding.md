# CRISP-DM Phase 1: Business Understanding

## Project

**Time Series Forecasting Engine** — daily revenue forecasting for the UCI/Kaggle
Online Retail store, reused deliberately from Projects 03/04/10 as the underlying
business (a UK online gift retailer), this time viewed through its **temporal**
dimension rather than customer or product structure. Same dataset, a genuinely
different mining task (forecasting vs. clustering/association/anomaly detection).

## Business Objective

A retailer needs to forecast daily revenue for inventory planning, staffing, and cash
flow management. The goal: build a forecasting engine that predicts near-future daily
revenue with a quantified, honestly-reported error margin, comparing classical
statistical methods against Facebook's Prophet (the modern practitioner's default) and
a naive baseline (the mandatory sanity-check floor any real model must beat).

## Data Mining Objective

Aggregate transaction-level data into a daily revenue time series (Dec 2010–Dec 2011),
then compare:
- **Seasonal Naive baseline** (predict "same day last week") — the floor any real
  model must beat to justify its complexity.
- **Holt-Winters Exponential Smoothing** (statsmodels) — classical, interpretable,
  captures trend + weekly seasonality explicitly.
- **SARIMA** (statsmodels) — the classical statistical workhorse for seasonal time
  series, hill-climbed over (p,d,q)(P,D,Q,s) orders.
- **Prophet** (Meta/Facebook, Taylor & Letham 2018) — additive trend + seasonality
  decomposition, designed for exactly this kind of business time series with holidays
  and irregular gaps.

Success is measured by **MAPE and RMSE on a held-out final-N-days test window**
(a proper time-series split — never randomly shuffled, since that would leak future
information into training, a classic time-series-specific pitfall this project
deliberately avoids).

## Scope & Constraints

- Only ~373 days of data exist (Dec 2010–Dec 2011) — a real constraint on how much
  seasonal structure (e.g., yearly seasonality) can be reliably estimated, stated
  explicitly rather than overclaimed.
- Following the pattern from every prior project: full CRISP-DM lifecycle as
  executable scripts, an AutoResearch hill-climbing phase (SARIMA order search,
  matched against classical time-series-analysis practice), and a
  data-scientist-facing admin dashboard.
