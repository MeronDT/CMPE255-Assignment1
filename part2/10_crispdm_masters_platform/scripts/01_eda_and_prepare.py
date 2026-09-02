"""CRISP-DM Phase 2+3: Data Understanding + Preparation (shared across all 5 techniques).

Cleans the raw transaction log once and derives every feature table this
capstone's five techniques need:
- customer_features.csv: full-history RFM+ (for clustering, anomaly detection, LSH)
- customer_early_features.csv + label: first-90-day features + "becomes high-value"
  label (for supervised learning) -- see leakage discussion below
- basket_matrix.pkl: one-hot basket x product (for association rule mining)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs" / "eda"
PROCESSED.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)

EARLY_WINDOW_DAYS = 90
TOP_N_PRODUCTS = 200


def main():
    df = pd.read_excel(RAW / "Online Retail.xlsx")
    df["StockCode"] = df["StockCode"].astype(str)

    eda = {
        "n_raw_transactions": len(df),
        "n_missing_customer_id": int(df["CustomerID"].isnull().sum()),
        "date_range": [str(df["InvoiceDate"].min()), str(df["InvoiceDate"].max())],
    }

    # --- Cleaning (same justified decisions as Projects 03/04) ---
    df_customers = df.dropna(subset=["CustomerID"]).copy()
    df_customers = df_customers[df_customers["UnitPrice"] > 0].copy()
    df_customers["CustomerID"] = df_customers["CustomerID"].astype(int)
    df_customers["LineTotal"] = df_customers["Quantity"] * df_customers["UnitPrice"]
    df_customers["IsCancellation"] = df_customers["InvoiceNo"].astype(str).str.startswith("C")

    snapshot_date = df_customers["InvoiceDate"].max() + pd.Timedelta(days=1)

    # --- Full-history RFM+ features (clustering, anomaly detection, LSH) ---
    g = df_customers.groupby("CustomerID")
    rfm = pd.DataFrame({
        "Recency": g["InvoiceDate"].apply(lambda s: (snapshot_date - s.max()).days),
        "Frequency": g["InvoiceNo"].nunique(),
        "Monetary": g["LineTotal"].sum(),
        "AvgBasketValue": g["LineTotal"].sum() / g["InvoiceNo"].nunique(),
        "DistinctProducts": g["StockCode"].nunique(),
        "TenureDays": g["InvoiceDate"].apply(lambda s: (s.max() - s.min()).days),
        "CancellationRate": g["IsCancellation"].mean(),
        "FirstPurchase": g["InvoiceDate"].min(),
    }).reset_index()
    rfm["Monetary"] = rfm["Monetary"].clip(lower=0.01)
    rfm["AvgBasketValue"] = rfm["AvgBasketValue"].clip(lower=0.01)
    rfm.to_csv(PROCESSED / "customer_features.csv", index=False)

    # --- Early-window features + supervised label ---
    # Only customers with >= EARLY_WINDOW_DAYS of observed tenure are included --
    # a customer who joined 10 days before the snapshot can't be fairly judged on
    # a 90-day window, so they're excluded rather than given a misleading label.
    eligible = rfm[rfm["TenureDays"] >= EARLY_WINDOW_DAYS].copy()
    eligible_ids = set(eligible["CustomerID"])
    first_purchase = rfm.set_index("CustomerID")["FirstPurchase"].to_dict()

    df_eligible = df_customers[df_customers["CustomerID"].isin(eligible_ids)].copy()
    df_eligible["DaysSinceFirst"] = df_eligible.apply(
        lambda r: (r["InvoiceDate"] - first_purchase[r["CustomerID"]]).days, axis=1
    )
    df_early = df_eligible[df_eligible["DaysSinceFirst"] <= EARLY_WINDOW_DAYS]

    g_early = df_early.groupby("CustomerID")
    early_feat = pd.DataFrame({
        "EarlyFrequency": g_early["InvoiceNo"].nunique(),
        "EarlyMonetary": g_early["LineTotal"].sum(),
        "EarlyAvgBasketValue": g_early["LineTotal"].sum() / g_early["InvoiceNo"].nunique(),
        "EarlyDistinctProducts": g_early["StockCode"].nunique(),
        "EarlyCancellationRate": g_early["IsCancellation"].mean(),
    }).reindex(eligible["CustomerID"]).fillna(0).reset_index()

    # Label: top-quartile TOTAL lifetime Monetary among eligible customers.
    threshold = eligible["Monetary"].quantile(0.75)
    eligible["IsHighValue"] = (eligible["Monetary"] >= threshold).astype(int)

    supervised = early_feat.merge(eligible[["CustomerID", "Monetary", "IsHighValue"]], on="CustomerID")
    supervised.to_csv(PROCESSED / "supervised_early_features.csv", index=False)

    # Honest leakage quantification: what fraction of a customer's total lifetime
    # spend, on average, occurred within the first EARLY_WINDOW_DAYS itself? This
    # is EXPECTED correlation (early behavior predicts total behavior), not
    # leakage in the strict sense -- but worth stating exactly how much overlap
    # exists rather than leaving it implicit.
    merged_for_check = early_feat.merge(eligible[["CustomerID", "Monetary"]], on="CustomerID")
    # Clip BOTH bounds: a net-negative early window (heavy early returns) contributes
    # 0%, not a large negative number, to this "share of lifetime value" statistic --
    # the earlier unclipped-lower version let one such customer (near-zero lifetime
    # Monetary, from the 0.01 floor) drag the average to -3287%, an obvious bug caught
    # by sanity-checking the output rather than trusting it blindly.
    ratio = (merged_for_check["EarlyMonetary"] / merged_for_check["Monetary"]).clip(lower=0, upper=1)
    pct_of_total_in_early_window = float(ratio.mean())

    # --- Basket matrix for association rule mining (same approach as Project 04) ---
    baskets_raw = df_customers[df_customers["Quantity"] > 0]
    product_names = baskets_raw.groupby("StockCode")["Description"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else "?")
    top_products = baskets_raw["StockCode"].value_counts().head(TOP_N_PRODUCTS).index
    baskets = baskets_raw[baskets_raw["StockCode"].isin(top_products)]
    transactions = baskets.groupby("InvoiceNo")["StockCode"].apply(lambda s: list(set(s))).tolist()
    transactions = [t for t in transactions if len(t) >= 2]
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    basket_matrix = pd.DataFrame(te_ary, columns=te.columns_)
    basket_matrix.to_pickle(PROCESSED / "basket_matrix.pkl")
    product_names.to_json(PROCESSED / "product_names.json")

    eda.update({
        "n_customers_full_history": len(rfm),
        "n_customers_eligible_for_supervised": len(eligible),
        "supervised_label_threshold_monetary": round(float(threshold), 2),
        "supervised_positive_rate": round(float(eligible["IsHighValue"].mean()), 4),
        "pct_of_lifetime_spend_in_first_90_days_avg": round(pct_of_total_in_early_window * 100, 1),
        "n_multi_item_baskets": len(transactions),
        "n_products_in_basket_matrix": len(te.columns_),
    })
    with open(DOCS / "eda_and_prep_summary.json", "w") as f:
        json.dump(eda, f, indent=2, default=str)

    print(json.dumps(eda, indent=2, default=str))


if __name__ == "__main__":
    main()
