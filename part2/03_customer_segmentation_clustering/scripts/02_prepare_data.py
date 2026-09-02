"""CRISP-DM Phase 3: Data Preparation.

Transaction log -> customer-level RFM(+) feature table.

Cleaning decisions (justified by 01_eda.py findings):
- Drop rows with no CustomerID (25% of rows, ~135K) — cannot attribute to a customer,
  so unusable for customer-level clustering. They remain visible in transaction EDA.
- Keep cancellations (negative Quantity) rather than dropping them: they net into each
  customer's totals so a customer who returns heavily nets out lower Monetary/Frequency,
  which IS the correct signal for a segmentation model, not noise to remove.
- Drop rows with UnitPrice <= 0 (~2,517 rows): these are non-sale artifacts (bank charges,
  manual adjustments, free samples with no price) that would distort Monetary if kept as
  revenue lines.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs" / "eda"
PROCESSED.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_excel(RAW / "Online Retail.xlsx")

    before = len(df)
    df = df.dropna(subset=["CustomerID"]).copy()
    df = df[df["UnitPrice"] > 0].copy()
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["LineTotal"] = df["Quantity"] * df["UnitPrice"]
    df["IsCancellation"] = df["InvoiceNo"].astype(str).str.startswith("C")

    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    g = df.groupby("CustomerID")
    rfm = pd.DataFrame({
        "Recency": g["InvoiceDate"].apply(lambda s: (snapshot_date - s.max()).days),
        "Frequency": g["InvoiceNo"].nunique(),
        "Monetary": g["LineTotal"].sum(),
        "AvgBasketValue": g["LineTotal"].sum() / g["InvoiceNo"].nunique(),
        "DistinctProducts": g["StockCode"].nunique(),
        "TotalItems": g["Quantity"].sum(),
        "TenureDays": g["InvoiceDate"].apply(lambda s: (s.max() - s.min()).days),
        "CancellationRate": g["IsCancellation"].mean(),
        "PrimaryCountry": g["Country"].agg(lambda s: s.mode().iat[0]),
    }).reset_index()

    # Guard against pathological rows: a customer whose only activity nets to <=0 spend
    # (heavy net-returner) has an undefined "value" for a value-based segmentation; keep
    # them but floor Monetary at a tiny positive epsilon so log-transform (used in
    # AutoResearch's feature-transform search) doesn't produce -inf.
    rfm["Monetary"] = rfm["Monetary"].clip(lower=0.01)
    rfm["AvgBasketValue"] = rfm["AvgBasketValue"].clip(lower=0.01)

    rfm.to_csv(PROCESSED / "customer_rfm.csv", index=False)

    summary = {
        "n_transactions_before_cleaning": before,
        "n_transactions_after_cleaning": len(df),
        "n_customers": len(rfm),
        "rfm_stats": rfm.describe().to_dict(),
        "snapshot_date": str(snapshot_date),
    }
    with open(DOCS / "prep_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Customers: {len(rfm):,} (from {before:,} -> {len(df):,} cleaned transactions)")
    print(rfm.describe())


if __name__ == "__main__":
    main()
