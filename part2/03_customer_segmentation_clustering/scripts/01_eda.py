"""CRISP-DM Phase 2: Data Understanding / EDA.

Loads the raw Online Retail transaction log and reports data-quality findings
that directly justify the cleaning decisions made in 02_prepare_data.py.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DOCS = ROOT / "docs" / "eda"
DOCS.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_excel(RAW / "Online Retail.xlsx")

    n_rows = len(df)
    n_missing_customer = int(df["CustomerID"].isnull().sum())
    is_cancellation = df["InvoiceNo"].astype(str).str.startswith("C")

    findings = {
        "n_transactions": n_rows,
        "n_columns": df.shape[1],
        "columns": list(df.columns),
        "date_range": [str(df["InvoiceDate"].min()), str(df["InvoiceDate"].max())],
        "n_unique_customers": int(df["CustomerID"].nunique()),
        "n_unique_products": int(df["StockCode"].nunique()),
        "n_countries": int(df["Country"].nunique()),
        "top_countries": df["Country"].value_counts().head(10).to_dict(),
        "pct_uk": round(float((df["Country"] == "United Kingdom").mean()) * 100, 2),
        "n_missing_customer_id": n_missing_customer,
        "pct_missing_customer_id": round(n_missing_customer / n_rows * 100, 2),
        "n_missing_description": int(df["Description"].isnull().sum()),
        "n_cancellation_invoices": int(is_cancellation.sum()),
        "pct_cancellation": round(float(is_cancellation.mean()) * 100, 2),
        "n_negative_quantity": int((df["Quantity"] < 0).sum()),
        "n_zero_or_negative_price": int((df["UnitPrice"] <= 0).sum()),
        "quantity_stats": df["Quantity"].describe().to_dict(),
        "unit_price_stats": df["UnitPrice"].describe().to_dict(),
    }

    # Sanity check: are all negative-quantity rows cancellations?
    neg_qty = df[df["Quantity"] < 0]
    findings["neg_qty_that_are_cancellations_pct"] = round(
        float(neg_qty["InvoiceNo"].astype(str).str.startswith("C").mean()) * 100, 2
    )

    with open(DOCS / "eda_findings.json", "w") as f:
        json.dump(findings, f, indent=2, default=str)

    print(json.dumps(findings, indent=2, default=str))


if __name__ == "__main__":
    main()
