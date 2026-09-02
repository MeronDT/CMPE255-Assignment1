"""CRISP-DM Phase 2: Data Understanding / EDA -- basket-level view.

Same raw file as Project 3, but examined at the invoice/basket level rather
than the customer level: basket size distribution, product frequency, and
country-mix findings that justify the cleaning decisions in 02_prepare_data.py.
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
    is_cancellation = df["InvoiceNo"].astype(str).str.startswith("C")
    baskets = df[~is_cancellation & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    basket_sizes = baskets.groupby("InvoiceNo")["StockCode"].nunique()
    product_freq = baskets["StockCode"].value_counts()
    product_names = baskets.groupby("StockCode")["Description"].first()

    findings = {
        "n_raw_rows": len(df),
        "n_cancellation_rows": int(is_cancellation.sum()),
        "n_basket_line_items": len(baskets),
        "n_baskets": int(baskets["InvoiceNo"].nunique()),
        "n_distinct_products": int(baskets["StockCode"].nunique()),
        "basket_size_stats": basket_sizes.describe().to_dict(),
        "pct_baskets_single_item": round(float((basket_sizes == 1).mean()) * 100, 2),
        "top_20_products": [
            {"stock_code": sc, "name": str(product_names.get(sc, "?")), "n_baskets": int(n)}
            for sc, n in product_freq.head(20).items()
        ],
    }

    with open(DOCS / "eda_findings.json", "w") as f:
        json.dump(findings, f, indent=2, default=str)

    print(json.dumps(findings, indent=2, default=str))


if __name__ == "__main__":
    main()
