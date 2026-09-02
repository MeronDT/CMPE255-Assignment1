"""CRISP-DM Phase 3: Data Preparation.

Transaction log -> one-hot encoded basket x product matrix, the standard
input format for Apriori/FP-Growth. Restricted to the top N most frequent
products (justified below) to keep the encoded matrix a tractable size --
mlxtend's apriori is combinatorial in the number of columns, and the raw
data has ~3,900 distinct StockCodes, the vast majority appearing in only a
handful of baskets (long-tail products can't produce statistically
meaningful support at any reasonable threshold anyway).
"""
import json
from pathlib import Path

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs" / "eda"
PROCESSED.mkdir(parents=True, exist_ok=True)

TOP_N_PRODUCTS = 200  # justified by EDA: covers the products with enough basket
# co-occurrence to yield statistically meaningful support/confidence; below the
# top few hundred, products appear too rarely (EDA: median basket-appearance
# count drops below 20) for any rule to be more than noise.


def main():
    df = pd.read_excel(RAW / "Online Retail.xlsx")
    df["StockCode"] = df["StockCode"].astype(str)
    is_cancellation = df["InvoiceNo"].astype(str).str.startswith("C")
    baskets = df[~is_cancellation & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()

    product_names = baskets.groupby("StockCode")["Description"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else "?")
    top_products = baskets["StockCode"].value_counts().head(TOP_N_PRODUCTS).index
    baskets = baskets[baskets["StockCode"].isin(top_products)]

    # Group into a list-of-items-per-basket transaction list.
    transactions = baskets.groupby("InvoiceNo")["StockCode"].apply(lambda s: list(set(s))).tolist()
    # Drop single-item baskets -- no co-occurrence possible, pure noise for rule mining.
    transactions = [t for t in transactions if len(t) >= 2]

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    basket_matrix = pd.DataFrame(te_ary, columns=te.columns_)

    basket_matrix.to_pickle(PROCESSED / "basket_matrix.pkl")
    product_names.to_json(PROCESSED / "product_names.json")

    summary = {
        "top_n_products_kept": TOP_N_PRODUCTS,
        "n_multi_item_baskets": len(transactions),
        "n_products_in_matrix": len(te.columns_),
        "avg_basket_size": round(sum(len(t) for t in transactions) / len(transactions), 2),
    }
    with open(DOCS / "prep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
