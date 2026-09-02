"""CRISP-DM Phase 2+3: Data Understanding + Preparation, for both tasks.

California Housing (regression) via sklearn.datasets.fetch_california_housing
(built-in, no download needed beyond sklearn's own cached copy).
Adult Census Income (classification) via sklearn.datasets.fetch_openml("adult"),
OpenML's public mirror of the Kaggle/UCI Adult dataset.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs" / "eda"
PROCESSED.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)


def prepare_housing():
    data = fetch_california_housing(as_frame=True)
    df = data.frame.rename(columns={"MedHouseVal": "target"})
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train.to_parquet(PROCESSED / "housing_train.parquet")
    test.to_parquet(PROCESSED / "housing_test.parquet")
    return {
        "task": "regression", "n_rows": len(df), "n_features": df.shape[1] - 1,
        "n_train": len(train), "n_test": len(test),
        "target_stats": df["target"].describe().to_dict(),
        "n_missing": int(df.isnull().sum().sum()),
    }


def prepare_adult():
    data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = data.frame.rename(columns={"class": "target"})
    df["target"] = df["target"].astype(str)
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["target"])
    train.to_parquet(PROCESSED / "adult_train.parquet")
    test.to_parquet(PROCESSED / "adult_test.parquet")
    return {
        "task": "classification", "n_rows": len(df), "n_features": df.shape[1] - 1,
        "n_train": len(train), "n_test": len(test),
        "class_balance": df["target"].value_counts(normalize=True).to_dict(),
        "n_missing": int(df.isnull().sum().sum()),
    }


def main():
    summary = {"california_housing": prepare_housing(), "adult_income": prepare_adult()}
    with open(DOCS / "prep_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
