"""CRISP-DM Phase 2+3: Data Understanding + Preparation.

Aggregates the Online Retail transaction log into a daily revenue series,
handles missing calendar days (the store isn't open every day -- weekends/
holidays have zero or near-zero transactions, and a few days have NO rows at
all in the raw log), and produces a clean, continuous daily series ready for
forecasting.
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
DOCS.mkdir(parents=True, exist_ok=True)

TEST_DAYS = 30  # held-out final window, a proper time-series split (never shuffled)


def main():
    df = pd.read_excel(RAW / "Online Retail.xlsx")
    is_cancellation = df["InvoiceNo"].astype(str).str.startswith("C")
    df = df[~is_cancellation & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["Date"] = df["InvoiceDate"].dt.date

    daily = df.groupby("Date")["Revenue"].sum().reset_index()
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily = daily.set_index("Date").asfreq("D")  # forces a continuous daily calendar

    n_missing_days = int(daily["Revenue"].isnull().sum())
    # Missing calendar days (no transactions logged at all) are filled with 0 --
    # a real absence of sales, not something to interpolate away, since this is
    # a real retailer with real closed/zero-activity days, not sensor dropout.
    daily["Revenue"] = daily["Revenue"].fillna(0)

    eda = {
        "date_range": [str(daily.index.min().date()), str(daily.index.max().date())],
        "n_calendar_days": len(daily),
        "n_missing_days_filled_zero": n_missing_days,
        "revenue_stats": daily["Revenue"].describe().to_dict(),
        "n_zero_revenue_days": int((daily["Revenue"] == 0).sum()),
        "weekday_avg_revenue": daily.groupby(daily.index.dayofweek)["Revenue"].mean().to_dict(),
    }

    train = daily.iloc[:-TEST_DAYS]
    test = daily.iloc[-TEST_DAYS:]

    train.to_csv(PROCESSED / "daily_revenue_train.csv")
    test.to_csv(PROCESSED / "daily_revenue_test.csv")
    daily.to_csv(PROCESSED / "daily_revenue_full.csv")

    eda["n_train_days"] = len(train)
    eda["n_test_days"] = len(test)

    with open(DOCS / "eda_and_prep_summary.json", "w") as f:
        json.dump(eda, f, indent=2, default=str)
    print(json.dumps(eda, indent=2, default=str))


if __name__ == "__main__":
    main()
