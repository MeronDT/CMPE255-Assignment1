"""CRISP-DM Phase 4: Modeling (baseline tournament).

Seasonal Naive (mandatory floor) vs. Holt-Winters Exponential Smoothing vs.
Prophet, evaluated on a proper held-out FINAL-30-days test window (never
shuffled -- a random split would leak future information into training,
the classic time-series-specific mistake this project deliberately avoids).
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs" / "eda"

SEASON_PERIOD = 7  # weekly seasonality (store closed Saturdays, per EDA)


def mape(y_true, y_pred):
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(y_true, y_pred, name):
    return {
        "model": name,
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "mape_pct": round(mape(np.array(y_true), np.array(y_pred)), 2),
    }


def seasonal_naive(train, n_forecast):
    last_season = train["Revenue"].values[-SEASON_PERIOD:]
    reps = int(np.ceil(n_forecast / SEASON_PERIOD))
    return np.tile(last_season, reps)[:n_forecast]


def main():
    train = pd.read_csv(PROCESSED / "daily_revenue_train.csv", index_col=0, parse_dates=True)
    test = pd.read_csv(PROCESSED / "daily_revenue_test.csv", index_col=0, parse_dates=True)
    n_forecast = len(test)
    y_test = test["Revenue"].values

    results = {}

    # 1. Seasonal Naive (mandatory floor)
    pred_naive = seasonal_naive(train, n_forecast)
    results["seasonal_naive"] = evaluate(y_test, pred_naive, "seasonal_naive")

    # 2. Holt-Winters Exponential Smoothing
    hw = ExponentialSmoothing(
        train["Revenue"], trend="add", seasonal="add", seasonal_periods=SEASON_PERIOD, damped_trend=True
    ).fit()
    pred_hw = hw.forecast(n_forecast).values
    results["holt_winters"] = evaluate(y_test, pred_hw, "holt_winters")

    # 3. Prophet
    prophet_train = train.reset_index().rename(columns={"index": "ds", "Revenue": "y"})
    prophet_train.columns = ["ds", "y"]
    m = Prophet(weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False)
    m.fit(prophet_train)
    future = m.make_future_dataframe(periods=n_forecast)
    forecast = m.predict(future)
    pred_prophet = forecast["yhat"].values[-n_forecast:]
    pred_prophet = np.clip(pred_prophet, 0, None)  # revenue can't be negative
    results["prophet"] = evaluate(y_test, pred_prophet, "prophet")

    winner = min(results, key=lambda k: results[k]["mape_pct"])

    forecasts_export = {
        "dates": [str(d.date()) for d in test.index],
        "actual": y_test.tolist(),
        "seasonal_naive": pred_naive.tolist(),
        "holt_winters": pred_hw.tolist(),
        "prophet": pred_prophet.tolist(),
    }

    summary = {"model_comparison": results, "winner": winner, "test_days": n_forecast}
    with open(DOCS / "modeling_baseline.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(PROCESSED / "forecast_comparison.json", "w") as f:
        json.dump(forecasts_export, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
