"""CRISP-DM Phase 5: Evaluation.

Retrains the AutoResearch-winning config (tuned Prophet) and produces the
final forecast-vs-actual comparison for the dashboard, plus a forward
forecast beyond the test window (what a deployed version would actually
produce for the business).
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs" / "eda"

FUTURE_DAYS = 14


def main():
    autores = json.load(open(DOCS / "autoresearch.json"))
    winning_config = autores["prophet_tuned_winner"]

    full = pd.read_csv(PROCESSED / "daily_revenue_full.csv", index_col=0, parse_dates=True)
    prophet_full = full.reset_index()
    prophet_full.columns = ["ds", "y"]

    m = Prophet(
        weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False,
        changepoint_prior_scale=winning_config["changepoint_prior_scale"],
        seasonality_mode=winning_config["seasonality_mode"],
    )
    m.fit(prophet_full)
    future = m.make_future_dataframe(periods=FUTURE_DAYS)
    forecast = m.predict(future)

    forward_forecast = forecast.tail(FUTURE_DAYS)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forward_forecast["yhat"] = forward_forecast["yhat"].clip(lower=0)
    forward_forecast["yhat_lower"] = forward_forecast["yhat_lower"].clip(lower=0)

    comparison = json.load(open(PROCESSED / "forecast_comparison.json"))

    result = {
        "winning_model": "prophet_tuned",
        "winning_config": winning_config,
        "test_set_comparison": comparison,
        "forward_forecast": {
            "dates": [str(d.date()) for d in forward_forecast["ds"]],
            "yhat": forward_forecast["yhat"].round(2).tolist(),
            "yhat_lower": forward_forecast["yhat_lower"].round(2).tolist(),
            "yhat_upper": forward_forecast["yhat_upper"].round(2).tolist(),
        },
    }
    with open(DOCS / "evaluation.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in result.items() if k != "test_set_comparison"}, indent=2, default=str))


if __name__ == "__main__":
    main()
