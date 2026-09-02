"""Retrain each target's AutoResearch-winning hyperparameters on the FULL
training set and compare against the currently deployed model on the real
held-out test set. Redeploys only if it's actually better."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
DOCS = ROOT / "docs" / "eda"
CAT_COLS = ["pickup_borough", "dropoff_borough", "PULocationID", "DOLocationID"]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    train_df = pd.read_parquet(PROCESSED / "train.parquet")
    test_df = pd.read_parquet(PROCESSED / "test.parquet")
    for col in CAT_COLS:
        train_df[col] = train_df[col].astype("category")
        test_df[col] = test_df[col].astype("category", copy=False)
        test_df[col] = test_df[col].cat.set_categories(train_df[col].cat.categories)

    with open(MODELS / "feature_cols.json") as f:
        feature_cols = json.load(f)
    with open(DOCS / "autoresearch_history.json") as f:
        autoresearch = json.load(f)
    with open(DOCS / "model_search_log.json") as f:
        current_log = json.load(f)

    filename_map = {"duration_min": "duration_model.txt", "fare_amount": "fare_model.txt"}
    finalize_report = {}

    for target, filename in filename_map.items():
        best_params = autoresearch["targets"][target]["phase3_hill_climbing"]["best_params"]
        current_test_rmse = current_log["targets"][target]["test_metrics"]["rmse"]

        model = lgb.LGBMRegressor(random_state=42, verbosity=-1, n_jobs=-1, **best_params)
        model.fit(train_df[feature_cols], train_df[target], categorical_feature=CAT_COLS)
        pred = model.predict(test_df[feature_cols])
        new_metrics = {
            "rmse": rmse(test_df[target], pred),
            "mae": float(mean_absolute_error(test_df[target], pred)),
            "r2": float(r2_score(test_df[target], pred)),
        }

        redeployed = new_metrics["rmse"] < current_test_rmse
        print(f"{target}: current_test_rmse={current_test_rmse:.4f} autoresearch_full_data_rmse={new_metrics['rmse']:.4f} -> {'REDEPLOYED' if redeployed else 'kept existing (no improvement on full data)'}")

        if redeployed:
            model.booster_.save_model(str(MODELS / filename))
            current_log["targets"][target]["test_metrics"] = new_metrics
            current_log["targets"][target]["best_params"] = best_params
            current_log["targets"][target]["autoresearch_finalized"] = True

        finalize_report[target] = {
            "autoresearch_params": best_params,
            "previous_test_rmse": current_test_rmse,
            "full_data_retrain_test_metrics": new_metrics,
            "redeployed": redeployed,
        }

    with open(DOCS / "autoresearch_finalize.json", "w") as f:
        json.dump(finalize_report, f, indent=2)
    with open(DOCS / "model_search_log.json", "w") as f:
        json.dump(current_log, f, indent=2)

    print(f"\nWrote {DOCS / 'autoresearch_finalize.json'}")


if __name__ == "__main__":
    main()
