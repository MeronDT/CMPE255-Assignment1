"""CRISP-DM Phase 4: Modeling.

Trains a LightGBM regressor for each target (trip duration, fare amount)
against a naive baseline (predict the training mean), with a small random
hyperparameter search per target. Every trial's params + validation score are
logged to model_search_log.json so the admin dashboard can show the search,
not just the winning config — this is the "hill climbing" search history a
data scientist would want to audit.
"""

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
DOCS = ROOT / "docs" / "eda"
MODELS.mkdir(parents=True, exist_ok=True)

CAT_COLS = ["pickup_borough", "dropoff_borough", "PULocationID", "DOLocationID"]

SEARCH_SPACE = [
    {"num_leaves": 31, "learning_rate": 0.1, "n_estimators": 300, "min_child_samples": 30},
    {"num_leaves": 63, "learning_rate": 0.05, "n_estimators": 500, "min_child_samples": 50},
    {"num_leaves": 127, "learning_rate": 0.05, "n_estimators": 500, "min_child_samples": 80},
    {"num_leaves": 63, "learning_rate": 0.1, "n_estimators": 400, "min_child_samples": 20},
    {"num_leaves": 15, "learning_rate": 0.1, "n_estimators": 300, "min_child_samples": 30},
]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_target(target, train_df, test_df, feature_cols):
    X = train_df[feature_cols]
    y = train_df[target]
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    baseline_pred = np.full_like(y_val, fill_value=y_tr.mean(), dtype=float)
    baseline = {
        "rmse": rmse(y_val, baseline_pred),
        "mae": float(mean_absolute_error(y_val, baseline_pred)),
        "r2": float(r2_score(y_val, baseline_pred)),
    }

    trials = []
    best_model, best_rmse, best_params = None, float("inf"), None

    for i, params in enumerate(SEARCH_SPACE):
        t0 = time.time()
        model = lgb.LGBMRegressor(
            objective="regression",
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
            **params,
        )
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            categorical_feature=CAT_COLS,
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        pred = model.predict(X_val)
        trial_rmse = rmse(y_val, pred)
        trial_mae = float(mean_absolute_error(y_val, pred))
        trial_r2 = float(r2_score(y_val, pred))
        elapsed = time.time() - t0

        trials.append(
            {
                "trial": i,
                "params": params,
                "val_rmse": trial_rmse,
                "val_mae": trial_mae,
                "val_r2": trial_r2,
                "best_iteration": int(model.best_iteration_ or params["n_estimators"]),
                "train_seconds": round(elapsed, 1),
            }
        )
        print(f"  [{target}] trial {i}: rmse={trial_rmse:.4f} mae={trial_mae:.4f} r2={trial_r2:.4f} ({elapsed:.1f}s)")

        if trial_rmse < best_rmse:
            best_rmse, best_model, best_params = trial_rmse, model, params

    # Final holdout evaluation on the untouched test set
    X_test, y_test = test_df[feature_cols], test_df[target]
    test_pred = best_model.predict(X_test)
    test_metrics = {
        "rmse": rmse(y_test, test_pred),
        "mae": float(mean_absolute_error(y_test, test_pred)),
        "r2": float(r2_score(y_test, test_pred)),
    }

    importance = sorted(
        zip(feature_cols, best_model.feature_importances_.tolist()),
        key=lambda kv: kv[1],
        reverse=True,
    )

    return best_model, {
        "target": target,
        "baseline_val_metrics": baseline,
        "search_trials": trials,
        "best_params": best_params,
        "best_val_rmse": best_rmse,
        "test_metrics": test_metrics,
        "feature_importance": [{"feature": f, "importance": imp} for f, imp in importance],
        "residual_sample": {
            "y_true": y_test.iloc[:2000].tolist(),
            "y_pred": test_pred[:2000].tolist(),
        },
    }


def main():
    train_df = pd.read_parquet(PROCESSED / "train.parquet")
    test_df = pd.read_parquet(PROCESSED / "test.parquet")

    for col in CAT_COLS:
        train_df[col] = train_df[col].astype("category")
        test_df[col] = test_df[col].astype("category", copy=False)
        test_df[col] = test_df[col].cat.set_categories(train_df[col].cat.categories)

    feature_cols = [c for c in train_df.columns if c not in ("duration_min", "fare_amount")]

    report = {"generated_at": pd.Timestamp.utcnow().isoformat(), "targets": {}}

    for target, filename in [("duration_min", "duration_model.txt"), ("fare_amount", "fare_model.txt")]:
        print(f"\n=== Training target: {target} ===")
        model, result = train_target(target, train_df, test_df, feature_cols)
        model.booster_.save_model(str(MODELS / filename))
        report["targets"][target] = result
        print(f"  best_params={result['best_params']} test_rmse={result['test_metrics']['rmse']:.4f} test_r2={result['test_metrics']['r2']:.4f}")

    with open(DOCS / "model_search_log.json", "w") as f:
        json.dump(report, f, indent=2)

    # Persist the categorical dtype's category lists so the serving layer can
    # rebuild identical dtypes for inference (LightGBM requires matching codes).
    category_maps = {col: train_df[col].cat.categories.astype(str).tolist() for col in CAT_COLS}
    with open(MODELS / "category_maps.json", "w") as f:
        json.dump(category_maps, f)

    with open(MODELS / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    print(f"\nWrote model search report -> {DOCS / 'model_search_log.json'}")
    print(f"Wrote models -> {MODELS}")


if __name__ == "__main__":
    main()
