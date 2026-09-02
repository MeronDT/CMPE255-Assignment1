"""CRISP-DM Phase 4: Modeling.

For each task: AutoGluon `best_quality` (multi-layer stacking + bagging ensemble)
vs. a single-model LightGBM baseline hill-climbed over a small hyperparameter
grid (the same greedy-search pattern used in every prior project's AutoResearch
phase, here serving as the control group AutoGluon is compared against).
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs" / "eda"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

TIME_LIMIT = 120  # seconds, deliberately capped per 01_business_understanding.md


def lgb_baseline_regression(train, test):
    X_train, y_train = train.drop(columns=["target"]), train["target"]
    X_test, y_test = test.drop(columns=["target"]), test["target"]

    grid = [
        {"num_leaves": 31, "learning_rate": 0.1, "n_estimators": 200},
        {"num_leaves": 63, "learning_rate": 0.05, "n_estimators": 400},
        {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 400},
        {"num_leaves": 127, "learning_rate": 0.05, "n_estimators": 300},
    ]
    best = None
    for params in grid:
        model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        if best is None or rmse < best["rmse"]:
            best = {"params": params, "rmse": rmse, "r2": float(r2_score(y_test, pred))}
    return best


def lgb_baseline_classification(train, test):
    X_train = train.drop(columns=["target"]).copy()
    y_train = (train["target"] == ">50K").astype(int)
    X_test = test.drop(columns=["target"]).copy()
    y_test = (test["target"] == ">50K").astype(int)

    for col in X_train.select_dtypes(include=["object", "category"]).columns:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = pd.Categorical(X_test[col], categories=X_train[col].cat.categories)

    grid = [
        {"num_leaves": 31, "learning_rate": 0.1, "n_estimators": 200},
        {"num_leaves": 63, "learning_rate": 0.05, "n_estimators": 400},
        {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 400},
    ]
    best = None
    for params in grid:
        model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, proba))
        acc = float(accuracy_score(y_test, (proba > 0.5).astype(int)))
        if best is None or auc > best["roc_auc"]:
            best = {"params": params, "roc_auc": auc, "accuracy": acc}
    return best


def run_autogluon_regression(train, test):
    t0 = time.time()
    predictor = TabularPredictor(label="target", problem_type="regression", path=str(MODELS / "ag_housing"), verbosity=0)
    predictor.fit(train, presets="best_quality", time_limit=TIME_LIMIT)
    elapsed = time.time() - t0
    pred = predictor.predict(test.drop(columns=["target"]))
    rmse = float(np.sqrt(mean_squared_error(test["target"], pred)))
    r2 = float(r2_score(test["target"], pred))
    leaderboard = predictor.leaderboard(test, silent=True)
    return {
        "rmse": rmse, "r2": r2, "fit_seconds": round(elapsed, 1),
        "n_base_models": len(leaderboard), "best_model": str(predictor.model_best),
        "leaderboard": leaderboard[["model", "score_test", "score_val", "stack_level"]].to_dict(orient="records"),
    }


def run_autogluon_classification(train, test):
    t0 = time.time()
    predictor = TabularPredictor(label="target", problem_type="binary", eval_metric="roc_auc", path=str(MODELS / "ag_adult"), verbosity=0)
    predictor.fit(train, presets="best_quality", time_limit=TIME_LIMIT)
    elapsed = time.time() - t0
    proba = predictor.predict_proba(test.drop(columns=["target"]))
    pos_class = predictor.class_labels[1]
    auc = float(roc_auc_score((test["target"] == pos_class).astype(int), proba[pos_class]))
    pred = predictor.predict(test.drop(columns=["target"]))
    acc = float(accuracy_score(test["target"], pred))
    leaderboard = predictor.leaderboard(test, silent=True)
    return {
        "roc_auc": auc, "accuracy": acc, "fit_seconds": round(elapsed, 1),
        "n_base_models": len(leaderboard), "best_model": str(predictor.model_best),
        "leaderboard": leaderboard[["model", "score_test", "score_val", "stack_level"]].to_dict(orient="records"),
    }


def main():
    housing_train = pd.read_parquet(PROCESSED / "housing_train.parquet")
    housing_test = pd.read_parquet(PROCESSED / "housing_test.parquet")
    adult_train = pd.read_parquet(PROCESSED / "adult_train.parquet")
    adult_test = pd.read_parquet(PROCESSED / "adult_test.parquet")

    print("=== California Housing: baseline ===")
    housing_baseline = lgb_baseline_regression(housing_train, housing_test)
    print(json.dumps(housing_baseline, indent=2))

    print("=== California Housing: AutoGluon ===")
    housing_ag = run_autogluon_regression(housing_train, housing_test)
    print(json.dumps({k: v for k, v in housing_ag.items() if k != "leaderboard"}, indent=2))

    print("=== Adult Income: baseline ===")
    adult_baseline = lgb_baseline_classification(adult_train, adult_test)
    print(json.dumps(adult_baseline, indent=2))

    print("=== Adult Income: AutoGluon ===")
    adult_ag = run_autogluon_classification(adult_train, adult_test)
    print(json.dumps({k: v for k, v in adult_ag.items() if k != "leaderboard"}, indent=2))

    results = {
        "housing": {"baseline": housing_baseline, "autogluon": housing_ag,
                    "improvement_pct": round((1 - housing_ag["rmse"] / housing_baseline["rmse"]) * 100, 2)},
        "adult": {"baseline": adult_baseline, "autogluon": adult_ag,
                  "improvement_pct": round((adult_ag["roc_auc"] - adult_baseline["roc_auc"]) / adult_baseline["roc_auc"] * 100, 2)},
    }
    with open(DOCS / "modeling_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
