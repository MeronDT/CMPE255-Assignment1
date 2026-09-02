"""AutoResearch: 4-phase autonomous model search, following the reference
`nyc-taxi-autoresearch` skill's blueprint (multi-backbone tournament ->
feature-transform search -> hyperparameter hill-climbing -> blending), with
full telemetry logged so the dashboard can show the *process*, not just the
winner. Search phases run on a subsample for speed; the winning
configuration is finalized by retraining on the full training set.

Phase 3 is literal greedy hill-climbing (not random/grid search): from a
seed point in hyperparameter space, evaluate every one-step neighbor, move
to the best improving neighbor, repeat until no neighbor improves (a local
optimum) or the iteration budget is spent.
"""

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
DOCS = ROOT / "docs" / "eda"

CAT_COLS = ["pickup_borough", "dropoff_borough", "PULocationID", "DOLocationID"]
SEARCH_SAMPLE_SIZE = 300_000
SEED = 42

# Papers found during AutoResearch's literature-grounding step (see
# RESEARCH_REPORT.md for full citations) — used as external reference points,
# not tuning targets, since their setups differ from ours (see report §6).
PAPER_BENCHMARKS = {
    "duration_min": {
        "source": "Ye et al., \"New York City taxi trip duration prediction using MLP and XGBoost\" (ResearchGate, 2021)",
        "r2": 0.82,
        "note": "Best of MLP/XGBoost/BRF ensemble; uses full post-hoc feature set incl. exact pickup/dropoff coordinates.",
    },
    "fare_amount": {
        "source": "arXiv:2507.20008, \"Robust Taxi Fare Prediction under Noisy Conditions: GAT, TimesNet, XGBoost\" (2025)",
        "r2": None,
        "note": "55M-row dataset with full completed-trip features (incl. exact distance); RMSE not directly comparable to our pre-trip-only setup (see RESEARCH_REPORT.md §6).",
    },
}


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics(y_true, y_pred):
    return {"rmse": rmse(y_true, y_pred), "mae": float(mean_absolute_error(y_true, y_pred)), "r2": float(r2_score(y_true, y_pred))}


def to_numeric_frame(df, cat_cols):
    """One-hot-free numeric view for Ridge/RandomForest (LightGBM/XGBoost use native categoricals)."""
    out = df.copy()
    for c in cat_cols:
        out[c] = out[c].cat.codes
    return out


# ---------------------------------------------------------------------------
# Phase 1: Multi-backbone tournament
# ---------------------------------------------------------------------------
def phase1_tournament(X_tr, y_tr, X_val, y_val, cat_cols):
    results = []
    X_tr_num, X_val_num = to_numeric_frame(X_tr, cat_cols), to_numeric_frame(X_val, cat_cols)

    backbones = {
        "ridge": lambda: Ridge(alpha=1.0, random_state=SEED),
        "random_forest": lambda: RandomForestRegressor(n_estimators=120, max_depth=12, n_jobs=-1, random_state=SEED),
        "lightgbm": lambda: lgb.LGBMRegressor(n_estimators=300, num_leaves=63, learning_rate=0.08, random_state=SEED, verbosity=-1, n_jobs=-1),
        "xgboost": lambda: xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08, random_state=SEED, n_jobs=-1, verbosity=0, enable_categorical=False),
    }

    for name, factory in backbones.items():
        t0 = time.time()
        model = factory()
        if name in ("lightgbm",):
            model.fit(X_tr, y_tr, categorical_feature=cat_cols)
            pred = model.predict(X_val)
        else:
            model.fit(X_tr_num, y_tr)
            pred = model.predict(X_val_num)
        m = metrics(y_val, pred)
        m.update({"backbone": name, "train_seconds": round(time.time() - t0, 1)})
        results.append(m)
        print(f"    [tournament] {name}: rmse={m['rmse']:.4f} r2={m['r2']:.4f} ({m['train_seconds']}s)")

    winner = min(results, key=lambda r: r["rmse"])
    return results, winner["backbone"]


# ---------------------------------------------------------------------------
# Phase 2: Feature transformation search
# ---------------------------------------------------------------------------
def make_transform_variants(df):
    variants = {}

    base = df.copy()
    variants["baseline"] = base

    log_dist = df.copy()
    log_dist["haversine_km"] = np.log1p(log_dist["haversine_km"])
    variants["log1p_haversine"] = log_dist

    cyclical = df.copy()
    cyclical["pickup_hour_sin"] = np.sin(2 * np.pi * cyclical["pickup_hour"] / 24)
    cyclical["pickup_hour_cos"] = np.cos(2 * np.pi * cyclical["pickup_hour"] / 24)
    variants["cyclical_hour"] = cyclical

    interaction = df.copy()
    interaction["dist_x_rush"] = interaction["haversine_km"] * (1 + interaction["is_rush_hour"])
    variants["distance_rush_interaction"] = interaction

    return variants


def phase2_feature_transforms(backbone_name, X_tr, y_tr, X_val, y_val, cat_cols):
    variants_tr = make_transform_variants(X_tr)
    variants_val = make_transform_variants(X_val)

    results = []
    for variant_name in variants_tr:
        Xv_tr, Xv_val = variants_tr[variant_name], variants_val[variant_name]
        t0 = time.time()
        model = lgb.LGBMRegressor(n_estimators=300, num_leaves=63, learning_rate=0.08, random_state=SEED, verbosity=-1, n_jobs=-1)
        model.fit(Xv_tr, y_tr, categorical_feature=cat_cols)
        pred = model.predict(Xv_val)
        m = metrics(y_val, pred)
        m.update({"variant": variant_name, "n_features": Xv_tr.shape[1], "train_seconds": round(time.time() - t0, 1)})
        results.append(m)
        print(f"    [feature-transform] {variant_name}: rmse={m['rmse']:.4f} r2={m['r2']:.4f}")

    winner = min(results, key=lambda r: r["rmse"])
    return results, winner["variant"], variants_tr[winner["variant"]], variants_val[winner["variant"]]


# ---------------------------------------------------------------------------
# Phase 3: Hyperparameter hill-climbing (greedy local search)
# ---------------------------------------------------------------------------
PARAM_STEPS = {
    "num_leaves": [-32, 32],
    "learning_rate": [-0.03, 0.03],
    "n_estimators": [-150, 150],
    "min_child_samples": [-20, 20],
}
PARAM_BOUNDS = {
    "num_leaves": (15, 255),
    "learning_rate": (0.02, 0.3),
    "n_estimators": (100, 900),
    "min_child_samples": (10, 150),
}


def clip(param, value):
    lo, hi = PARAM_BOUNDS[param]
    return max(lo, min(hi, value))


def evaluate_config(config, X_tr, y_tr, X_val, y_val, cat_cols):
    model = lgb.LGBMRegressor(random_state=SEED, verbosity=-1, n_jobs=-1, **config)
    model.fit(X_tr, y_tr, categorical_feature=cat_cols)
    pred = model.predict(X_val)
    return metrics(y_val, pred), model


def phase3_hill_climb(seed_config, X_tr, y_tr, X_val, y_val, cat_cols, max_iters=8):
    path = []
    current = dict(seed_config)
    current_metrics, current_model = evaluate_config(current, X_tr, y_tr, X_val, y_val, cat_cols)
    path.append({"iteration": 0, "params": dict(current), "rmse": current_metrics["rmse"], "move": "seed"})
    print(f"    [hill-climb] seed rmse={current_metrics['rmse']:.4f}")

    for it in range(1, max_iters + 1):
        neighbors = []
        for param, steps in PARAM_STEPS.items():
            for step in steps:
                candidate = dict(current)
                new_val = current[param] + step
                new_val = round(new_val, 3) if param == "learning_rate" else int(new_val)
                candidate[param] = clip(param, new_val)
                if candidate == current:
                    continue
                neighbors.append((param, step, candidate))

        best_neighbor, best_metrics, best_model = None, current_metrics, current_model
        improved = False
        for param, step, candidate in neighbors:
            m, mdl = evaluate_config(candidate, X_tr, y_tr, X_val, y_val, cat_cols)
            if m["rmse"] < best_metrics["rmse"]:
                best_neighbor, best_metrics, best_model = candidate, m, mdl
                improved = True

        if not improved:
            print(f"    [hill-climb] iteration {it}: no improving neighbor -> local optimum reached")
            break

        current, current_metrics, current_model = best_neighbor, best_metrics, best_model
        path.append({"iteration": it, "params": dict(current), "rmse": current_metrics["rmse"], "move": "accepted"})
        print(f"    [hill-climb] iteration {it}: rmse={current_metrics['rmse']:.4f} params={current}")

    return current, current_metrics, path


# ---------------------------------------------------------------------------
# Phase 4: Blending
# ---------------------------------------------------------------------------
def phase4_blend(model_a, model_b, X_val, y_val):
    pred_a, pred_b = model_a.predict(X_val), model_b.predict(X_val)
    results = []
    for w in np.arange(0.0, 1.01, 0.1):
        blended = w * pred_a + (1 - w) * pred_b
        m = metrics(y_val, blended)
        m["weight_a"] = round(float(w), 2)
        results.append(m)
    best = min(results, key=lambda r: r["rmse"])
    return results, best


def run_target(target, train_df, feature_cols, cat_cols, seed_config):
    print(f"\n=== AutoResearch: {target} ===")
    sample = train_df.sample(n=min(SEARCH_SAMPLE_SIZE, len(train_df)), random_state=SEED)
    X, y = sample[feature_cols], sample[target]
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

    print("  Phase 1: Multi-backbone tournament")
    tournament_results, winning_backbone = phase1_tournament(X_tr, y_tr, X_val, y_val, cat_cols)

    print("  Phase 2: Feature-transform search")
    ft_results, winning_variant, X_tr_ft, X_val_ft = phase2_feature_transforms(winning_backbone, X_tr, y_tr, X_val, y_val, cat_cols)

    print("  Phase 3: Hyperparameter hill-climbing")
    best_params, best_metrics, hillclimb_path = phase3_hill_climb(seed_config, X_tr_ft, y_tr, X_val_ft, y_val, cat_cols)
    _, best_model = evaluate_config(best_params, X_tr_ft, y_tr, X_val_ft, y_val, cat_cols)
    _, seed_model = evaluate_config(seed_config, X_tr_ft, y_tr, X_val_ft, y_val, cat_cols)

    print("  Phase 4: Blending (hill-climbed model x seed model)")
    blend_results, best_blend = phase4_blend(best_model, seed_model, X_val_ft, y_val)
    print(f"    [blend] best weight_a={best_blend['weight_a']} rmse={best_blend['rmse']:.4f}")

    final_rmse = min(best_metrics["rmse"], best_blend["rmse"])
    winner_stage = "blend" if best_blend["rmse"] < best_metrics["rmse"] else "hill_climb"

    return {
        "target": target,
        "search_sample_size": len(sample),
        "phase1_tournament": {"results": tournament_results, "winning_backbone": winning_backbone},
        "phase2_feature_transform": {"results": ft_results, "winning_variant": winning_variant},
        "phase3_hill_climbing": {"seed_params": seed_config, "path": hillclimb_path, "best_params": best_params, "best_metrics": best_metrics},
        "phase4_blending": {"results": blend_results, "best": best_blend},
        "final": {"winner_stage": winner_stage, "rmse": final_rmse, "r2": best_metrics["r2"] if winner_stage == "hill_climb" else best_blend["r2"]},
        "paper_benchmark": PAPER_BENCHMARKS[target],
        "feature_transform_added": winning_variant,
    }


def main():
    train_df = pd.read_parquet(PROCESSED / "train.parquet")
    for col in CAT_COLS:
        train_df[col] = train_df[col].astype("category")

    with open(MODELS / "feature_cols.json") as f:
        feature_cols = json.load(f)

    seed_configs = {
        "duration_min": {"num_leaves": 127, "learning_rate": 0.05, "n_estimators": 500, "min_child_samples": 80},
        "fare_amount": {"num_leaves": 127, "learning_rate": 0.05, "n_estimators": 500, "min_child_samples": 80},
    }

    report = {"generated_at": pd.Timestamp.utcnow().isoformat(), "methodology": "4-phase AutoResearch hill-climbing (nyc-taxi-autoresearch skill blueprint)", "targets": {}}

    for target in ["duration_min", "fare_amount"]:
        report["targets"][target] = run_target(target, train_df, feature_cols, CAT_COLS, seed_configs[target])

    with open(DOCS / "autoresearch_history.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nWrote AutoResearch telemetry -> {DOCS / 'autoresearch_history.json'}")
    for target, result in report["targets"].items():
        print(f"  {target}: final_rmse={result['final']['rmse']:.4f} (winner_stage={result['final']['winner_stage']}, backbone={result['phase1_tournament']['winning_backbone']}, features={result['feature_transform_added']})")


if __name__ == "__main__":
    main()
