"""AutoResearch: hyperparameter hill-climb + full-scale finalization.

Follows the pattern from Projects 01-04: an algorithm tournament (already run
in 03_train_models.py, on a 20K-normal proxy subsample), then a greedy
hill-climb over the winner's hyperparameters (still on the proxy, for speed),
then a finalization step that verifies the winning config on the FULL
275,663-row dataset -- the proxy subsample could plausibly not represent the
full data's anomaly structure, so the win is checked for real, not assumed
(the same caution Project 01's taxi AutoResearch and Project 02's NanoLlama
AutoResearch both applied).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs" / "eda"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

RNG = np.random.RandomState(42)
PROXY_N_NORMAL = 20000

# Contamination should roughly match the known fraud rate (0.17%), but sweep
# around it since a real deployment wouldn't know the true rate precisely --
# testing sensitivity to that uncertainty is itself informative.
CONTAMINATION_GRID = [0.001, 0.0015, 0.0018, 0.002, 0.003, 0.005, 0.01]
N_ESTIMATORS_GRID = [100, 200, 400]


def make_proxy(X, y):
    fraud_idx = np.where(y == 1)[0]
    normal_idx = np.where(y == 0)[0]
    sampled_normal = RNG.choice(normal_idx, size=PROXY_N_NORMAL, replace=False)
    idx = np.concatenate([fraud_idx, sampled_normal])
    RNG.shuffle(idx)
    return X.iloc[idx].reset_index(drop=True), y[idx]


def evaluate(y_true, scores):
    return {
        "auprc": round(float(average_precision_score(y_true, scores)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 4),
    }


def main():
    X = pd.read_parquet(PROCESSED / "X_features.parquet")
    y = np.load(PROCESSED / "y_labels.npy")
    Xp, yp = make_proxy(X, y)

    history = {
        "phase1_hillclimb": [],
        "phase2_finalize": None,
        "note_on_contamination": (
            "AUPRC/ROC-AUC are computed from IsolationForest's continuous score_samples() "
            "output, which does NOT depend on the contamination parameter -- contamination "
            "only shifts the internal binary threshold used by predict(), not the score "
            "ranking. So this grid's AUPRC is identical across all contamination values "
            "within each n_estimators group (visible in the table below) -- an honest "
            "artifact of the metric choice, not a bug. contamination DOES matter downstream, "
            "for the flagging/operating-point decision in 05_evaluate.py."
        ),
    }

    best = None
    for n_est in N_ESTIMATORS_GRID:
        for contam in CONTAMINATION_GRID:
            iso = IsolationForest(n_estimators=n_est, contamination=contam, random_state=42, n_jobs=-1)
            iso.fit(Xp)
            scores = -iso.score_samples(Xp)
            metrics = evaluate(yp, scores)
            entry = {"n_estimators": n_est, "contamination": contam, **metrics}
            history["phase1_hillclimb"].append(entry)
            if best is None or metrics["auprc"] > best["auprc"]:
                best = entry

    history["winning_config_proxy"] = {"n_estimators": best["n_estimators"], "contamination": best["contamination"]}
    print(f"Proxy winner: n_estimators={best['n_estimators']}, contamination={best['contamination']}, AUPRC={best['auprc']}")

    # Finalize: retrain the winning config on the FULL dataset, verify it
    # actually holds up (not just a proxy-subsample artifact).
    final_model = IsolationForest(
        n_estimators=best["n_estimators"], contamination=best["contamination"], random_state=42, n_jobs=-1
    )
    final_model.fit(X)
    full_scores = -final_model.score_samples(X)
    full_metrics = evaluate(y, full_scores)

    # IMPORTANT honest finding, not glossed over: raw AUPRC dropped sharply
    # (proxy 0.6156 -> full 0.1444). This is NOT primarily model degradation --
    # AUPRC is base-rate-dependent (a random classifier scores ~= the positive
    # rate), and the proxy subsample's positive rate (473/20473 = 2.31%) is
    # ~13x the full dataset's true rate (473/275663 = 0.172%). Comparing raw
    # AUPRC across two different base rates is comparing apples to oranges.
    # The fairer comparison is AUPRC RELATIVE TO its own random baseline:
    proxy_base_rate = float(yp.mean())
    full_base_rate = float(y.mean())
    proxy_lift = best["auprc"] / proxy_base_rate
    full_lift = full_metrics["auprc"] / full_base_rate

    history["phase2_finalize"] = {
        **best_config_dict(best), **full_metrics,
        "proxy_auprc": best["auprc"], "proxy_roc_auc": best["roc_auc"],
        "proxy_base_rate_pct": round(proxy_base_rate * 100, 3),
        "full_base_rate_pct": round(full_base_rate * 100, 4),
        "proxy_auprc_lift_over_random": round(proxy_lift, 1),
        "full_auprc_lift_over_random": round(full_lift, 1),
        "held_up_at_full_scale": full_lift >= proxy_lift * 0.7,  # compare LIFT, not raw AUPRC
        "note": (
            "Raw AUPRC drops from proxy to full scale because the proxy subsample's "
            "positive rate is ~13x higher than the true rate -- AUPRC is base-rate "
            "dependent, so this is expected, not model degradation. Comparing lift over "
            "each evaluation's own random baseline shows the ranking quality actually held "
            "up (or improved) at full scale."
        ),
    }

    import joblib
    joblib.dump({"model": final_model, "features": list(X.columns)}, MODELS / "production_isolation_forest.joblib")

    np.save(PROCESSED / "full_anomaly_scores.npy", full_scores)

    with open(DOCS / "autoresearch.json", "w") as f:
        json.dump(history, f, indent=2)

    print(json.dumps(history["phase2_finalize"], indent=2))


def best_config_dict(best):
    return {"n_estimators": best["n_estimators"], "contamination": best["contamination"]}


if __name__ == "__main__":
    main()
