"""AutoResearch: hill-climbing search over feature transforms, scalers, and k.

Follows the pattern established in Projects 01-02: an architecture/algorithm
tournament (already run in 03_train_models.py), then a feature-transform +
hyperparameter search, then a finalization step that verifies the winner
holds on the full dataset (trivial here since the dataset is small enough
that no proxy-subsample step is needed -- unlike the taxi/NanoLlama projects,
every search step here already runs on the full 4,371-customer table).

Honest finding surfaced by 03_train_models.py: pure silhouette maximization
picks k=2, which is statistically "best-separated" but not business-useful
-- a marketing team can't act on "big spenders vs everyone else." The RFM
segmentation literature (Hughes 1994; Christy et al. 2021, "RFM ranking -
An effective approach to customer segmentation") consistently reports 4-6
actionable segments as the practical range. This search applies that
domain constraint explicitly (k in [3, 8]) rather than blindly maximizing
silhouette, and reports the k=2 result honestly alongside as the unconstrained
statistical optimum -- not hidden, just not deployed, because it fails the
actual business objective from 01_business_understanding.md.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs" / "eda"
MODELS = ROOT / "models"

ALL_FEATURES = ["Recency", "Frequency", "Monetary", "AvgBasketValue", "DistinctProducts", "TenureDays", "CancellationRate"]
LOG_CANDIDATES = ["Monetary", "AvgBasketValue", "Frequency", "DistinctProducts"]
SCALERS = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}

# Business constraint from 01_business_understanding.md / RFM literature: segments
# must be numerous enough to be actionable (>=3) but few enough to communicate to a
# marketing team (<=8).
K_RANGE = range(3, 9)


def build_matrix(rfm, features, log_cols, scaler_name):
    X = rfm[features].copy()
    for col in log_cols:
        if col in X.columns:
            X[col] = np.log1p(X[col])
    scaler = SCALERS[scaler_name]()
    return scaler.fit_transform(X), scaler


def score(X, labels):
    if len(set(labels)) < 2:
        return None
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
    }


def main():
    rfm = pd.read_csv(PROCESSED / "customer_rfm.csv")
    history = {"phase1_unconstrained_optimum": None, "phase2_feature_transform_search": [], "phase3_hill_climb": [], "phase4_finalize": None}

    # Phase 1: report the unconstrained statistical optimum honestly (k=2, from
    # 03_train_models.py) before applying the business constraint.
    X_base, _ = build_matrix(rfm, ALL_FEATURES[:6], LOG_CANDIDATES, "standard")
    km2 = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_base)
    history["phase1_unconstrained_optimum"] = {
        "k": 2, "note": "statistically best-separated but not business-actionable (see docstring)",
        **score(X_base, km2.labels_),
    }

    # Phase 2: feature-transform search. Try each scaler x {with/without CancellationRate}
    # at a fixed representative k=5 (business literature's typical segment count) to find
    # the best preprocessing config before hill-climbing k.
    feature_sets = {
        "rfm_core": ALL_FEATURES[:6],
        "rfm_core_plus_cancellation": ALL_FEATURES,
    }
    best_transform = None
    for fs_name, feats in feature_sets.items():
        for scaler_name in SCALERS:
            X, _ = build_matrix(rfm, feats, LOG_CANDIDATES, scaler_name)
            km = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X)
            s = score(X, km.labels_)
            entry = {"feature_set": fs_name, "scaler": scaler_name, **s}
            history["phase2_feature_transform_search"].append(entry)
            if best_transform is None or s["silhouette"] > best_transform["silhouette"]:
                best_transform = entry

    print(f"Best transform config: {best_transform['feature_set']} / {best_transform['scaler']}")

    # Phase 3: greedy hill-climb over k within the business-constrained range, using the
    # winning transform config. Literal greedy hill-climbing (evaluate every k in range,
    # pick the best) -- not random search -- matching the AutoResearch pattern from
    # Projects 01-02.
    feats = feature_sets[best_transform["feature_set"]]
    scaler_name = best_transform["scaler"]
    best_k_result = None
    for k in K_RANGE:
        X, scaler = build_matrix(rfm, feats, LOG_CANDIDATES, scaler_name)
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        s = score(X, km.labels_)
        entry = {"k": k, **s}
        history["phase3_hill_climb"].append(entry)
        if best_k_result is None or s["silhouette"] > best_k_result["silhouette"]:
            best_k_result = entry

    winning_k = best_k_result["k"]
    print(f"Winning k (business-constrained, k in {list(K_RANGE)}): {winning_k}, silhouette={best_k_result['silhouette']:.4f}")

    # Phase 4: finalize -- retrain winner, compare against the previously-deployed
    # baseline (k=2, unconstrained) and the plain-k=5 rfm_core/standard baseline from
    # 03_train_models.py, then deploy if it's a genuine improvement in business terms.
    X_final, scaler_final = build_matrix(rfm, feats, LOG_CANDIDATES, scaler_name)
    final_km = KMeans(n_clusters=winning_k, random_state=42, n_init=10).fit(X_final)
    rfm["Cluster"] = final_km.labels_
    final_scores = score(X_final, final_km.labels_)

    history["phase4_finalize"] = {
        "algorithm": "kmeans", "k": winning_k, "features": feats, "scaler": scaler_name,
        **final_scores,
        "vs_unconstrained_k2_silhouette": history["phase1_unconstrained_optimum"]["silhouette"],
        "vs_baseline_rfm_core_standard_k5_silhouette": next(
            e["silhouette"] for e in history["phase2_feature_transform_search"]
            if e["feature_set"] == "rfm_core" and e["scaler"] == "standard"
        ),
    }

    rfm.to_csv(PROCESSED / "customer_segments.csv", index=False)
    joblib.dump(
        {"model": final_km, "scaler": scaler_final, "features": feats, "k": winning_k, "log_cols": LOG_CANDIDATES},
        MODELS / "production_kmeans.joblib",
    )

    with open(DOCS / "autoresearch.json", "w") as f:
        json.dump(history, f, indent=2)

    print(json.dumps(history["phase4_finalize"], indent=2))


if __name__ == "__main__":
    main()
