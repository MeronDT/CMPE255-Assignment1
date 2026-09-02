"""CRISP-DM Phase 4: Modeling (baseline).

Compares K-Means, Agglomerative (Ward), Gaussian Mixture, and DBSCAN on
log-transformed, standardized RFM features. K for the partition-based methods
is chosen via silhouette score over k=2..10 (elbow/inertia reported alongside
for cross-reference, per standard clustering-evaluation practice since there
is no ground-truth label to optimize against directly).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs" / "eda"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

FEATURES = ["Recency", "Frequency", "Monetary", "AvgBasketValue", "DistinctProducts", "TenureDays"]


def build_feature_matrix(rfm):
    X = rfm[FEATURES].copy()
    # log1p on right-skewed monetary/count features — standard RFM preprocessing
    # (Monetary and Frequency both span multiple orders of magnitude; log transform
    # prevents whale customers from dominating the Euclidean distance K-Means uses).
    for col in ["Monetary", "AvgBasketValue", "Frequency", "DistinctProducts"]:
        X[col] = np.log1p(X[col])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler


def evaluate(X, labels):
    if len(set(labels)) < 2 or len(set(labels)) >= len(X):
        return None
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
    }


def main():
    rfm = pd.read_csv(PROCESSED / "customer_rfm.csv")
    X, scaler = build_feature_matrix(rfm)

    results = {"k_sweep": [], "algorithm_comparison": {}}

    # K sweep for KMeans: silhouette + inertia (elbow) over k=2..10
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        metrics = evaluate(X, labels)
        results["k_sweep"].append({"k": k, "inertia": float(km.inertia_), **(metrics or {})})

    best_k = max(results["k_sweep"], key=lambda r: r["silhouette"])["k"]
    results["best_k_by_silhouette"] = best_k

    # Algorithm comparison at best_k
    algos = {
        "kmeans": KMeans(n_clusters=best_k, random_state=42, n_init=10),
        "agglomerative_ward": AgglomerativeClustering(n_clusters=best_k, linkage="ward"),
        "gaussian_mixture": GaussianMixture(n_components=best_k, random_state=42),
    }
    for name, model in algos.items():
        labels = model.fit_predict(X)
        results["algorithm_comparison"][name] = evaluate(X, labels)

    # DBSCAN: no k parameter, sweep eps instead (min_samples fixed at a reasonable default)
    dbscan_sweep = []
    for eps in [0.3, 0.5, 0.7, 1.0, 1.5]:
        db = DBSCAN(eps=eps, min_samples=10)
        labels = db.fit_predict(X)
        n_noise = int((labels == -1).sum())
        metrics = evaluate(X, labels)
        dbscan_sweep.append({"eps": eps, "n_noise": n_noise, **(metrics or {"n_clusters": 0})})
    results["dbscan_eps_sweep"] = dbscan_sweep

    # Deploy K-Means at best_k as the baseline production model (best silhouette among
    # partition methods, and unlike GMM/DBSCAN produces convex, marketer-interpretable
    # segments — a reasonable default motivated by the RFM segmentation literature,
    # e.g. Christy et al. 2021's comparison of clustering algorithms for RFM analysis).
    final_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = final_km.fit_predict(X)
    rfm["Cluster"] = labels

    rfm.to_csv(PROCESSED / "customer_segments.csv", index=False)
    joblib.dump({"model": final_km, "scaler": scaler, "features": FEATURES, "k": best_k}, MODELS / "baseline_kmeans.joblib")

    results["deployed"] = {"algorithm": "kmeans", "k": best_k, **evaluate(X, labels)}
    with open(DOCS / "modeling_baseline.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
