"""Technique 1: Unsupervised Learning (Clustering).

Same business-constrained methodology validated in Project 03: pure silhouette
maximization tends to pick an uninformative low-k split, so this applies the
k in [3,8] business constraint from the start rather than re-discovering it.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs" / "eda"

FEATURES = ["Recency", "Frequency", "Monetary", "AvgBasketValue", "DistinctProducts", "TenureDays", "CancellationRate"]
LOG_COLS = ["Monetary", "AvgBasketValue", "Frequency", "DistinctProducts"]
K_RANGE = range(3, 9)


def main():
    df = pd.read_csv(PROCESSED / "customer_features.csv")
    X = df[FEATURES].copy()
    for c in LOG_COLS:
        X[c] = np.log1p(X[c])
    Xs = MinMaxScaler().fit_transform(X)

    sweep = []
    best = None
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
        s = silhouette_score(Xs, km.labels_)
        sweep.append({"k": k, "silhouette": round(float(s), 4)})
        if best is None or s > best["silhouette"]:
            best = {"k": k, "silhouette": round(float(s), 4), "model": km}

    df["Cluster"] = best["model"].labels_
    profiles = []
    for cid, grp in df.groupby("Cluster"):
        profiles.append({
            "cluster": int(cid), "n_customers": len(grp),
            "pct_of_customers": round(len(grp) / len(df) * 100, 1),
            "avg_recency": round(float(grp["Recency"].mean()), 1),
            "avg_frequency": round(float(grp["Frequency"].mean()), 1),
            "avg_monetary": round(float(grp["Monetary"].mean()), 2),
            "pct_of_revenue": round(float(grp["Monetary"].sum() / df["Monetary"].sum() * 100), 1),
        })
    profiles.sort(key=lambda p: -p["avg_monetary"])

    df[["CustomerID", "Cluster"]].to_csv(PROCESSED / "clustering_assignments.csv", index=False)
    result = {"sweep": sweep, "winning_k": best["k"], "winning_silhouette": best["silhouette"], "profiles": profiles}
    with open(DOCS / "clustering_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
