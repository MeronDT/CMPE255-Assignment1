"""Technique 2: Anomaly / Outlier Detection.

Unlike Project 06's Credit Card Fraud data, this dataset has NO natural anomaly
labels -- there's no "this transaction is fraudulent" ground truth. Addressed
honestly rather than pretending an evaluation exists: a small set of synthetic,
deliberately extreme outlier customers is injected (10x-50x normal spend/quantity
combinations that don't occur naturally in this data) specifically to sanity-check
that Isolation Forest actually ranks genuine outliers above normal customers --
a necessary-but-not-sufficient validation, stated as such, not oversold as a real
precision/recall evaluation.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs" / "eda"

FEATURES = ["Recency", "Frequency", "Monetary", "AvgBasketValue", "DistinctProducts", "TenureDays", "CancellationRate"]
LOG_COLS = ["Monetary", "AvgBasketValue", "Frequency", "DistinctProducts"]


def build_matrix(df, scaler=None):
    X = df[FEATURES].copy()
    for c in LOG_COLS:
        X[c] = np.log1p(X[c].clip(lower=0))
    if scaler is None:
        scaler = StandardScaler().fit(X)
    return scaler.transform(X), scaler


def main():
    df = pd.read_csv(PROCESSED / "customer_features.csv")
    rng = np.random.RandomState(7)

    # Inject 15 synthetic extreme-outlier "customers" -- values far outside this
    # dataset's real range on multiple dimensions simultaneously (real outliers
    # are rarely extreme on just one axis).
    n_synth = 15
    synthetic = pd.DataFrame({
        "CustomerID": [-1000 - i for i in range(n_synth)],
        "Recency": rng.uniform(0, 5, n_synth),
        "Frequency": rng.uniform(200, 500, n_synth),
        "Monetary": rng.uniform(200000, 800000, n_synth),
        "AvgBasketValue": rng.uniform(5000, 20000, n_synth),
        "DistinctProducts": rng.uniform(500, 1000, n_synth),
        "TenureDays": rng.uniform(300, 370, n_synth),
        "CancellationRate": rng.uniform(0, 0.05, n_synth),
    })
    synthetic["IsSyntheticOutlier"] = 1
    df["IsSyntheticOutlier"] = 0
    combined = pd.concat([df, synthetic], ignore_index=True)

    X, _ = build_matrix(combined)
    iso = IsolationForest(n_estimators=300, contamination=0.02, random_state=42, n_jobs=-1)
    iso.fit(X)
    scores = -iso.score_samples(X)
    combined["AnomalyScore"] = scores

    # Validation: do the 15 synthetic outliers rank near the top?
    combined_sorted = combined.sort_values("AnomalyScore", ascending=False).reset_index(drop=True)
    synth_ranks = combined_sorted.index[combined_sorted["IsSyntheticOutlier"] == 1].tolist()
    n_total = len(combined)
    mean_percentile = float(np.mean([100 * (1 - r / n_total) for r in synth_ranks]))
    all_in_top_5pct = all(r < n_total * 0.05 for r in synth_ranks)

    real_customers = combined[combined["IsSyntheticOutlier"] == 0].copy()
    real_customers.to_csv(PROCESSED / "anomaly_scores.csv", index=False)

    top_real_anomalies = real_customers.sort_values("AnomalyScore", ascending=False).head(15)
    top_anomalies_export = top_real_anomalies[["CustomerID", "AnomalyScore"] + FEATURES].to_dict(orient="records")

    result = {
        "n_customers": len(real_customers),
        "n_synthetic_outliers_injected": n_synth,
        "synthetic_outlier_mean_percentile_rank": round(mean_percentile, 1),
        "all_synthetic_outliers_in_top_5pct": all_in_top_5pct,
        "validation_note": (
            "This dataset has no natural anomaly labels, unlike Project 06's Credit "
            "Card Fraud data. This validation confirms the detector correctly ranks "
            "deliberately extreme synthetic outliers above real customers -- a sanity "
            "check that the model works as intended, NOT a precision/recall evaluation "
            "against real fraud (no such ground truth exists in this dataset)."
        ),
        "top_real_anomalies": top_anomalies_export,
    }
    with open(DOCS / "anomaly_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in result.items() if k != "top_real_anomalies"}, indent=2))


if __name__ == "__main__":
    main()
