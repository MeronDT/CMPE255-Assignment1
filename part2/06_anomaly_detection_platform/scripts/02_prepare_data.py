"""CRISP-DM Phase 3: Data Preparation.

Cleaning decisions (justified by 01_eda.py findings):
- Drop 9,144 duplicate rows: with 28 continuous PCA-transformed features, two
  transactions matching on every single value to full float precision by pure
  chance is vanishingly improbable -- these are near-certainly data-collection
  artifacts (e.g. re-logged transactions), not genuine repeat purchases. Keeping
  them would let the same fraud pattern leak into both a "train" view and an
  "eval" view of an unsupervised model, inflating apparent performance.
- StandardScaler on all features (V1-V28 are already roughly standardized by the
  PCA step, but Amount is heavily right-skewed and on a totally different scale
  -- log1p + standardize it so no single feature dominates the distance-based
  methods, matching standard practice for this exact dataset in the literature).
- The Class label is split off and saved separately -- used only for evaluation
  (05_evaluate.py), never fed to any model as a training feature.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs" / "eda"
PROCESSED.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RAW / "creditcard.csv")
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)

    y = df["Class"].values
    X = df.drop(columns=["Class"]).copy()
    X["Amount"] = np.log1p(X["Amount"])

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_scaled.to_parquet(PROCESSED / "X_features.parquet")
    np.save(PROCESSED / "y_labels.npy", y)

    summary = {
        "n_before_dedup": before,
        "n_after_dedup": after,
        "n_duplicates_dropped": before - after,
        "n_fraud_after_dedup": int(y.sum()),
        "fraud_rate_after_dedup_pct": round(float(y.mean()) * 100, 4),
        "n_features": X_scaled.shape[1],
    }
    with open(DOCS / "prep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
