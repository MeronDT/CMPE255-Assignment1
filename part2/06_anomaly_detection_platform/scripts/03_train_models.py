"""CRISP-DM Phase 4: Modeling (baseline).

Algorithm tournament: Isolation Forest, Local Outlier Factor, One-Class SVM,
Elliptic Envelope -- the four standard unsupervised anomaly detectors.

Run on a stratified proxy subsample (20K normal + all fraud), not the full
275K-row set: One-Class SVM and LOF are both super-linear in sample count
(LOF is O(n^2) for its k-NN step, SVM's QP solver similarly), so a full-data
tournament for 4 algorithms would take far longer than the time budget
justifies for a *comparison* pass. The proxy-then-finalize pattern matches
Projects 01/02's AutoResearch approach -- winner gets verified at full scale
in 04_autoresearch.py's finalization step, not just assumed from the proxy.

Class labels are used ONLY to compute AUPRC after each model scores every
point -- never as a training input to any of the four models.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs" / "eda"

RNG = np.random.RandomState(42)
PROXY_N_NORMAL = 20000


def make_proxy(X, y):
    fraud_idx = np.where(y == 1)[0]
    normal_idx = np.where(y == 0)[0]
    sampled_normal = RNG.choice(normal_idx, size=PROXY_N_NORMAL, replace=False)
    idx = np.concatenate([fraud_idx, sampled_normal])
    RNG.shuffle(idx)
    return X.iloc[idx].reset_index(drop=True), y[idx]


def evaluate(y_true, scores):
    # scores: higher = more anomalous, for all four methods below (sign-flipped
    # where the sklearn convention is the opposite, e.g. decision_function).
    return {
        "auprc": round(float(average_precision_score(y_true, scores)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 4),
    }


def main():
    X = pd.read_parquet(PROCESSED / "X_features.parquet")
    y = np.load(PROCESSED / "y_labels.npy")
    Xp, yp = make_proxy(X, y)

    results = {}

    t0 = time.time()
    iso = IsolationForest(n_estimators=200, contamination=0.0018, random_state=42, n_jobs=-1)
    iso.fit(Xp)
    scores = -iso.score_samples(Xp)  # higher = more anomalous
    results["isolation_forest"] = {**evaluate(yp, scores), "seconds": round(time.time() - t0, 2)}

    t0 = time.time()
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.0018, novelty=False)
    lof.fit_predict(Xp)
    scores = -lof.negative_outlier_factor_
    results["local_outlier_factor"] = {**evaluate(yp, scores), "seconds": round(time.time() - t0, 2)}

    t0 = time.time()
    ocsvm = OneClassSVM(kernel="rbf", nu=0.0018, gamma="scale")
    ocsvm.fit(Xp)
    scores = -ocsvm.decision_function(Xp)
    results["one_class_svm"] = {**evaluate(yp, scores), "seconds": round(time.time() - t0, 2)}

    t0 = time.time()
    ee = EllipticEnvelope(contamination=0.0018, random_state=42)
    ee.fit(Xp)
    scores = -ee.decision_function(Xp)
    results["elliptic_envelope"] = {**evaluate(yp, scores), "seconds": round(time.time() - t0, 2)}

    winner = max(results, key=lambda k: results[k]["auprc"])
    results["_proxy_n_normal"] = PROXY_N_NORMAL
    results["_proxy_n_fraud"] = int(yp.sum())
    results["_winner_by_auprc"] = winner

    with open(DOCS / "modeling_baseline.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nWinner (proxy scale): {winner}")


if __name__ == "__main__":
    main()
