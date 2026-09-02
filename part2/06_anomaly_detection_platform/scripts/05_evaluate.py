"""CRISP-DM Phase 5: Evaluation.

Precision/recall at multiple operating thresholds, confusion-matrix-style
breakdown, and top-N flagged transactions for the dashboard.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs" / "eda"


def main():
    y = np.load(PROCESSED / "y_labels.npy")
    scores = np.load(PROCESSED / "full_anomaly_scores.npy")

    precision, recall, thresholds = precision_recall_curve(y, scores)
    # Downsample the PR curve for dashboard rendering (275K points is too many).
    step = max(1, len(precision) // 300)
    pr_curve = [
        {"precision": round(float(p), 4), "recall": round(float(r), 4)}
        for p, r in zip(precision[::step], recall[::step])
    ]

    # Operating point: flag the top-N most anomalous transactions, matching
    # the true contamination rate the AutoResearch phase converged on.
    n_flag = int(y.sum())  # flag as many as there are true frauds, a natural operating point
    top_idx = np.argsort(-scores)[:n_flag]
    flagged_is_fraud = y[top_idx]
    precision_at_n = float(flagged_is_fraud.mean())
    recall_at_n = float(flagged_is_fraud.sum() / y.sum())

    summary = {
        "n_transactions": len(y),
        "n_fraud": int(y.sum()),
        "operating_point": {
            "n_flagged": n_flag,
            "precision": round(precision_at_n, 4),
            "recall": round(recall_at_n, 4),
            "note": f"Flagging the top {n_flag} most-anomalous transactions (matching the true fraud count as a natural operating point) catches {int(flagged_is_fraud.sum())}/{int(y.sum())} actual frauds.",
        },
        "pr_curve": pr_curve,
    }

    with open(DOCS / "evaluation.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({k: v for k, v in summary.items() if k != "pr_curve"}, indent=2))


if __name__ == "__main__":
    main()
