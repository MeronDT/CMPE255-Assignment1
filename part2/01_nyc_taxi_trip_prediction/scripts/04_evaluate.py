"""CRISP-DM Phase 5: Evaluation.

Turns model_search_log.json into the charts a data scientist actually wants:
hyperparameter-search convergence ("hill climbing"), predicted-vs-actual,
residual distribution, and feature importance, for both targets.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "eda"

TARGET_META = {
    "duration_min": {"label": "Trip Duration (min)", "color": "#6366f1"},
    "fare_amount": {"label": "Fare Amount ($)", "color": "#22c55e"},
}


def main():
    with open(DOCS / "model_search_log.json") as f:
        report = json.load(f)

    for target, meta in TARGET_META.items():
        result = report["targets"][target]
        trials = result["search_trials"]

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        fig.suptitle(f"Model Evaluation — {meta['label']}", fontsize=13, fontweight="bold")

        # 1. Hyperparameter search convergence ("hill climbing")
        rmses = [t["val_rmse"] for t in trials]
        best_so_far = np.minimum.accumulate(rmses)
        axes[0, 0].plot(range(len(trials)), rmses, "o--", color="#94a3b8", label="trial RMSE")
        axes[0, 0].plot(range(len(trials)), best_so_far, "-", color=meta["color"], linewidth=2, label="best so far")
        axes[0, 0].axhline(result["baseline_val_metrics"]["rmse"], color="#ef4444", linestyle=":", label="mean baseline")
        axes[0, 0].set_title("Hyperparameter Search (validation RMSE)")
        axes[0, 0].set_xlabel("trial")
        axes[0, 0].legend(fontsize=8)

        # 2. Predicted vs actual (holdout test set)
        y_true = np.array(result["residual_sample"]["y_true"])
        y_pred = np.array(result["residual_sample"]["y_pred"])
        lim = np.percentile(y_true, 99)
        axes[0, 1].scatter(y_true, y_pred, s=4, alpha=0.25, color=meta["color"])
        axes[0, 1].plot([0, lim], [0, lim], "--", color="#ef4444", linewidth=1)
        axes[0, 1].set_xlim(0, lim)
        axes[0, 1].set_ylim(0, lim)
        axes[0, 1].set_title("Predicted vs Actual (test set)")
        axes[0, 1].set_xlabel("actual")
        axes[0, 1].set_ylabel("predicted")

        # 3. Residual distribution
        residuals = y_pred - y_true
        axes[1, 0].hist(residuals, bins=60, color=meta["color"], edgecolor="none")
        axes[1, 0].axvline(0, color="#ef4444", linestyle="--")
        axes[1, 0].set_title(f"Residuals (mean={residuals.mean():.2f}, std={residuals.std():.2f})")

        # 4. Feature importance
        importances = result["feature_importance"][:8][::-1]
        axes[1, 1].barh([f["feature"] for f in importances], [f["importance"] for f in importances], color=meta["color"])
        axes[1, 1].set_title("Top Feature Importance (gain)")

        plt.tight_layout()
        out = DOCS / f"evaluation_{target}.png"
        plt.savefig(out, dpi=130)
        plt.close(fig)
        print(f"Wrote {out}")

        improvement_pct = (
            (result["baseline_val_metrics"]["rmse"] - result["best_val_rmse"])
            / result["baseline_val_metrics"]["rmse"]
            * 100
        )
        print(
            f"  {target}: baseline_rmse={result['baseline_val_metrics']['rmse']:.3f} -> "
            f"model_rmse={result['best_val_rmse']:.3f} ({improvement_pct:.1f}% improvement), "
            f"test_r2={result['test_metrics']['r2']:.3f}"
        )


if __name__ == "__main__":
    main()
