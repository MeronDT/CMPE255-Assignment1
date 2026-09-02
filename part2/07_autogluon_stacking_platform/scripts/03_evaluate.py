"""CRISP-DM Phase 5: Evaluation.

Summarizes both tasks' leaderboards, surfaces the AutoGluon-vs-baseline
comparison, and explicitly checks whether AutoGluon's *selected* best model
(by internal validation score) is actually the best model on the held-out
test set -- a genuine methodology check, not assumed.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "eda"


def check_selection_integrity(leaderboard, model_best):
    """AutoGluon selects `model_best` using internal VALIDATION performance
    (out-of-fold, never touching the test set). Does that selection also turn
    out to have the best score on the held-out TEST set the leaderboard above
    was scored against? NOTE: leaderboard(test_data=...) sorts by TEST score,
    not validation score -- so leaderboard[0] is already "best on test" by
    construction; the real question is whether model_best (the val-selected
    winner) matches leaderboard[0] (the test-set winner)."""
    best_on_test = leaderboard[0]  # leaderboard(test_data) is sorted by score_test
    selected_row = next((r for r in leaderboard if r["model"] == model_best), None)
    return {
        "selected_model_by_validation": model_best,
        "selected_model_test_score": selected_row["score_test"] if selected_row else None,
        "best_on_test_model": best_on_test["model"],
        "best_on_test_score": best_on_test["score_test"],
        "selection_matches_test_best": model_best == best_on_test["model"],
    }


def main():
    results = json.load(open(DOCS / "modeling_results.json"))

    housing_check = check_selection_integrity(
        results["housing"]["autogluon"]["leaderboard"], results["housing"]["autogluon"]["best_model"]
    )
    adult_check = check_selection_integrity(
        results["adult"]["autogluon"]["leaderboard"], results["adult"]["autogluon"]["best_model"]
    )

    summary = {
        "housing": {
            "task": "regression", "metric": "RMSE (lower better)",
            "baseline_rmse": results["housing"]["baseline"]["rmse"],
            "autogluon_rmse": results["housing"]["autogluon"]["rmse"],
            "improvement_pct": results["housing"]["improvement_pct"],
            "n_stacked_models": results["housing"]["autogluon"]["n_base_models"],
            "selection_check": housing_check,
        },
        "adult": {
            "task": "classification", "metric": "ROC-AUC (higher better)",
            "baseline_auc": results["adult"]["baseline"]["roc_auc"],
            "autogluon_auc": results["adult"]["autogluon"]["roc_auc"],
            "improvement_pct": results["adult"]["improvement_pct"],
            "n_stacked_models": results["adult"]["autogluon"]["n_base_models"],
            "selection_check": adult_check,
        },
    }

    with open(DOCS / "evaluation.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
