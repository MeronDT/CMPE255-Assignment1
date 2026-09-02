"""AutoResearch: hill-climbing search over min_support / min_confidence.

Follows the pattern from Projects 01-03: an algorithm tournament (already run
in 03_train_models.py), then a hyperparameter search, then finalization.

Association-rule mining doesn't have a single scalar "accuracy" to optimize --
per the founding literature (Agrawal & Srikant 1994), the real objective is
finding a support/confidence operating point that yields a merchandiser-
reviewable set of HIGH-LIFT rules. A first pass scoring purely by raw
high-lift rule count picked the loosest thresholds and returned 7,799 rules
-- unusable in practice, the mining equivalent of Project 3's unconstrained
k=2 finding failing the business objective. Fixed the same way Project 3
was: apply an explicit business constraint (a merchandising team can review
roughly 10-50 rules in a sitting) BEFORE ranking, mirroring the k-range
constraint pattern.
"""
import json
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs" / "eda"

SUPPORT_GRID = [0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.1]
CONFIDENCE_GRID = [0.2, 0.3, 0.4, 0.5, 0.6]
# Business constraint (from 01_business_understanding.md): a merchandising
# team can meaningfully review a small rule set, not thousands.
MAX_RULES = 50
MIN_RULES = 5


def score_config(basket_matrix, support, confidence):
    freq = fpgrowth(basket_matrix, min_support=support, use_colnames=True)
    if len(freq) == 0:
        return None, None
    rules = association_rules(freq, metric="confidence", min_threshold=confidence)
    rules = rules[rules["lift"] > 1]
    n_high_lift = int((rules["lift"] > 2).sum())
    n_total = len(rules)
    avg_lift = float(rules["lift"].mean()) if n_total else 0.0
    return rules, {"n_rules": n_total, "n_high_lift_rules": n_high_lift, "avg_lift": round(avg_lift, 3)}


def main():
    basket_matrix = pd.read_pickle(PROCESSED / "basket_matrix.pkl")
    history = {"phase1_grid_search": [], "phase2_finalize": None}

    candidates = []
    for support in SUPPORT_GRID:
        for confidence in CONFIDENCE_GRID:
            rules, metrics = score_config(basket_matrix, support, confidence)
            if metrics is None:
                continue
            entry = {"min_support": support, "min_confidence": confidence, **metrics}
            history["phase1_grid_search"].append(entry)
            candidates.append((entry, rules))

    # Business-constrained selection: among configs whose rule COUNT falls in
    # [MIN_RULES, MAX_RULES], pick the one with the highest average lift (the
    # most genuinely surprising/useful rules, not just the most numerous).
    in_range = [(e, r) for e, r in candidates if MIN_RULES <= e["n_rules"] <= MAX_RULES]
    if in_range:
        best, best_rules = max(in_range, key=lambda pair: pair[0]["avg_lift"])
        history["constraint_satisfied"] = True
    else:
        # Honest fallback: no grid point landed in the target band -- report
        # this rather than silently picking something else. Take whichever
        # config has the fewest rules ABOVE MAX_RULES (closest miss).
        above = [(e, r) for e, r in candidates if e["n_rules"] > MAX_RULES]
        best, best_rules = min(above, key=lambda pair: pair[0]["n_rules"])
        history["constraint_satisfied"] = False

    history["winning_config"] = {"min_support": best["min_support"], "min_confidence": best["min_confidence"]}

    final_rules = best_rules.sort_values("lift", ascending=False).copy()
    final_rules["antecedents"] = final_rules["antecedents"].apply(lambda s: list(s))
    final_rules["consequents"] = final_rules["consequents"].apply(lambda s: list(s))
    final_export = final_rules[["antecedents", "consequents", "support", "confidence", "lift"]]
    final_export.to_json(PROCESSED / "production_rules.json", orient="records", indent=2)

    history["phase2_finalize"] = {
        **best,
        "n_rules_final": len(final_rules),
    }

    with open(DOCS / "autoresearch.json", "w") as f:
        json.dump(history, f, indent=2)

    print(json.dumps(history["phase2_finalize"], indent=2))
    print(f"Winning config: support={best['min_support']}, confidence={best['min_confidence']}")


if __name__ == "__main__":
    main()
