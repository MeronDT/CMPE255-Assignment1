"""Technique 4: Associative Rule Mining.

Same business-constrained selection methodology validated in Project 04: search
directly within a merchandiser-reviewable rule-count band rather than maximizing
raw rule count.
"""
import json
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs" / "eda"

SUPPORT_GRID = [0.02, 0.03, 0.05, 0.08]
CONFIDENCE_GRID = [0.3, 0.4, 0.5, 0.6]
MIN_RULES, MAX_RULES = 5, 50


def main():
    basket_matrix = pd.read_pickle(PROCESSED / "basket_matrix.pkl")
    product_names = json.load(open(PROCESSED / "product_names.json"))

    candidates = []
    for support in SUPPORT_GRID:
        freq = fpgrowth(basket_matrix, min_support=support, use_colnames=True)
        if len(freq) == 0:
            continue
        for confidence in CONFIDENCE_GRID:
            rules = association_rules(freq, metric="confidence", min_threshold=confidence)
            rules = rules[rules["lift"] > 1]
            n = len(rules)
            avg_lift = float(rules["lift"].mean()) if n else 0.0
            candidates.append({"support": support, "confidence": confidence, "n_rules": n, "avg_lift": round(avg_lift, 3), "rules": rules})

    in_range = [c for c in candidates if MIN_RULES <= c["n_rules"] <= MAX_RULES]
    winner = max(in_range, key=lambda c: c["avg_lift"]) if in_range else min(
        [c for c in candidates if c["n_rules"] > MAX_RULES], key=lambda c: c["n_rules"]
    )

    final_rules = winner["rules"].sort_values("lift", ascending=False).copy()
    final_rules["antecedents"] = final_rules["antecedents"].apply(lambda s: [product_names.get(p, p) for p in s])
    final_rules["consequents"] = final_rules["consequents"].apply(lambda s: [product_names.get(p, p) for p in s])
    top_rules = final_rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(20).to_dict(orient="records")

    result = {
        "grid": [{"support": c["support"], "confidence": c["confidence"], "n_rules": c["n_rules"], "avg_lift": c["avg_lift"]} for c in candidates],
        "winning_support": winner["support"],
        "winning_confidence": winner["confidence"],
        "n_rules": winner["n_rules"],
        "avg_lift": winner["avg_lift"],
        "top_rules": top_rules,
    }
    with open(DOCS / "association_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in result.items() if k not in ("grid", "top_rules")}, indent=2))


if __name__ == "__main__":
    main()
