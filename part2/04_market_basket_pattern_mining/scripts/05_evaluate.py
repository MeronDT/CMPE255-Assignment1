"""CRISP-DM Phase 5: Evaluation.

Summarizes the production rule set for the dashboard: top rules by lift,
product-level "hub" analysis (which products appear in the most rules --
prime candidates for endcap/cross-sell placement), and a rule-network graph
structure for visualization.
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs" / "eda"


def main():
    rules = json.load(open(PROCESSED / "production_rules.json"))
    names = json.load(open(PROCESSED / "product_names.json"))

    product_rule_count = Counter()
    for r in rules:
        for p in r["antecedents"] + r["consequents"]:
            product_rule_count[p] += 1

    top_hub_products = [
        {"stock_code": p, "name": names.get(p, p), "n_rules": n}
        for p, n in product_rule_count.most_common(10)
    ]

    nodes = {}
    edges = []
    for r in rules:
        for p in r["antecedents"] + r["consequents"]:
            nodes[p] = names.get(p, p)
        for a in r["antecedents"]:
            for c in r["consequents"]:
                edges.append({"source": a, "target": c, "lift": round(r["lift"], 2), "confidence": round(r["confidence"], 2)})

    summary = {
        "n_rules": len(rules),
        "avg_lift": round(sum(r["lift"] for r in rules) / len(rules), 2),
        "max_lift": round(max(r["lift"] for r in rules), 2),
        "top_hub_products": top_hub_products,
        "graph": {
            "nodes": [{"id": k, "name": v} for k, v in nodes.items()],
            "edges": edges,
        },
        "top_rules_by_lift": sorted(
            [{"antecedents": [names.get(p, p) for p in r["antecedents"]],
              "consequents": [names.get(p, p) for p in r["consequents"]],
              "support": round(r["support"], 4), "confidence": round(r["confidence"], 3), "lift": round(r["lift"], 2)}
             for r in rules],
            key=lambda x: -x["lift"],
        )[:20],
    }

    with open(DOCS / "rule_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({k: v for k, v in summary.items() if k != "graph"}, indent=2))


if __name__ == "__main__":
    main()
