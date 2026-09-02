"""CRISP-DM Phase 2: Data Understanding / EDA."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DOCS = ROOT / "docs" / "eda"
DOCS.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RAW / "creditcard.csv")
    n_fraud = int((df["Class"] == 1).sum())
    n_normal = int((df["Class"] == 0).sum())

    findings = {
        "n_transactions": len(df),
        "n_features": df.shape[1] - 1,  # excluding Class
        "n_fraud": n_fraud,
        "n_normal": n_normal,
        "fraud_rate_pct": round(n_fraud / len(df) * 100, 4),
        "amount_stats_normal": df[df["Class"] == 0]["Amount"].describe().to_dict(),
        "amount_stats_fraud": df[df["Class"] == 1]["Amount"].describe().to_dict(),
        "n_missing": int(df.isnull().sum().sum()),
        "n_duplicate_rows": int(df.duplicated().sum()),
    }

    with open(DOCS / "eda_findings.json", "w") as f:
        json.dump(findings, f, indent=2, default=str)

    print(json.dumps(findings, indent=2, default=str))


if __name__ == "__main__":
    main()
