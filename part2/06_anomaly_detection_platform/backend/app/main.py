"""FastAPI backend for the Anomaly Detection admin dashboard.

CRISP-DM Phase 6: Deployment.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "eda"
PROCESSED = ROOT / "data" / "processed"

app = FastAPI(title="Anomaly Detection API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def load_json(name):
    path = DOCS / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/eda")
def eda():
    return load_json("eda_findings.json")


@app.get("/api/prep-summary")
def prep_summary():
    return load_json("prep_summary.json")


@app.get("/api/modeling-baseline")
def modeling_baseline():
    return load_json("modeling_baseline.json")


@app.get("/api/autoresearch")
def autoresearch():
    return load_json("autoresearch.json")


@app.get("/api/evaluation")
def evaluation():
    return load_json("evaluation.json")


@app.get("/api/top-anomalies")
def top_anomalies(limit: int = 50):
    scores = np.load(PROCESSED / "full_anomaly_scores.npy")
    y = np.load(PROCESSED / "y_labels.npy")
    X = pd.read_parquet(PROCESSED / "X_features.parquet")
    top_idx = np.argsort(-scores)[:limit]
    rows = []
    for i in top_idx:
        rows.append({
            "index": int(i),
            "anomaly_score": round(float(scores[i]), 4),
            "is_actual_fraud": bool(y[i]),
            "amount_scaled": round(float(X.iloc[i]["Amount"]), 3),
        })
    return rows


@app.get("/api/summary")
def summary():
    prep = load_json("prep_summary.json")
    autores = load_json("autoresearch.json")
    evald = load_json("evaluation.json")
    finalize = autores["phase2_finalize"] if autores else None
    return {
        "n_transactions": prep["n_after_dedup"] if prep else None,
        "n_fraud": prep["n_fraud_after_dedup"] if prep else None,
        "fraud_rate_pct": prep["fraud_rate_after_dedup_pct"] if prep else None,
        "full_auprc": finalize["auprc"] if finalize else None,
        "full_roc_auc": finalize["roc_auc"] if finalize else None,
        "lift_over_random": finalize["full_auprc_lift_over_random"] if finalize else None,
        "operating_point": evald["operating_point"] if evald else None,
    }
