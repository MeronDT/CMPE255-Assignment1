"""FastAPI backend for the Customer Segmentation admin dashboard.

CRISP-DM Phase 6: Deployment. Serves the AutoResearch-winning segmentation
(production_kmeans.joblib) plus every artifact a data scientist would want
to audit: EDA findings, baseline model comparison, the AutoResearch search
history, and per-cluster business profiles.
"""
import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "eda"
PROCESSED = ROOT / "data" / "processed"

app = FastAPI(title="Customer Segmentation API")
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


@app.get("/api/cluster-profiles")
def cluster_profiles():
    return load_json("cluster_profiles.json")


@app.get("/api/customers")
def customers(limit: int = 4500):
    df = pd.read_csv(PROCESSED / "customer_segments.csv")
    cols = ["CustomerID", "Recency", "Frequency", "Monetary", "AvgBasketValue",
            "DistinctProducts", "TenureDays", "CancellationRate", "PrimaryCountry", "Cluster"]
    return df[cols].head(limit).to_dict(orient="records")


@app.get("/api/summary")
def summary():
    prep = load_json("prep_summary.json")
    profiles = load_json("cluster_profiles.json")
    autores = load_json("autoresearch.json")
    return {
        "n_customers": prep["n_customers"] if prep else None,
        "n_clusters": profiles["n_clusters"] if profiles else None,
        "revenue_concentration_check": profiles["revenue_concentration_check"] if profiles else None,
        "winning_silhouette": autores["phase4_finalize"]["silhouette"] if autores else None,
    }
