"""FastAPI backend for the Market Basket Pattern Mining admin dashboard.

CRISP-DM Phase 6: Deployment.
"""
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "eda"
PROCESSED = ROOT / "data" / "processed"

app = FastAPI(title="Market Basket Mining API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def load_json(base, name):
    path = base / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/eda")
def eda():
    return load_json(DOCS, "eda_findings.json")


@app.get("/api/prep-summary")
def prep_summary():
    return load_json(DOCS, "prep_summary.json")


@app.get("/api/modeling-baseline")
def modeling_baseline():
    return load_json(DOCS, "modeling_baseline.json")


@app.get("/api/autoresearch")
def autoresearch():
    return load_json(DOCS, "autoresearch.json")


@app.get("/api/rule-summary")
def rule_summary():
    return load_json(DOCS, "rule_summary.json")


@app.get("/api/rules")
def rules():
    return load_json(PROCESSED, "production_rules.json")


@app.get("/api/summary")
def summary():
    rs = load_json(DOCS, "rule_summary.json")
    prep = load_json(DOCS, "prep_summary.json")
    autores = load_json(DOCS, "autoresearch.json")
    return {
        "n_baskets": prep["n_multi_item_baskets"] if prep else None,
        "n_rules": rs["n_rules"] if rs else None,
        "avg_lift": rs["avg_lift"] if rs else None,
        "max_lift": rs["max_lift"] if rs else None,
        "winning_support": autores["winning_config"]["min_support"] if autores else None,
        "winning_confidence": autores["winning_config"]["min_confidence"] if autores else None,
    }
