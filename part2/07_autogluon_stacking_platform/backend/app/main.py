"""FastAPI backend for the AutoGluon Stacking Platform admin dashboard.

CRISP-DM Phase 6: Deployment.
"""
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "eda"

app = FastAPI(title="AutoGluon Stacking Platform API")
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


@app.get("/api/prep-summary")
def prep_summary():
    return load_json("prep_summary.json")


@app.get("/api/modeling-results")
def modeling_results():
    return load_json("modeling_results.json")


@app.get("/api/evaluation")
def evaluation():
    return load_json("evaluation.json")


@app.get("/api/summary")
def summary():
    ev = load_json("evaluation.json")
    return {
        "housing_improvement_pct": ev["housing"]["improvement_pct"] if ev else None,
        "adult_improvement_pct": ev["adult"]["improvement_pct"] if ev else None,
        "housing_n_models": ev["housing"]["n_stacked_models"] if ev else None,
        "adult_n_models": ev["adult"]["n_stacked_models"] if ev else None,
        "housing_selection_matched": ev["housing"]["selection_check"]["selection_matches_test_best"] if ev else None,
        "adult_selection_matched": ev["adult"]["selection_check"]["selection_matches_test_best"] if ev else None,
    }
