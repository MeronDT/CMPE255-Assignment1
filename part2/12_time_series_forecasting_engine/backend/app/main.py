"""FastAPI backend for the Time Series Forecasting Engine admin dashboard."""
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "eda"

app = FastAPI(title="Time Series Forecasting Engine API")
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
    return load_json("eda_and_prep_summary.json")


@app.get("/api/modeling-baseline")
def modeling_baseline():
    return load_json("modeling_baseline.json")


@app.get("/api/autoresearch")
def autoresearch():
    return load_json("autoresearch.json")


@app.get("/api/evaluation")
def evaluation():
    return load_json("evaluation.json")


@app.get("/api/summary")
def summary():
    autores = load_json("autoresearch.json")
    ev = load_json("evaluation.json")
    return {
        "overall_winner": autores["overall_winner"] if autores else None,
        "winning_mape": autores["all_candidates_mape"].get(autores["overall_winner"]) if autores else None,
        "all_candidates_mape": autores["all_candidates_mape"] if autores else None,
        "n_forward_forecast_days": len(ev["forward_forecast"]["dates"]) if ev else None,
    }
