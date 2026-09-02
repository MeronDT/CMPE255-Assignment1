"""FastAPI backend for the CRISP-DM Master's Platform admin dashboard."""
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "eda"

app = FastAPI(title="CRISP-DM Master's Platform API")
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


@app.get("/api/clustering")
def clustering():
    return load_json("clustering_results.json")


@app.get("/api/anomaly")
def anomaly():
    return load_json("anomaly_results.json")


@app.get("/api/supervised")
def supervised():
    return load_json("supervised_results.json")


@app.get("/api/association")
def association():
    return load_json("association_results.json")


@app.get("/api/lsh")
def lsh():
    return load_json("lsh_results.json")


@app.get("/api/synthesis")
def synthesis():
    return load_json("synthesis.json")
