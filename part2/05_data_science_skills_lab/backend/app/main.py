"""FastAPI backend for the Data Science Skills Mastery Lab admin dashboard."""
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "eda"

app = FastAPI(title="Data Science Skills Mastery Lab API")
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


@app.get("/api/catalog")
def catalog():
    return load_json("skill_catalog.json")


@app.get("/api/execution-results")
def execution_results():
    return load_json("skill_execution_results.json")


@app.get("/api/execution-results/{skill_name}")
def execution_result(skill_name: str):
    results = load_json("skill_execution_results.json")
    if results is None or skill_name not in results:
        return {"error": "not found"}
    return results[skill_name]
