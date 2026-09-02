import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .predict import get_service
from .schemas import PredictRequest, PredictResponse, ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "eda"

app = FastAPI(title="NYC Taxi Trip Estimator API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/zones", response_model=list[ZoneInfo])
def zones():
    return get_service().list_zones()


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        return get_service().predict(
            req.pickup_location_id, req.dropoff_location_id, req.pickup_datetime, req.passenger_count
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/model-info")
def model_info():
    """Everything the admin dashboard needs: CRISP-DM artifacts, search history,
    feature importance, evaluation metrics — for both targets."""
    with open(DOCS / "model_search_log.json") as f:
        search_log = json.load(f)
    with open(DOCS / "eda_findings.json") as f:
        eda = json.load(f)
    with open(DOCS / "prep_summary.json") as f:
        prep = json.load(f)
    return {"eda": eda, "data_preparation": prep, "model_search": search_log}


@app.get("/api/autoresearch")
def autoresearch():
    """4-phase AutoResearch telemetry: multi-backbone tournament, feature-transform
    search, hyperparameter hill-climbing path, and blending — plus the full-data
    finalization outcome and literature benchmark comparison."""
    with open(DOCS / "autoresearch_history.json") as f:
        history = json.load(f)
    with open(DOCS / "autoresearch_finalize.json") as f:
        finalize = json.load(f)
    return {"history": history, "finalize": finalize}
