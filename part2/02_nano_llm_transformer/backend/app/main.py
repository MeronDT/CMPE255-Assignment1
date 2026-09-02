import json
import platform
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .generate import get_service
from .schemas import ChatRequest, ChatResponse

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "eda"

app = FastAPI(title="NanoLlama Chat API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/system")
def system_info():
    """GPU/hardware info — the kind of detail an AI engineer checks before trusting a benchmark."""
    info = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        info.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_total_memory_mb": torch.cuda.get_device_properties(0).total_memory / 1e6,
                "gpu_allocated_memory_mb": torch.cuda.memory_allocated() / 1e6,
                "cuda_version": torch.version.cuda,
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    return info


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.messages or req.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Last message must be from the user")
    try:
        return get_service().chat(req.messages[-1].content, req.temperature, req.top_p, req.max_new_tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model-info")
def model_info():
    """Full CRISP-DM artifact bundle for the admin dashboard: EDA, data prep,
    base + SFT training histories, and the qualitative evaluation summary."""
    def read(name):
        path = DOCS / name
        return json.loads(path.read_text()) if path.exists() else None

    return {
        "eda": read("eda_findings.json"),
        "data_preparation": read("prep_summary.json"),
        "train_history_base": read("train_history_base.json"),
        "train_history_sft": read("train_history_sft.json"),
        "evaluation": read("evaluation_summary.json"),
    }


@app.get("/api/autoresearch")
def autoresearch():
    def read(name):
        path = DOCS / name
        return json.loads(path.read_text()) if path.exists() else None

    return {
        "history": read("autoresearch_history.json"),
        "finalize": read("autoresearch_finalize.json"),
    }
