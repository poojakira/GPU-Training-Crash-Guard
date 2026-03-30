"""
gpudefrag.api — FastAPI surface for querying simulated OOM risk.

Exposes a lightweight REST API for real-time GPU memory telemetry
and OOM-risk prediction.

Usage::

    uvicorn gpudefrag.api:app --host 0.0.0.0 --port 8000
    # GET  /health
    # GET  /memory
    # POST /risk   {"fragmentation": 0.7, "utilisation": 0.9, "alloc_delta_mb": 15}
    # GET  /risk/history
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

app = FastAPI(
    title="gpudefrag API",
    description="Query simulated OOM risk and GPU memory telemetry",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Shared state (lightweight — single-process research tool)
# ---------------------------------------------------------------------------

from src.predictor.risk_model import OOMRiskModel
from src.monitor.allocator_logger import AllocatorLogger

_risk_model = OOMRiskModel(mode="rule")
_logger = AllocatorLogger()

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class RiskRequest(BaseModel):
    fragmentation: float = Field(0.0, ge=0.0, le=1.0, description="Fragmentation ratio")
    utilisation: float = Field(0.0, ge=0.0, le=1.0, description="GPU memory utilisation")
    alloc_delta_mb: float = Field(0.0, description="Recent allocation delta in MB")

class RiskResponse(BaseModel):
    risk_score: float
    tier: str
    message: str

class MemoryResponse(BaseModel):
    allocated_mb: float
    reserved_mb: float
    free_estimate_mb: float
    fragmentation_ratio: float
    cuda_available: bool

# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def _gpu_snapshot() -> Dict[str, Any]:
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            resv = torch.cuda.memory_reserved() / (1024 ** 2)
            return {
                "allocated_mb": round(alloc, 2),
                "reserved_mb": round(resv, 2),
                "free_estimate_mb": round(resv - alloc, 2),
                "fragmentation_ratio": round(1 - alloc / resv, 4) if resv > 0 else 0.0,
                "cuda_available": True,
            }
    except ImportError:
        pass
    return {
        "allocated_mb": 0.0,
        "reserved_mb": 0.0,
        "free_estimate_mb": 0.0,
        "fragmentation_ratio": 0.0,
        "cuda_available": False,
    }

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok", "service": "gpudefrag-api"}


@app.get("/memory", response_model=MemoryResponse)
def get_memory():
    """Return current GPU memory state."""
    return _gpu_snapshot()


@app.post("/risk", response_model=RiskResponse)
def compute_risk(req: RiskRequest):
    """
    Compute OOM-risk score from memory statistics.

    Returns a score in [0, 1] and an action tier (SAFE / WARN / ACT).
    """
    score = _risk_model.score(
        fragmentation=req.fragmentation,
        utilisation=req.utilisation,
        alloc_delta_mb=req.alloc_delta_mb,
    )

    if score >= 0.8:
        tier, msg = "ACT", "High OOM risk — consider clearing cache or reducing batch size"
    elif score >= 0.5:
        tier, msg = "WARN", "Elevated OOM risk — monitor closely"
    else:
        tier, msg = "SAFE", "OOM risk is low"

    return RiskResponse(risk_score=score, tier=tier, message=msg)


@app.get("/risk/history")
def risk_history():
    """Return all past risk evaluations."""
    return {"count": len(_risk_model.history), "entries": _risk_model.history}
