"""
FastAPI backend for the Pipe Leak Dashboard.

Exposes simulation, ML training, and prediction as REST endpoints
consumed by the React frontend.
"""

import sys
from pathlib import Path

_src = str(Path(__file__).resolve().parent.parent.parent)
if _src not in sys.path:
    sys.path.insert(0, _src)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd

from pipe_leak.api.state import AppState

app = FastAPI(title="Pipe Leak Dashboard API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state = AppState()


# ── Request / Response models ────────────────────────────────────────────────

class SimulationParams(BaseModel):
    num_pipes: int = 1000
    sim_years: int = 5
    seed: int = 42


class TrainParams(BaseModel):
    model_type: str = "xgboost"


class FilterParams(BaseModel):
    severity: list[str] | None = None
    material: list[str] | None = None
    date_from: str | None = None
    date_to: str | None = None


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    """Current state of the backend (has data, has model, etc.)."""
    return state.get_status()


@app.post("/api/simulate")
def run_simulation(params: SimulationParams):
    """Generate pipe network and leak events."""
    try:
        state.run_simulation(params.num_pipes, params.sim_years, params.seed)
        return state.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train")
def train_model(params: TrainParams = TrainParams()):
    """Train the ML model on current simulation data."""
    if not state.has_data:
        raise HTTPException(status_code=400, detail="No simulation data. Run /api/simulate first.")
    try:
        state.train(model_type=params.model_type)
        return state.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/overview")
def get_overview():
    """KPI metrics and summary data for the overview page."""
    if not state.has_data:
        raise HTTPException(status_code=400, detail="No data available.")
    return state.get_overview()


@app.get("/api/pipes")
def get_pipes():
    """Pipe network data with risk scores."""
    if not state.has_data:
        raise HTTPException(status_code=400, detail="No data available.")
    return state.get_pipes()


@app.post("/api/events")
def get_events(filters: FilterParams | None = None):
    """Leak events, optionally filtered."""
    if not state.has_data:
        raise HTTPException(status_code=400, detail="No data available.")
    return state.get_events(filters)


@app.get("/api/analysis")
def get_analysis():
    """Analysis data: trends, material breakdown, cost analysis."""
    if not state.has_data:
        raise HTTPException(status_code=400, detail="No data available.")
    return state.get_analysis()


@app.get("/api/model")
def get_model_performance():
    """Model metrics, curves, feature importance."""
    if not state.has_model:
        raise HTTPException(status_code=400, detail="No trained model.")
    return state.get_model_performance()


@app.get("/api/filters")
def get_filter_options():
    """Available filter values for the sidebar."""
    if not state.has_data:
        return {"severities": [], "materials": [], "date_range": None}
    return state.get_filter_options()
