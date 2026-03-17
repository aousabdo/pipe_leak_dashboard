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
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
import io
from datetime import datetime

from pipe_leak.api.state import AppState
from pipe_leak.api.reports import build_overview_report, build_analysis_report, build_model_report

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


# ── Download Endpoints ──────────────────────────────────────────────────────

@app.get("/api/download/pipes")
def download_pipes_csv():
    """Download pipe network data as CSV."""
    if not state.has_data:
        raise HTTPException(status_code=400, detail="No data available.")
    pipes = state.pipes_gdf.copy()
    if state.risk_scores is not None:
        pipes["risk_score"] = state.risk_scores
    else:
        pipes["risk_score"] = 0.0
    # Drop geometry column for CSV export
    cols = [c for c in pipes.columns if c != "geometry"]
    buf = io.StringIO()
    pipes[cols].to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=pipe_network_data.csv"},
    )


@app.get("/api/download/events")
def download_events_csv():
    """Download leak events data as CSV."""
    if not state.has_data:
        raise HTTPException(status_code=400, detail="No data available.")
    buf = io.StringIO()
    state.events_df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=leak_events_data.csv"},
    )


@app.get("/api/download/report")
def download_overview_report():
    """Download overview report as HTML with KPIs, charts, tables."""
    if not state.has_data:
        raise HTTPException(status_code=400, detail="No data available.")
    html = build_overview_report(state)
    return StreamingResponse(
        iter([html.encode("utf-8")]),
        media_type="text/html",
        headers={"Content-Disposition": "attachment; filename=overview_report.html"},
    )


@app.get("/api/download/report/analysis")
def download_analysis_report():
    """Download analysis report with trends, material risk, cost breakdowns."""
    if not state.has_data:
        raise HTTPException(status_code=400, detail="No data available.")
    html = build_analysis_report(state)
    return StreamingResponse(
        iter([html.encode("utf-8")]),
        media_type="text/html",
        headers={"Content-Disposition": "attachment; filename=analysis_report.html"},
    )


@app.get("/api/download/report/model")
def download_model_report():
    """Download ML model performance report with metrics, confusion matrix, feature importance."""
    if not state.has_data:
        raise HTTPException(status_code=400, detail="No data available.")
    html = build_model_report(state)
    return StreamingResponse(
        iter([html.encode("utf-8")]),
        media_type="text/html",
        headers={"Content-Disposition": "attachment; filename=model_report.html"},
    )
