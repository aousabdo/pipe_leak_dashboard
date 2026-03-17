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
def download_report():
    """Download a summary report as HTML."""
    if not state.has_data:
        raise HTTPException(status_code=400, detail="No data available.")

    overview = state.get_overview()
    kpis = overview["kpis"]
    pipes = state.pipes_gdf
    events = state.events_df

    # Material breakdown
    mat_rows = ""
    mat_dist = overview["material_distribution"]
    for mat, count in sorted(mat_dist.items(), key=lambda x: -x[1]):
        mat_rows += f"<tr><td>{mat}</td><td>{count}</td><td>{count / kpis['total_pipes'] * 100:.1f}%</td></tr>"

    # Severity breakdown
    sev_rows = ""
    sev_counts = overview["severity_counts"]
    for sev in ["Minor", "Moderate", "Major", "Critical"]:
        count = sev_counts.get(sev, 0)
        sev_rows += f"<tr><td>{sev}</td><td>{count}</td></tr>"

    # Cost by severity
    cost_rows = ""
    cost_by_sev = overview["cost_by_severity"]
    for sev in ["Minor", "Moderate", "Major", "Critical"]:
        cost = cost_by_sev.get(sev, 0)
        cost_rows += f"<tr><td>{sev}</td><td>${cost:,.2f}</td></tr>"

    # Top 10 riskiest pipes
    top_risky_rows = ""
    if state.risk_scores is not None:
        risk_df = pipes[["pipe_id", "material", "age", "diameter_m", "soil_type", "prev_repairs"]].copy()
        risk_df["risk_score"] = state.risk_scores
        risk_df = risk_df.sort_values("risk_score", ascending=False).head(10)
        for _, row in risk_df.iterrows():
            top_risky_rows += (
                f"<tr><td>{row['pipe_id']}</td><td>{row['risk_score']:.3f}</td>"
                f"<td>{row['material']}</td><td>{int(row['age'])} yr</td>"
                f"<td>{row['prev_repairs']}</td></tr>"
            )

    # Model info
    model_section = ""
    if state.has_model:
        metrics = state.metrics
        model_section = f"""
        <h2>Model Performance</h2>
        <table>
            <tr><td><strong>Model Type</strong></td><td>{state.model.model_type}</td></tr>
            <tr><td><strong>AUC-ROC</strong></td><td>{metrics.get('roc_auc', 0):.4f}</td></tr>
            <tr><td><strong>Precision</strong></td><td>{metrics.get('precision', 0):.4f}</td></tr>
            <tr><td><strong>Recall</strong></td><td>{metrics.get('recall', 0):.4f}</td></tr>
            <tr><td><strong>F1 Score</strong></td><td>{metrics.get('f1', 0):.4f}</td></tr>
            <tr><td><strong>Accuracy</strong></td><td>{metrics.get('accuracy', 0):.4f}</td></tr>
        </table>
        """

    report_date = datetime.now().strftime("%B %d, %Y")
    sim_params = state.sim_params or {}

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Water Network Leak Analysis Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #1e293b; line-height: 1.6; }}
  h1 {{ color: #0f172a; border-bottom: 3px solid #3b82f6; padding-bottom: 10px; }}
  h2 {{ color: #1e40af; margin-top: 30px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px 0; }}
  th, td {{ border: 1px solid #e2e8f0; padding: 8px 12px; text-align: left; }}
  th {{ background: #f1f5f9; font-weight: 600; color: #475569; }}
  tr:nth-child(even) {{ background: #f8fafc; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 20px 0; }}
  .kpi-card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; text-align: center; }}
  .kpi-value {{ font-size: 24px; font-weight: 700; color: #1e40af; }}
  .kpi-label {{ font-size: 12px; color: #64748b; margin-top: 4px; }}
  .meta {{ color: #64748b; font-size: 14px; }}
  @media print {{ body {{ margin: 20px; }} }}
</style>
</head>
<body>
<h1>Water Network Leak Analysis Report</h1>
<p class="meta">Generated: {report_date} | Pipes: {sim_params.get('num_pipes', 'N/A')} | Simulation Years: {sim_params.get('sim_years', 'N/A')} | Seed: {sim_params.get('seed', 'N/A')}</p>

<h2>Key Performance Indicators</h2>
<div class="kpi-grid">
  <div class="kpi-card"><div class="kpi-value">{kpis['total_pipes']}</div><div class="kpi-label">Total Pipes</div></div>
  <div class="kpi-card"><div class="kpi-value">{kpis['total_events']}</div><div class="kpi-label">Total Leak Events</div></div>
  <div class="kpi-card"><div class="kpi-value">${kpis['total_cost']:,.0f}</div><div class="kpi-label">Total Repair Cost</div></div>
  <div class="kpi-card"><div class="kpi-value">{kpis['total_water_loss']:,.0f}</div><div class="kpi-label">Water Loss (gal)</div></div>
  <div class="kpi-card"><div class="kpi-value">{kpis['avg_pipe_age']:.1f} yr</div><div class="kpi-label">Avg Pipe Age</div></div>
  <div class="kpi-card"><div class="kpi-value">{kpis['high_risk_pipes']}</div><div class="kpi-label">High Risk Pipes</div></div>
  <div class="kpi-card"><div class="kpi-value">${kpis['avg_cost_per_event']:,.0f}</div><div class="kpi-label">Avg Cost/Event</div></div>
  <div class="kpi-card"><div class="kpi-value">{kpis['events_per_pipe']:.2f}</div><div class="kpi-label">Events/Pipe</div></div>
</div>

<h2>Event Severity Breakdown</h2>
<table>
  <thead><tr><th>Severity</th><th>Count</th></tr></thead>
  <tbody>{sev_rows}</tbody>
</table>

<h2>Repair Costs by Severity</h2>
<table>
  <thead><tr><th>Severity</th><th>Total Cost</th></tr></thead>
  <tbody>{cost_rows}</tbody>
</table>

<h2>Material Distribution</h2>
<table>
  <thead><tr><th>Material</th><th>Pipe Count</th><th>Percentage</th></tr></thead>
  <tbody>{mat_rows}</tbody>
</table>

{"<h2>Top 10 Riskiest Pipes</h2>" + '''
<table>
  <thead><tr><th>Pipe ID</th><th>Risk Score</th><th>Material</th><th>Age</th><th>Prev Repairs</th></tr></thead>
  <tbody>''' + top_risky_rows + "</tbody></table>" if top_risky_rows else ""}

{model_section}

<hr>
<p class="meta" style="text-align: center;">Water Network Leak Predictor &mdash; Hydraulic simulation &bull; Weibull deterioration &bull; ML risk scoring</p>
</body>
</html>"""

    buf = io.BytesIO(html.encode("utf-8"))
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/html",
        headers={"Content-Disposition": "attachment; filename=leak_analysis_report.html"},
    )
