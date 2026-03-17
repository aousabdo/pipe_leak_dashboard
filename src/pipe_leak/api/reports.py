"""
HTML report generators with embedded inline charts.

Produces self-contained HTML files (no external dependencies) with:
- CSS bar/donut charts rendered as inline HTML
- Data tables with styling
- KPI cards
"""

from datetime import datetime

# ── Shared CSS & helpers ─────────────────────────────────────────────────────

_SEVERITY_COLORS = {"Minor": "#22c55e", "Moderate": "#f59e0b", "Major": "#f97316", "Critical": "#ef4444"}
_MATERIAL_COLORS = ["#3b82f6", "#8b5cf6", "#06b6d4", "#f59e0b", "#ef4444", "#22c55e", "#ec4899"]


def _fmt(n: float) -> str:
    if n >= 1_000_000:
        return f"${n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"${n / 1_000:,.0f}K"
    return f"${n:,.0f}"


def _pct(n: float, total: float) -> str:
    return f"{n / max(total, 1) * 100:.1f}%"


def _bar_chart_html(items: list[tuple[str, float, str]], max_val: float = 0, title: str = "", subtitle: str = "") -> str:
    """Render a horizontal bar chart as pure HTML/CSS."""
    if max_val == 0:
        max_val = max((v for _, v, _ in items), default=1)
    bars = ""
    for label, value, color in items:
        width_pct = min(value / max(max_val, 1) * 100, 100)
        bars += f"""
        <div style="display:flex;align-items:center;margin:6px 0;">
          <div style="width:120px;font-size:12px;color:#475569;text-align:right;padding-right:10px;">{label}</div>
          <div style="flex:1;background:#f1f5f9;border-radius:4px;height:22px;overflow:hidden;">
            <div style="width:{width_pct:.1f}%;background:{color};height:100%;border-radius:4px;transition:width .3s;"></div>
          </div>
          <div style="width:70px;font-size:11px;color:#64748b;text-align:right;padding-left:8px;">{value:,.1f}</div>
        </div>"""
    header = ""
    if title:
        header = f'<h3 style="margin:0 0 2px 0;color:#1e293b;font-size:15px;">{title}</h3>'
    if subtitle:
        header += f'<p style="margin:0 0 8px 0;color:#94a3b8;font-size:11px;">{subtitle}</p>'
    return f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;margin:12px 0;">{header}{bars}</div>'


def _stacked_bar_html(data: list[dict], categories: list[str], colors: dict, value_key_prefix: str = "",
                      x_key: str = "year", title: str = "", subtitle: str = "") -> str:
    """Render a stacked vertical bar chart as pure HTML/CSS."""
    if not data:
        return ""
    max_total = max(sum(row.get(c, 0) for c in categories) for row in data)
    chart_height = 200
    bars_html = ""
    for row in data:
        total = sum(row.get(c, 0) for c in categories)
        segments = ""
        for cat in categories:
            val = row.get(cat, 0)
            h_pct = val / max(max_total, 1) * 100
            segments += f'<div style="height:{h_pct:.1f}%;background:{colors.get(cat, "#94a3b8")};width:100%;" title="{cat}: {_fmt(val)}"></div>'
        bars_html += f"""
        <div style="display:flex;flex-direction:column;align-items:center;flex:1;">
          <div style="height:{chart_height}px;width:100%;display:flex;flex-direction:column-reverse;gap:1px;padding:0 4px;">
            {segments}
          </div>
          <div style="font-size:11px;color:#64748b;margin-top:4px;">{row.get(x_key, "")}</div>
        </div>"""
    legend = " ".join(
        f'<span style="display:inline-flex;align-items:center;gap:4px;margin-right:12px;">'
        f'<span style="width:10px;height:10px;background:{colors.get(c, "#94a3b8")};border-radius:2px;display:inline-block;"></span>'
        f'<span style="font-size:11px;color:#64748b;">{c}</span></span>'
        for c in categories
    )
    header = ""
    if title:
        header = f'<h3 style="margin:0 0 2px 0;color:#1e293b;font-size:15px;">{title}</h3>'
    if subtitle:
        header += f'<p style="margin:0 0 8px 0;color:#94a3b8;font-size:11px;">{subtitle}</p>'
    return f"""
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;margin:12px 0;">
      {header}
      <div style="display:flex;align-items:flex-end;gap:2px;">{bars_html}</div>
      <div style="text-align:center;margin-top:8px;">{legend}</div>
    </div>"""


def _table_html(headers: list[str], rows: list[list[str]], title: str = "", subtitle: str = "") -> str:
    hdr = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for row in rows:
        cells = "".join(f"<td>{c}</td>" for c in row)
        body += f"<tr>{cells}</tr>"
    header = ""
    if title:
        header = f'<h3 style="margin:0 0 2px 0;color:#1e293b;font-size:15px;">{title}</h3>'
    if subtitle:
        header += f'<p style="margin:0 0 8px 0;color:#94a3b8;font-size:11px;">{subtitle}</p>'
    return f"""{header}<table><thead><tr>{hdr}</tr></thead><tbody>{body}</tbody></table>"""


def _kpi_card(value: str, label: str) -> str:
    return f'<div class="kpi-card"><div class="kpi-value">{value}</div><div class="kpi-label">{label}</div></div>'


def _wrap_html(title: str, body: str, sim_params: dict | None = None) -> str:
    report_date = datetime.now().strftime("%B %d, %Y")
    params = sim_params or {}
    meta_parts = [f"Generated: {report_date}"]
    if params.get("num_pipes"):
        meta_parts.append(f"Pipes: {params['num_pipes']}")
    if params.get("sim_years"):
        meta_parts.append(f"Years: {params['sim_years']}")
    if params.get("seed") is not None:
        meta_parts.append(f"Seed: {params['seed']}")
    meta = " | ".join(meta_parts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 20px; color: #1e293b; line-height: 1.6; }}
  h1 {{ color: #0f172a; border-bottom: 3px solid #3b82f6; padding-bottom: 10px; }}
  h2 {{ color: #1e40af; margin-top: 30px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; }}
  h3 {{ color: #334155; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px 0; }}
  th, td {{ border: 1px solid #e2e8f0; padding: 8px 12px; text-align: left; }}
  th {{ background: #f1f5f9; font-weight: 600; color: #475569; }}
  tr:nth-child(even) {{ background: #f8fafc; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 20px 0; }}
  .kpi-card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; text-align: center; }}
  .kpi-value {{ font-size: 24px; font-weight: 700; color: #1e40af; }}
  .kpi-label {{ font-size: 12px; color: #64748b; margin-top: 4px; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .meta {{ color: #64748b; font-size: 14px; }}
  .metric-bar {{ display: flex; align-items: center; margin: 8px 0; }}
  .metric-bar .bar-bg {{ flex: 1; height: 18px; background: #f1f5f9; border-radius: 4px; overflow: hidden; }}
  .metric-bar .bar-fill {{ height: 100%; border-radius: 4px; }}
  .metric-bar .bar-label {{ width: 100px; font-size: 12px; color: #475569; }}
  .metric-bar .bar-value {{ width: 60px; text-align: right; font-size: 12px; font-weight: 600; color: #1e293b; }}
  .cm-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; max-width: 300px; margin: 12px auto; }}
  .cm-cell {{ border-radius: 8px; padding: 12px; text-align: center; border: 1px solid #e2e8f0; }}
  .cm-cell .val {{ font-size: 22px; font-weight: 700; }}
  .cm-cell .lbl {{ font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; opacity: 0.7; }}
  @media print {{ body {{ margin: 20px; }} .two-col {{ grid-template-columns: 1fr 1fr; }} }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">{meta}</p>
{body}
<hr>
<p class="meta" style="text-align:center;">Water Network Leak Predictor &mdash; Hydraulic simulation &bull; Weibull deterioration &bull; ML risk scoring</p>
</body>
</html>"""


# ── Overview Report ──────────────────────────────────────────────────────────

def build_overview_report(state) -> str:
    """Build the overview/summary report with KPIs, severity, materials, and top risks."""
    overview = state.get_overview()
    kpis = overview["kpis"]
    pipes = state.pipes_gdf
    events = state.events_df

    body = '<h2>Key Performance Indicators</h2>\n<div class="kpi-grid">'
    body += _kpi_card(str(kpis["total_pipes"]), "Total Pipes")
    body += _kpi_card(str(kpis["total_events"]), "Leak Events")
    body += _kpi_card(_fmt(kpis["total_cost"]), "Total Repair Cost")
    body += _kpi_card(f"{kpis['total_water_loss']:,.0f} gal", "Water Loss")
    body += _kpi_card(f"{kpis['avg_pipe_age']:.1f} yr", "Avg Pipe Age")
    body += _kpi_card(str(kpis["high_risk_pipes"]), "High Risk Pipes")
    body += _kpi_card(_fmt(kpis["avg_cost_per_event"]), "Avg Cost/Event")
    body += _kpi_card(f"{kpis['events_per_pipe']:.2f}", "Events/Pipe")
    body += "</div>"

    # Severity breakdown
    sev_counts = overview["severity_counts"]
    sev_rows = [[sev, str(sev_counts.get(sev, 0))] for sev in ["Minor", "Moderate", "Major", "Critical"]]
    body += _table_html(["Severity", "Count"], sev_rows, title="Event Severity Breakdown")

    # Cost by severity
    cost_by_sev = overview["cost_by_severity"]
    cost_rows = [[sev, _fmt(cost_by_sev.get(sev, 0))] for sev in ["Minor", "Moderate", "Major", "Critical"]]
    body += _table_html(["Severity", "Total Cost"], cost_rows, title="Repair Costs by Severity")

    # Material distribution chart + table
    mat_dist = overview["material_distribution"]
    mat_items = sorted(mat_dist.items(), key=lambda x: -x[1])
    mat_bars = [(m, float(c), _MATERIAL_COLORS[i % len(_MATERIAL_COLORS)]) for i, (m, c) in enumerate(mat_items)]
    body += '<div class="two-col">'
    body += _bar_chart_html(mat_bars, title="Material Distribution", subtitle="Pipe count by material")
    mat_rows = [[m, str(c), _pct(c, kpis["total_pipes"])] for m, c in mat_items]
    body += _table_html(["Material", "Count", "%"], mat_rows)
    body += "</div>"

    # Age distribution chart
    age_dist = overview.get("age_distribution", {})
    if age_dist:
        age_bars = [(k, float(v), "#3b82f6") for k, v in age_dist.items()]
        body += _bar_chart_html(age_bars, title="Pipe Age Distribution", subtitle="Number of pipes by age range")

    # Top 10 riskiest pipes
    if state.risk_scores is not None:
        risk_df = pipes[["pipe_id", "material", "age", "diameter_m", "soil_type", "prev_repairs"]].copy()
        risk_df["risk_score"] = state.risk_scores
        risk_df = risk_df.sort_values("risk_score", ascending=False).head(10)
        risk_rows = [
            [row["pipe_id"], f"{row['risk_score']:.3f}", row["material"],
             f"{int(row['age'])} yr", str(row["prev_repairs"])]
            for _, row in risk_df.iterrows()
        ]
        body += _table_html(
            ["Pipe ID", "Risk Score", "Material", "Age", "Prev Repairs"],
            risk_rows, title="Top 10 Riskiest Pipes",
        )

    return _wrap_html("Water Network Overview Report", body, state.sim_params)


# ── Analysis Report ──────────────────────────────────────────────────────────

def build_analysis_report(state) -> str:
    """Build the analysis report with trends, root causes, and cost analysis."""
    analysis = state.get_analysis()
    overview = state.get_overview()
    kpis = overview["kpis"]

    # Insights banner
    ins = analysis["insights"]
    body = '<div class="kpi-grid">'
    body += _kpi_card(ins["peak_month"], "Peak Month")
    body += _kpi_card(ins["most_affected_material"], "Most At-Risk Material")
    body += _kpi_card(_fmt(ins["total_cost"]), "Total Repair Cost")
    body += _kpi_card(f'{ins["avg_detection_hours"]} hrs', "Avg Detection Time")
    body += "</div>"

    # Yearly trend chart
    yearly = analysis["yearly_trend"]
    if yearly:
        max_count = max(r["count"] for r in yearly)
        bars = [(str(r["year"]), float(r["count"]), "#3b82f6") for r in yearly]
        body += _bar_chart_html(bars, max_val=max_count, title="Yearly Event Count", subtitle="Leak events per year")

        # Yearly cost
        cost_bars = [(str(r["year"]), float(r["cost"]), "#ef4444") for r in yearly]
        body += _bar_chart_html(cost_bars, title="Yearly Repair Cost", subtitle="Total cost per year")

    # Cost by severity over time (stacked)
    cost_sev = analysis.get("cost_by_severity_year", [])
    if cost_sev:
        pivoted = {}
        for row in cost_sev:
            yr = row["year"]
            if yr not in pivoted:
                pivoted[yr] = {"year": yr}
            pivoted[yr][row["severity"]] = row["repair_cost"]
        stacked_data = sorted(pivoted.values(), key=lambda x: x["year"])
        body += _stacked_bar_html(
            stacked_data, ["Minor", "Moderate", "Major", "Critical"], _SEVERITY_COLORS,
            title="Cost by Severity Over Time", subtitle="Annual breakdown",
        )

    # Material risk profile
    mat_risk = analysis.get("material_risk", [])
    if mat_risk:
        mat_bars = [
            (r["material"], float(r["events_per_pipe"]), _MATERIAL_COLORS[i % len(_MATERIAL_COLORS)])
            for i, r in enumerate(mat_risk)
        ]
        body += _bar_chart_html(mat_bars, title="Material Risk Profile", subtitle="Events per pipe by material")

        # Full material table
        mat_rows = [
            [r["material"], str(r.get("pipe_count", "")), str(r["event_count"]),
             str(r["events_per_pipe"]), _fmt(r["avg_cost"]), _fmt(r["total_cost"])]
            for r in mat_risk
        ]
        body += _table_html(
            ["Material", "Pipes", "Events", "Events/Pipe", "Avg Cost", "Total Cost"],
            mat_rows, title="Material Details",
        )

    # Soil analysis
    soil = analysis.get("soil_analysis", [])
    if soil:
        soil_bars = [(r["soil_type"], float(r["event_count"]), "#f59e0b") for r in soil]
        body += _bar_chart_html(soil_bars, title="Soil Type Impact", subtitle="Event count by soil type")

        soil_rows = [
            [r["soil_type"], str(r["event_count"]), _fmt(r["avg_cost"])]
            for r in soil
        ]
        body += _table_html(["Soil Type", "Events", "Avg Cost"], soil_rows)

    return _wrap_html("Leak Analysis Report", body, state.sim_params)


# ── ML Model Report ──────────────────────────────────────────────────────────

def _metric_bar(label: str, value: float, color: str = "#3b82f6") -> str:
    pct = min(value * 100, 100)
    return f"""
    <div class="metric-bar">
      <div class="bar-label">{label}</div>
      <div class="bar-bg"><div class="bar-fill" style="width:{pct:.1f}%;background:{color};"></div></div>
      <div class="bar-value">{pct:.1f}%</div>
    </div>"""


def build_model_report(state) -> str:
    """Build the ML model performance report with metrics, charts, and feature importance."""
    if not state.has_model:
        return _wrap_html("ML Model Report", "<p>No model has been trained yet.</p>", state.sim_params)

    perf = state.get_model_performance()
    metrics = perf["metrics"]
    cm = perf.get("confusion_matrix", [[0, 0], [0, 0]])
    feat_imp = perf.get("feature_importance", [])
    model_type = perf.get("model_type", "unknown")
    threshold = perf.get("optimal_threshold", 0.5)

    # Model info
    body = f"""
    <h2>Model Configuration</h2>
    <div class="kpi-grid">
      {_kpi_card(model_type.replace("_", " ").title(), "Model Type")}
      {_kpi_card(f"{threshold:.3f}", "Optimal Threshold")}
      {_kpi_card(str(metrics.get("n_positive", 0)), "Positive Samples")}
      {_kpi_card(str(metrics.get("n_negative", 0)), "Negative Samples")}
    </div>
    """

    # Performance metrics as visual bars
    body += "<h2>Performance Metrics</h2>"
    body += _metric_bar("Accuracy", metrics.get("accuracy", 0), "#3b82f6")
    body += _metric_bar("Precision", metrics.get("precision", 0), "#8b5cf6")
    body += _metric_bar("Recall", metrics.get("recall", 0), "#06b6d4")
    body += _metric_bar("F1 Score", metrics.get("f1", 0), "#22c55e")
    body += _metric_bar("ROC AUC", metrics.get("roc_auc", 0), "#3b82f6")
    body += _metric_bar("PR AUC", metrics.get("pr_auc", 0), "#f59e0b")
    body += f"""
    <div class="metric-bar">
      <div class="bar-label">Brier Score</div>
      <div class="bar-bg"><div class="bar-fill" style="width:{min(metrics.get('brier_score', 0) * 100, 100):.1f}%;background:#ef4444;"></div></div>
      <div class="bar-value">{metrics.get('brier_score', 0):.4f}</div>
    </div>"""

    # Metrics table
    metric_rows = [
        ["Accuracy", f"{metrics.get('accuracy', 0):.4f}"],
        ["Precision", f"{metrics.get('precision', 0):.4f}"],
        ["Recall", f"{metrics.get('recall', 0):.4f}"],
        ["F1 Score", f"{metrics.get('f1', 0):.4f}"],
        ["ROC AUC", f"{metrics.get('roc_auc', 0):.4f}"],
        ["PR AUC", f"{metrics.get('pr_auc', 0):.4f}"],
        ["Brier Score", f"{metrics.get('brier_score', 0):.4f}"],
    ]
    body += _table_html(["Metric", "Value"], metric_rows, title="Detailed Metrics")

    # Confusion matrix
    body += "<h2>Confusion Matrix</h2>"
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    total = tn + fp + fn + tp
    body += f"""
    <div class="cm-grid">
      <div class="cm-cell" style="background:#dcfce7;color:#166534;">
        <div class="val">{tn}</div><div class="lbl">True Neg ({tn / max(total, 1) * 100:.1f}%)</div>
      </div>
      <div class="cm-cell" style="background:#fef2f2;color:#dc2626;">
        <div class="val">{fp}</div><div class="lbl">False Pos ({fp / max(total, 1) * 100:.1f}%)</div>
      </div>
      <div class="cm-cell" style="background:#fff7ed;color:#ea580c;">
        <div class="val">{fn}</div><div class="lbl">False Neg ({fn / max(total, 1) * 100:.1f}%)</div>
      </div>
      <div class="cm-cell" style="background:#dbeafe;color:#1e40af;">
        <div class="val">{tp}</div><div class="lbl">True Pos ({tp / max(total, 1) * 100:.1f}%)</div>
      </div>
    </div>
    <div style="text-align:center;margin:8px 0;">
      <span style="font-size:11px;color:#64748b;">Predicted: No Leak &nbsp;|&nbsp; Predicted: Leak</span>
    </div>"""

    # Feature importance chart
    if feat_imp:
        body += "<h2>Feature Importance</h2>"
        max_imp = max(f["importance"] for f in feat_imp) if feat_imp else 1
        imp_bars = [
            (f["feature"], float(f["importance"]) * 100,
             "#3b82f6" if i == 0 else "#60a5fa" if i < 3 else "#93c5fd" if i < 6 else "#bfdbfe")
            for i, f in enumerate(feat_imp)
        ]
        body += _bar_chart_html(imp_bars, title="Top Features", subtitle="By importance score (%)")

        imp_rows = [[f["feature"], f"{f['importance'] * 100:.2f}%"] for f in feat_imp]
        body += _table_html(["Feature", "Importance"], imp_rows)

    return _wrap_html("ML Model Performance Report", body, state.sim_params)
