"""Reusable Plotly chart builders — polished, publication-quality."""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Consistent color palette
SEVERITY_COLORS = {
    "Minor": "#22c55e",
    "Moderate": "#f59e0b",
    "Major": "#ef4444",
    "Critical": "#991b1b",
}

SEVERITY_ORDER = ["Minor", "Moderate", "Major", "Critical"]

# Shared layout defaults
_LAYOUT_DEFAULTS = dict(
    font=dict(family="Inter, -apple-system, sans-serif", size=12),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=50, b=40),
    title_font=dict(size=15, color="#1e293b"),
    xaxis=dict(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0"),
    yaxis=dict(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0"),
    hoverlabel=dict(bgcolor="#1e293b", font_size=12, font_color="white"),
)


def _apply_style(fig: go.Figure) -> go.Figure:
    """Apply shared styling to a figure."""
    fig.update_layout(**_LAYOUT_DEFAULTS)
    return fig


def severity_distribution(events_df: pd.DataFrame) -> go.Figure:
    """Gradient bar chart of leak severity distribution."""
    if events_df is None or events_df.empty:
        return _empty_chart("Leak Severity Distribution")

    counts = events_df["severity"].value_counts().reset_index()
    counts.columns = ["Severity", "Count"]
    counts["Severity"] = pd.Categorical(counts["Severity"], categories=SEVERITY_ORDER, ordered=True)
    counts = counts.sort_values("Severity")

    fig = go.Figure()
    for _, row in counts.iterrows():
        sev = row["Severity"]
        fig.add_trace(go.Bar(
            x=[sev], y=[row["Count"]],
            marker_color=SEVERITY_COLORS.get(sev, "#64748b"),
            marker_line=dict(width=0),
            name=sev,
            text=[f"{row['Count']:,}"],
            textposition="outside",
            textfont=dict(size=13, color="#334155"),
            hovertemplate=f"<b>{sev}</b><br>Count: {row['Count']:,}<extra></extra>",
        ))

    fig.update_layout(
        title="Leak Severity Distribution",
        xaxis_title="", yaxis_title="Number of Events",
        showlegend=False,
        bargap=0.3,
    )
    return _apply_style(fig)


def leaks_over_time(events_df: pd.DataFrame) -> go.Figure:
    """Area chart of leaks over time by severity."""
    if events_df is None or events_df.empty:
        return _empty_chart("Leaks Over Time")

    df = events_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    grouped = df.groupby([pd.Grouper(key="date", freq="M"), "severity"]).size().reset_index()
    grouped.columns = ["Date", "Severity", "Count"]

    fig = go.Figure()
    for sev in SEVERITY_ORDER:
        sev_data = grouped[grouped["Severity"] == sev]
        if not sev_data.empty:
            fig.add_trace(go.Scatter(
                x=sev_data["Date"], y=sev_data["Count"],
                mode="lines",
                name=sev,
                line=dict(color=SEVERITY_COLORS[sev], width=2.5),
                fill="tonexty" if sev != "Minor" else "tozeroy",
                fillcolor=SEVERITY_COLORS[sev].replace(")", ",0.1)").replace("rgb", "rgba")
                if "rgb" in SEVERITY_COLORS[sev]
                else SEVERITY_COLORS[sev] + "18",
                hovertemplate=f"<b>{sev}</b><br>Date: %{{x|%b %Y}}<br>Count: %{{y}}<extra></extra>",
            ))

    fig.update_layout(
        title="Leak Events Over Time",
        xaxis_title="", yaxis_title="Monthly Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return _apply_style(fig)


def leak_rate_by_material(events_df: pd.DataFrame, pipes_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of leak rate by pipe material."""
    if events_df is None or events_df.empty:
        return _empty_chart("Leak Rate by Material")

    leak_counts = events_df.groupby("material").size().reset_index(name="leak_count")
    pipe_counts = pipes_df["material"].value_counts().reset_index()
    pipe_counts.columns = ["material", "pipe_count"]

    merged = leak_counts.merge(pipe_counts, on="material")
    merged["leak_rate"] = (merged["leak_count"] / merged["pipe_count"] * 100).round(1)
    merged = merged.sort_values("leak_rate", ascending=True)

    # Color gradient based on rate
    max_rate = merged["leak_rate"].max()
    colors = [
        f"rgb({int(50 + 180 * r / max_rate)}, {int(180 - 130 * r / max_rate)}, {int(80 - 40 * r / max_rate)})"
        for r in merged["leak_rate"]
    ]

    fig = go.Figure(go.Bar(
        y=merged["material"],
        x=merged["leak_rate"],
        orientation="h",
        marker_color=colors,
        marker_line=dict(width=0),
        text=[f"{r:.1f}%" for r in merged["leak_rate"]],
        textposition="outside",
        textfont=dict(size=12, color="#334155"),
        hovertemplate="<b>%{y}</b><br>Leak Rate: %{x:.1f}%<br>Leaks: %{customdata[0]:,}<br>Pipes: %{customdata[1]:,}<extra></extra>",
        customdata=merged[["leak_count", "pipe_count"]].values,
    ))

    fig.update_layout(
        title="Leak Rate by Pipe Material",
        xaxis_title="Leak Rate (%)", yaxis_title="",
    )
    return _apply_style(fig)


def leak_rate_by_age(events_df: pd.DataFrame, pipes_df: pd.DataFrame) -> go.Figure:
    """Bar chart of leak rate by pipe age bins with gradient colors."""
    if events_df is None or events_df.empty:
        return _empty_chart("Leak Rate by Age")

    bins = [0, 20, 40, 60, 80, 100, 200]
    labels = ["0-20", "21-40", "41-60", "61-80", "81-100", "100+"]

    ev = events_df.copy()
    age_col = "age_at_event" if "age_at_event" in ev.columns else "age"
    ev["age_bin"] = pd.cut(ev[age_col], bins=bins, labels=labels)
    leak_counts = ev.groupby("age_bin", observed=False).size().reset_index(name="leak_count")

    pipe_ages = pipes_df.copy()
    pipe_ages["age_bin"] = pd.cut(pipe_ages["age"], bins=bins, labels=labels)
    pipe_counts = pipe_ages.groupby("age_bin", observed=False).size().reset_index(name="pipe_count")

    merged = leak_counts.merge(pipe_counts, on="age_bin")
    merged["leak_rate"] = np.where(
        merged["pipe_count"] > 0,
        (merged["leak_count"] / merged["pipe_count"] * 100).round(1),
        0,
    )

    # Age gradient: young=blue, old=red
    n = len(merged)
    colors = [f"rgb({int(50 + 200 * i / (n - 1))}, {int(130 - 90 * i / (n - 1))}, {int(220 - 170 * i / (n - 1))})" for i in range(n)]

    fig = go.Figure(go.Bar(
        x=merged["age_bin"].astype(str),
        y=merged["leak_rate"],
        marker_color=colors,
        marker_line=dict(width=0),
        text=[f"{r:.1f}%" for r in merged["leak_rate"]],
        textposition="outside",
        textfont=dict(size=12, color="#334155"),
        hovertemplate="<b>Age %{x} years</b><br>Leak Rate: %{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        title="Leak Rate by Pipe Age",
        xaxis_title="Age Group (years)", yaxis_title="Leak Rate (%)",
        showlegend=False,
        bargap=0.25,
    )
    return _apply_style(fig)


def model_metrics_chart(metrics: dict) -> go.Figure:
    """Gauge-style metric cards as a grouped bar chart."""
    display = {}
    for key, label in [
        ("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
        ("f1", "F1 Score"), ("roc_auc", "ROC AUC"), ("pr_auc", "PR AUC"),
    ]:
        if key in metrics:
            display[label] = metrics[key]

    if not display:
        return _empty_chart("Model Performance")

    names = list(display.keys())
    values = list(display.values())
    colors = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"][:len(names)]

    fig = go.Figure(go.Bar(
        x=names, y=values,
        marker_color=colors,
        marker_line=dict(width=0),
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        textfont=dict(size=13, color="#334155", weight="bold"),
        hovertemplate="<b>%{x}</b><br>Score: %{y:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title="Model Performance Metrics",
        yaxis=dict(title="Score", range=[0, max(values) * 1.15]),
        xaxis_title="",
        showlegend=False,
        bargap=0.35,
    )
    return _apply_style(fig)


def feature_importance_chart(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal lollipop chart of feature importances."""
    if importance_df is None or importance_df.empty:
        return _empty_chart("Feature Importance")

    top = importance_df.head(top_n).iloc[::-1]

    fig = go.Figure()

    # Lollipop stems
    for i, (_, row) in enumerate(top.iterrows()):
        fig.add_trace(go.Scatter(
            x=[0, row["importance"]],
            y=[row["feature"], row["feature"]],
            mode="lines",
            line=dict(color="#cbd5e1", width=2),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Lollipop dots
    fig.add_trace(go.Scatter(
        x=top["importance"],
        y=top["feature"],
        mode="markers",
        marker=dict(
            size=10,
            color=top["importance"],
            colorscale=[[0, "#93c5fd"], [1, "#1d4ed8"]],
            line=dict(width=1, color="white"),
        ),
        text=[f"{v:.4f}" for v in top["importance"]],
        hovertemplate="<b>%{y}</b><br>Importance: %{text}<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        title=f"Top {min(top_n, len(top))} Feature Importances",
        xaxis_title="Importance Score",
        yaxis_title="",
        height=max(400, top_n * 30),
    )
    return _apply_style(fig)


def cost_by_severity_chart(events_df: pd.DataFrame) -> go.Figure:
    """Donut chart of total repair cost by severity."""
    if events_df is None or events_df.empty:
        return _empty_chart("Cost by Severity")

    costs = events_df.groupby("severity")["repair_cost"].sum().reset_index()
    costs.columns = ["Severity", "Cost"]
    costs["Severity"] = pd.Categorical(costs["Severity"], categories=SEVERITY_ORDER, ordered=True)
    costs = costs.sort_values("Severity")

    fig = go.Figure(go.Pie(
        labels=costs["Severity"],
        values=costs["Cost"],
        hole=0.55,
        marker_colors=[SEVERITY_COLORS[s] for s in costs["Severity"]],
        textinfo="label+percent",
        textfont=dict(size=12),
        hovertemplate="<b>%{label}</b><br>Cost: $%{value:,.0f}<br>Share: %{percent}<extra></extra>",
    ))

    total = costs["Cost"].sum()
    fig.update_layout(
        title="Repair Cost by Severity",
        annotations=[dict(
            text=f"<b>${total / 1e6:.1f}M</b><br>Total",
            x=0.5, y=0.5, font_size=16, font_color="#1e293b",
            showarrow=False,
        )],
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    return _apply_style(fig)


def water_loss_timeline(events_df: pd.DataFrame) -> go.Figure:
    """Cumulative water loss over time."""
    if events_df is None or events_df.empty:
        return _empty_chart("Cumulative Water Loss")

    df = events_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["cumulative_loss"] = df["water_loss_gallons"].cumsum() / 1e6  # millions

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["cumulative_loss"],
        mode="lines",
        fill="tozeroy",
        line=dict(color="#3b82f6", width=2.5),
        fillcolor="rgba(59, 130, 246, 0.1)",
        hovertemplate="Date: %{x|%b %d, %Y}<br>Cumulative Loss: %{y:.2f}M gal<extra></extra>",
    ))

    fig.update_layout(
        title="Cumulative Water Loss",
        xaxis_title="", yaxis_title="Million Gallons",
    )
    return _apply_style(fig)


def pipe_age_distribution(pipes_df: pd.DataFrame) -> go.Figure:
    """Histogram of pipe ages with material color coding."""
    if pipes_df is None or pipes_df.empty:
        return _empty_chart("Pipe Age Distribution")

    fig = go.Figure()
    material_colors = {
        "Cast Iron": "#ef4444", "Asbestos Cement": "#f59e0b", "Steel": "#64748b",
        "Ductile Iron": "#3b82f6", "PVC": "#22c55e", "HDPE": "#06b6d4",
    }

    for material in pipes_df["material"].unique():
        subset = pipes_df[pipes_df["material"] == material]
        fig.add_trace(go.Histogram(
            x=subset["age"],
            name=material,
            marker_color=material_colors.get(material, "#94a3b8"),
            opacity=0.75,
            nbinsx=20,
        ))

    fig.update_layout(
        title="Pipe Age Distribution by Material",
        xaxis_title="Age (years)", yaxis_title="Number of Pipes",
        barmode="stack",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return _apply_style(fig)


def _empty_chart(title: str) -> go.Figure:
    """Return a clean empty chart placeholder."""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="No data available",
            showarrow=False,
            font=dict(size=18, color="#cbd5e1", family="Inter, sans-serif"),
        )],
    )
    return fig
