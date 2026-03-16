"""Reusable Plotly chart builders."""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


SEVERITY_COLORS = {
    "Minor": "#38a169",
    "Moderate": "#dd6b20",
    "Major": "#e53e3e",
    "Critical": "#9b2c2c",
}

SEVERITY_ORDER = ["Minor", "Moderate", "Major", "Critical"]


def severity_distribution(events_df: pd.DataFrame) -> go.Figure:
    """Bar chart of leak severity distribution."""
    if events_df is None or events_df.empty:
        return _empty_chart("Leak Severity Distribution")

    counts = events_df["severity"].value_counts().reset_index()
    counts.columns = ["Severity", "Count"]
    counts["Severity"] = pd.Categorical(counts["Severity"], categories=SEVERITY_ORDER, ordered=True)
    counts = counts.sort_values("Severity")

    fig = px.bar(
        counts, x="Severity", y="Count", color="Severity",
        color_discrete_map=SEVERITY_COLORS,
        title="Leak Severity Distribution",
    )
    fig.update_layout(showlegend=False, xaxis_title="Severity", yaxis_title="Count")
    return fig


def leaks_over_time(events_df: pd.DataFrame) -> go.Figure:
    """Line chart of leaks over time by severity."""
    if events_df is None or events_df.empty:
        return _empty_chart("Leaks Over Time")

    df = events_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    grouped = df.groupby([pd.Grouper(key="date", freq="M"), "severity"]).size().reset_index()
    grouped.columns = ["Date", "Severity", "Count"]

    fig = px.line(
        grouped, x="Date", y="Count", color="Severity",
        color_discrete_map=SEVERITY_COLORS,
        title="Leaks Over Time",
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Leak Count")
    return fig


def leak_rate_by_material(events_df: pd.DataFrame, pipes_df: pd.DataFrame) -> go.Figure:
    """Bar chart of leak rate by pipe material."""
    if events_df is None or events_df.empty:
        return _empty_chart("Leak Rate by Material")

    # Count leaks per material
    leak_counts = events_df.groupby("material").size().reset_index(name="leak_count")
    pipe_counts = pipes_df["material"].value_counts().reset_index()
    pipe_counts.columns = ["material", "pipe_count"]

    merged = leak_counts.merge(pipe_counts, on="material")
    merged["leak_rate_pct"] = merged["leak_count"] / merged["pipe_count"] * 100

    fig = px.bar(
        merged.sort_values("leak_rate_pct", ascending=False),
        x="material", y="leak_rate_pct",
        color="leak_rate_pct",
        color_continuous_scale="RdYlGn_r",
        title="Leak Rate by Pipe Material (%)",
    )
    fig.update_layout(xaxis_title="Material", yaxis_title="Leak Rate (%)")
    return fig


def leak_rate_by_age(events_df: pd.DataFrame, pipes_df: pd.DataFrame) -> go.Figure:
    """Bar chart of leak rate by pipe age bins."""
    if events_df is None or events_df.empty:
        return _empty_chart("Leak Rate by Age")

    bins = [0, 20, 40, 60, 80, 100, 200]
    labels = ["0-20", "21-40", "41-60", "61-80", "81-100", "100+"]

    # Events carry age_at_event
    ev = events_df.copy()
    age_col = "age_at_event" if "age_at_event" in ev.columns else "age"
    ev["age_bin"] = pd.cut(ev[age_col], bins=bins, labels=labels)
    leak_counts = ev.groupby("age_bin", observed=False).size().reset_index(name="leak_count")

    pipe_ages = pipes_df.copy()
    pipe_ages["age_bin"] = pd.cut(pipe_ages["age"], bins=bins, labels=labels)
    pipe_counts = pipe_ages.groupby("age_bin", observed=False).size().reset_index(name="pipe_count")

    merged = leak_counts.merge(pipe_counts, on="age_bin")
    merged["leak_rate_pct"] = np.where(
        merged["pipe_count"] > 0,
        merged["leak_count"] / merged["pipe_count"] * 100,
        0,
    )

    fig = px.bar(
        merged, x="age_bin", y="leak_rate_pct",
        color="leak_rate_pct",
        color_continuous_scale="RdYlGn_r",
        title="Leak Rate by Pipe Age (%)",
    )
    fig.update_layout(xaxis_title="Age Group (years)", yaxis_title="Leak Rate (%)")
    return fig


def model_metrics_chart(metrics: dict) -> go.Figure:
    """Bar chart of model performance metrics."""
    display_metrics = {}
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        if key in metrics:
            display_metrics[key.upper().replace("_", " ")] = metrics[key]

    if not display_metrics:
        return _empty_chart("Model Performance")

    fig = go.Figure(
        go.Bar(
            x=list(display_metrics.keys()),
            y=list(display_metrics.values()),
            marker_color=["#3182ce", "#38a169", "#dd6b20", "#e53e3e", "#805ad5", "#d69e2e"][
                : len(display_metrics)
            ],
        )
    )
    fig.update_layout(
        title="Model Performance Metrics",
        yaxis=dict(title="Score", range=[0, 1.05]),
        xaxis_title="Metric",
        showlegend=False,
    )
    return fig


def feature_importance_chart(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of feature importances."""
    if importance_df is None or importance_df.empty:
        return _empty_chart("Feature Importance")

    top = importance_df.head(top_n).iloc[::-1]  # reverse for horizontal bar

    fig = go.Figure(
        go.Bar(
            y=top["feature"],
            x=top["importance"],
            orientation="h",
            marker_color="#3182ce",
        )
    )
    fig.update_layout(
        title=f"Top {top_n} Feature Importances",
        xaxis_title="Importance",
        yaxis_title="Feature",
    )
    return fig


def _empty_chart(title: str) -> go.Figure:
    """Return an empty chart with a 'no data' message."""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        annotations=[
            dict(
                x=0.5, y=0.5, xref="paper", yref="paper",
                text="No data available",
                showarrow=False, font=dict(size=16, color="#999"),
            )
        ],
    )
    return fig
