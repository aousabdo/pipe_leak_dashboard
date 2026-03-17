"""Overview page: rich KPI cards, summary charts, and network stats."""

import streamlit as st
import pandas as pd
import numpy as np

from pipe_leak.dashboard.components.charts import (
    cost_by_severity_chart,
    water_loss_timeline,
    pipe_age_distribution,
)


def _kpi_card(icon: str, value: str, label: str, accent: str = "blue", delta: str = "") -> str:
    """Generate HTML for a single KPI card."""
    delta_html = ""
    if delta:
        delta_html = f'<div class="kpi-delta neutral">{delta}</div>'
    return f"""
    <div class="kpi-card accent-{accent}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>
    """


def render(pipes_df: pd.DataFrame, events_df: pd.DataFrame, metrics: dict | None = None):
    """Render the overview page."""

    n_pipes = len(pipes_df) if pipes_df is not None else 0
    n_events = len(events_df) if events_df is not None and not events_df.empty else 0
    has_events = events_df is not None and not events_df.empty

    # Primary KPIs
    cols = st.columns(4)

    with cols[0]:
        st.markdown(_kpi_card(
            "🔧", f"{n_pipes:,}", "Total Pipes", "blue",
            f"{pipes_df['material'].nunique()} materials" if pipes_df is not None else "",
        ), unsafe_allow_html=True)

    with cols[1]:
        st.markdown(_kpi_card(
            "💧", f"{n_events:,}", "Leak Events", "red",
            f"{events_df['pipe_id'].nunique():,} unique pipes" if has_events else "",
        ), unsafe_allow_html=True)

    with cols[2]:
        if has_events:
            total_cost = events_df["repair_cost"].sum()
            avg_cost = events_df["repair_cost"].mean()
            st.markdown(_kpi_card(
                "💰", f"${total_cost / 1e6:.1f}M", "Total Repair Cost", "amber",
                f"Avg ${avg_cost:,.0f} per event",
            ), unsafe_allow_html=True)
        else:
            st.markdown(_kpi_card("💰", "$0", "Total Repair Cost", "amber"), unsafe_allow_html=True)

    with cols[3]:
        if has_events:
            total_loss = events_df["water_loss_gallons"].sum()
            st.markdown(_kpi_card(
                "🌊", f"{total_loss / 1e6:.1f}M", "Gallons Lost", "cyan",
                "Total water loss",
            ), unsafe_allow_html=True)
        else:
            st.markdown(_kpi_card("🌊", "0", "Gallons Lost", "cyan"), unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # Secondary KPIs
    cols2 = st.columns(4)

    with cols2[0]:
        if n_pipes > 0 and has_events:
            unique_leaked = events_df["pipe_id"].nunique()
            rate = unique_leaked / n_pipes * 100
            st.markdown(_kpi_card(
                "📊", f"{rate:.1f}%", "Leak Rate", "purple",
            ), unsafe_allow_html=True)
        else:
            st.markdown(_kpi_card("📊", "0%", "Leak Rate", "purple"), unsafe_allow_html=True)

    with cols2[1]:
        if has_events:
            critical = (events_df["severity"] == "Critical").sum()
            major = (events_df["severity"] == "Major").sum()
            st.markdown(_kpi_card(
                "🚨", f"{critical + major:,}", "Critical + Major", "red",
                f"{critical} critical, {major} major",
            ), unsafe_allow_html=True)
        else:
            st.markdown(_kpi_card("🚨", "0", "Critical + Major", "red"), unsafe_allow_html=True)

    with cols2[2]:
        if pipes_df is not None:
            avg_age = pipes_df["age"].mean()
            oldest = pipes_df["age"].max()
            st.markdown(_kpi_card(
                "📅", f"{avg_age:.0f} yrs", "Avg Pipe Age", "blue",
                f"Oldest: {oldest} years",
            ), unsafe_allow_html=True)

    with cols2[3]:
        if metrics and "roc_auc" in metrics:
            st.markdown(_kpi_card(
                "🤖", f"{metrics['roc_auc']:.3f}", "Model ROC AUC", "green",
                f"PR AUC: {metrics.get('pr_auc', 0):.3f}" if "pr_auc" in metrics else "",
            ), unsafe_allow_html=True)
        else:
            st.markdown(_kpi_card("🤖", "N/A", "Model AUC", "green"), unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Charts row
    if has_events:
        st.markdown('<div class="section-title">Network Overview</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(cost_by_severity_chart(events_df), use_container_width=True)
        with col2:
            st.plotly_chart(water_loss_timeline(events_df), use_container_width=True)
        with col3:
            st.plotly_chart(pipe_age_distribution(pipes_df), use_container_width=True)

    # Network summary table
    if pipes_df is not None and not pipes_df.empty:
        st.markdown('<div class="section-title">Pipe Inventory</div>', unsafe_allow_html=True)

        summary = pipes_df.groupby("material").agg(
            Pipes=("pipe_id", "size"),
            Avg_Age=("age", "mean"),
            Max_Age=("age", "max"),
            Avg_Pressure=("pressure_avg_m", "mean"),
            Avg_Length=("length_m", "mean"),
        ).round(1).sort_values("Pipes", ascending=False)

        summary.columns = ["Pipes", "Avg Age (yr)", "Max Age (yr)", "Avg Pressure (m)", "Avg Length (m)"]

        st.dataframe(
            summary,
            use_container_width=True,
            column_config={
                "Pipes": st.column_config.NumberColumn(format="%d"),
                "Avg Age (yr)": st.column_config.NumberColumn(format="%.0f"),
                "Max Age (yr)": st.column_config.NumberColumn(format="%d"),
                "Avg Pressure (m)": st.column_config.NumberColumn(format="%.1f"),
                "Avg Length (m)": st.column_config.NumberColumn(format="%.0f"),
            },
        )
