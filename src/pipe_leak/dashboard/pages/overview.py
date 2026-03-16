"""Overview page: KPI cards and summary statistics."""

import streamlit as st
import pandas as pd
import numpy as np


def render(pipes_df: pd.DataFrame, events_df: pd.DataFrame, metrics: dict | None = None):
    """Render the overview page with key metrics."""
    st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    n_pipes = len(pipes_df) if pipes_df is not None else 0
    n_events = len(events_df) if events_df is not None and not events_df.empty else 0

    with col1:
        st.metric("Total Pipes", f"{n_pipes:,}")

    with col2:
        st.metric("Total Leak Events", f"{n_events:,}")

    with col3:
        if n_pipes > 0 and n_events > 0:
            # Unique pipes that leaked / total pipes
            unique_leaked = events_df["pipe_id"].nunique()
            rate = unique_leaked / n_pipes * 100
            st.metric("Pipes with Leaks", f"{rate:.1f}%")
        else:
            st.metric("Pipes with Leaks", "0%")

    with col4:
        if events_df is not None and not events_df.empty:
            total_cost = events_df["repair_cost"].sum()
            st.metric("Total Repair Cost", f"${total_cost:,.0f}")
        else:
            st.metric("Total Repair Cost", "$0")

    # Second row — more detail
    if events_df is not None and not events_df.empty:
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            avg_cost = events_df["repair_cost"].mean()
            st.metric("Avg Repair Cost", f"${avg_cost:,.0f}")

        with col6:
            total_water_loss = events_df["water_loss_gallons"].sum()
            st.metric("Total Water Loss", f"{total_water_loss / 1e6:.1f}M gal")

        with col7:
            critical_count = (events_df["severity"] == "Critical").sum()
            st.metric("Critical Events", f"{critical_count:,}")

        with col8:
            if metrics and "roc_auc" in metrics:
                st.metric("Model AUC", f"{metrics['roc_auc']:.3f}")
            elif metrics and "accuracy" in metrics:
                st.metric("Model Accuracy", f"{metrics['accuracy']:.3f}")
            else:
                st.metric("Model AUC", "N/A")

    # Summary table
    if pipes_df is not None and not pipes_df.empty:
        with st.expander("Pipe Network Summary"):
            summary = pipes_df.groupby("material").agg(
                count=("pipe_id", "size"),
                avg_age=("age", "mean"),
                avg_pressure=("pressure_avg_m", "mean"),
            ).round(1)
            st.dataframe(summary, use_container_width=True)
