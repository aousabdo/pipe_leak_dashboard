"""Map view page: interactive pydeck maps with stats panels."""

import streamlit as st
import pandas as pd
import numpy as np

from pipe_leak.dashboard.components.maps import (
    pipe_network_map,
    leak_events_map,
    risk_heatmap,
)


def render(
    pipes_df: pd.DataFrame,
    events_df: pd.DataFrame,
    risk_scores: np.ndarray | None = None,
):
    """Render the map view page."""
    tabs = st.tabs(["🔧 Pipe Network", "💧 Historical Leaks", "🔥 Risk Heatmap"])

    with tabs[0]:
        st.markdown(
            "Pipe segments colored by predicted leak risk — "
            "<span style='color:#22c55e;font-weight:600'>green</span> = low risk, "
            "<span style='color:#f59e0b;font-weight:600'>yellow</span> = moderate, "
            "<span style='color:#ef4444;font-weight:600'>red</span> = high risk. "
            "Hover over any pipe for details.",
            unsafe_allow_html=True,
        )
        deck = pipe_network_map(pipes_df, risk_scores=risk_scores)
        st.pydeck_chart(deck, use_container_width=True, height=550)

        # Quick stats below the map
        if risk_scores is not None:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Pipes", f"{len(pipes_df):,}")
            c2.metric("Avg Risk", f"{risk_scores.mean() * 100:.1f}%")
            c3.metric("Max Risk", f"{risk_scores.max() * 100:.1f}%")
            c4.metric("High Risk (>50%)", f"{(risk_scores > 0.5).sum():,}")

    with tabs[1]:
        st.markdown(
            "Each circle represents a leak event. Size and color indicate severity. "
            "The heatmap layer shows leak concentration areas.",
            unsafe_allow_html=True,
        )
        deck = leak_events_map(events_df)
        st.pydeck_chart(deck, use_container_width=True, height=550)

        # Event stats
        if events_df is not None and not events_df.empty:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Events", f"{len(events_df):,}")
            c2.metric("Critical", f"{(events_df['severity'] == 'Critical').sum():,}")
            c3.metric("Avg Flow Rate", f"{events_df['flow_rate_gpm'].mean():.1f} GPM")
            c4.metric("Avg Detection", f"{events_df['detection_hours'].mean():.0f} hrs")

    with tabs[2]:
        if risk_scores is not None:
            st.markdown(
                "3D risk columns — taller columns represent higher predicted failure probability. "
                "Drag to rotate the view.",
                unsafe_allow_html=True,
            )
            deck = risk_heatmap(pipes_df, risk_scores)
            st.pydeck_chart(deck, use_container_width=True, height=550)

            # Risk breakdown
            st.markdown('<div class="section-title">Risk Distribution</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)

            n_high = (risk_scores > 0.5).sum()
            n_med = ((risk_scores > 0.2) & (risk_scores <= 0.5)).sum()
            n_low = (risk_scores <= 0.2).sum()

            with c1:
                st.markdown(
                    f'<div class="kpi-card accent-red"><div class="kpi-value">{n_high:,}</div>'
                    f'<div class="kpi-label">High Risk Pipes (&gt;50%)</div></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<div class="kpi-card accent-amber"><div class="kpi-value">{n_med:,}</div>'
                    f'<div class="kpi-label">Medium Risk (20-50%)</div></div>',
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f'<div class="kpi-card accent-green"><div class="kpi-value">{n_low:,}</div>'
                    f'<div class="kpi-label">Low Risk (&lt;20%)</div></div>',
                    unsafe_allow_html=True,
                )

            # Top 10 riskiest pipes table
            if risk_scores is not None and len(risk_scores) > 0:
                st.markdown('<div class="section-title">Top 10 Highest Risk Pipes</div>', unsafe_allow_html=True)
                risk_df = pipes_df[["pipe_id", "material", "age", "diameter_category", "soil_type", "pressure_avg_m"]].copy()
                risk_df["Risk Score"] = (risk_scores * 100).round(1)
                risk_df = risk_df.sort_values("Risk Score", ascending=False).head(10)
                risk_df.columns = ["Pipe ID", "Material", "Age", "Diameter", "Soil", "Pressure (m)", "Risk %"]
                st.dataframe(risk_df, use_container_width=True, hide_index=True)
        else:
            st.info("Train the model to see risk predictions.")
