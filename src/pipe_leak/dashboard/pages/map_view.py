"""Map view page: interactive pydeck maps for network, events, and risk."""

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
    """Render the map view page with tabbed maps."""
    st.markdown('<div class="section-header">Network Maps</div>', unsafe_allow_html=True)

    tabs = st.tabs(["Pipe Network", "Historical Leaks", "Risk Prediction"])

    with tabs[0]:
        st.markdown("Pipe segments colored by predicted risk (green=low, red=high).")
        deck = pipe_network_map(pipes_df, risk_scores=risk_scores)
        st.pydeck_chart(deck, use_container_width=True)

    with tabs[1]:
        st.markdown("Historical leak locations sized and colored by severity.")
        deck = leak_events_map(events_df)
        st.pydeck_chart(deck, use_container_width=True)

    with tabs[2]:
        if risk_scores is not None:
            st.markdown("Predicted leak risk heatmap for the next 90 days.")
            deck = risk_heatmap(pipes_df, risk_scores)
            st.pydeck_chart(deck, use_container_width=True)

            # Summary stats
            n_high = (risk_scores > 0.5).sum()
            n_med = ((risk_scores > 0.2) & (risk_scores <= 0.5)).sum()
            n_low = (risk_scores <= 0.2).sum()
            col1, col2, col3 = st.columns(3)
            col1.metric("High Risk Pipes", f"{n_high:,}")
            col2.metric("Medium Risk Pipes", f"{n_med:,}")
            col3.metric("Low Risk Pipes", f"{n_low:,}")
        else:
            st.info("Train the model to see risk predictions.")
