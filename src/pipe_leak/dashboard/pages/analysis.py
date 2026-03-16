"""Analysis page: leak patterns and root cause charts."""

import streamlit as st
import pandas as pd

from pipe_leak.dashboard.components.charts import (
    severity_distribution,
    leaks_over_time,
    leak_rate_by_material,
    leak_rate_by_age,
)


def render(events_df: pd.DataFrame, pipes_df: pd.DataFrame):
    """Render the analysis page with pattern and root cause charts."""
    st.markdown('<div class="section-header">Leak Analysis</div>', unsafe_allow_html=True)

    tabs = st.tabs(["Leak Patterns", "Root Causes"])

    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(severity_distribution(events_df), use_container_width=True)
        with col2:
            st.plotly_chart(leaks_over_time(events_df), use_container_width=True)

    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(leak_rate_by_material(events_df, pipes_df), use_container_width=True)
        with col2:
            st.plotly_chart(leak_rate_by_age(events_df, pipes_df), use_container_width=True)
