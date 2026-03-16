"""Analysis page: leak patterns, root causes, and deep-dive charts."""

import streamlit as st
import pandas as pd
import numpy as np

from pipe_leak.dashboard.components.charts import (
    severity_distribution,
    leaks_over_time,
    leak_rate_by_material,
    leak_rate_by_age,
    cost_by_severity_chart,
    water_loss_timeline,
)


def render(events_df: pd.DataFrame, pipes_df: pd.DataFrame):
    """Render the analysis page."""
    tabs = st.tabs(["📈 Leak Patterns", "🔍 Root Causes", "💰 Cost Analysis"])

    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(severity_distribution(events_df), use_container_width=True)
        with col2:
            st.plotly_chart(leaks_over_time(events_df), use_container_width=True)

        # Summary insights
        if events_df is not None and not events_df.empty:
            st.markdown('<div class="section-title">Key Insights</div>', unsafe_allow_html=True)
            events = events_df.copy()
            events["date"] = pd.to_datetime(events["date"])
            events["month"] = events["date"].dt.month
            events["year"] = events["date"].dt.year

            c1, c2, c3 = st.columns(3)

            # Peak month
            month_counts = events["month"].value_counts()
            peak_month = month_counts.idxmax()
            month_names = {
                1: "January", 2: "February", 3: "March", 4: "April",
                5: "May", 6: "June", 7: "July", 8: "August",
                9: "September", 10: "October", 11: "November", 12: "December",
            }
            c1.metric("Peak Month", month_names.get(peak_month, ""), f"{month_counts.max()} events")

            # Most affected pipe
            top_pipe = events["pipe_id"].value_counts().head(1)
            c2.metric("Most Affected Pipe", top_pipe.index[0], f"{top_pipe.values[0]} events")

            # Worst year
            year_counts = events["year"].value_counts()
            worst_year = year_counts.idxmax()
            c3.metric("Worst Year", str(worst_year), f"{year_counts.max()} events")

    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(leak_rate_by_material(events_df, pipes_df), use_container_width=True)
        with col2:
            st.plotly_chart(leak_rate_by_age(events_df, pipes_df), use_container_width=True)

        # Material breakdown table
        if events_df is not None and not events_df.empty:
            st.markdown('<div class="section-title">Material Risk Profile</div>', unsafe_allow_html=True)

            leak_counts = events_df.groupby("material").agg(
                Events=("pipe_id", "size"),
                Unique_Pipes=("pipe_id", "nunique"),
                Avg_Cost=("repair_cost", "mean"),
                Total_Cost=("repair_cost", "sum"),
                Avg_Flow=("flow_rate_gpm", "mean"),
            ).round(0)

            pipe_counts = pipes_df["material"].value_counts().rename("Total_Pipes")
            summary = leak_counts.join(pipe_counts)
            summary["Leak_Rate_%"] = (summary["Unique_Pipes"] / summary["Total_Pipes"] * 100).round(1)
            summary = summary[["Total_Pipes", "Events", "Unique_Pipes", "Leak_Rate_%", "Avg_Cost", "Total_Cost"]].sort_values("Leak_Rate_%", ascending=False)
            summary.columns = ["Total Pipes", "Events", "Pipes Affected", "Leak Rate %", "Avg Cost ($)", "Total Cost ($)"]

            st.dataframe(
                summary,
                use_container_width=True,
                column_config={
                    "Avg Cost ($)": st.column_config.NumberColumn(format="$%.0f"),
                    "Total Cost ($)": st.column_config.NumberColumn(format="$%.0f"),
                },
            )

    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(cost_by_severity_chart(events_df), use_container_width=True)
        with col2:
            st.plotly_chart(water_loss_timeline(events_df), use_container_width=True)

        # Cost summary
        if events_df is not None and not events_df.empty:
            st.markdown('<div class="section-title">Cost Summary</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Cost", f"${events_df['repair_cost'].sum():,.0f}")
            c2.metric("Average Cost", f"${events_df['repair_cost'].mean():,.0f}")
            c3.metric("Median Cost", f"${events_df['repair_cost'].median():,.0f}")
            c4.metric("Max Single Event", f"${events_df['repair_cost'].max():,.0f}")
