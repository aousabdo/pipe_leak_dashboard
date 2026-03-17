"""Sidebar filter widgets — clean, grouped sections."""

import streamlit as st
import pandas as pd


def render_sidebar_filters(
    events_df: pd.DataFrame, pipes_df: pd.DataFrame
) -> dict:
    """Render sidebar filters and return filter selections."""
    filters = {}

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data Filters")

    if events_df is not None and not events_df.empty:
        events_df = events_df.copy()
        events_df["date"] = pd.to_datetime(events_df["date"])

        # Date range
        min_date = events_df["date"].min().date()
        max_date = events_df["date"].max().date()
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        filters["date_range"] = date_range if len(date_range) == 2 else (min_date, max_date)

        # Severity
        severities = ["Minor", "Moderate", "Major", "Critical"]
        available = [s for s in severities if s in events_df["severity"].unique()]
        filters["severities"] = st.sidebar.multiselect(
            "Severity Level",
            options=available,
            default=available,
        )

        # Material
        materials = sorted(events_df["material"].unique().tolist())
        filters["materials"] = st.sidebar.multiselect(
            "Pipe Material",
            options=materials,
            default=materials,
        )
    else:
        st.sidebar.caption("No leak events available for filtering.")
        filters["date_range"] = None
        filters["severities"] = []
        filters["materials"] = []

    # Age range
    if pipes_df is not None and "age" in pipes_df.columns:
        age_min = int(pipes_df["age"].min())
        age_max = int(pipes_df["age"].max())
        if age_min < age_max:
            filters["age_range"] = st.sidebar.slider(
                "Pipe Age (years)",
                min_value=age_min,
                max_value=age_max,
                value=(age_min, age_max),
            )

    return filters


def apply_event_filters(events_df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply filters to the events DataFrame."""
    if events_df is None or events_df.empty:
        return events_df

    df = events_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if filters.get("date_range"):
        start, end = filters["date_range"]
        df = df[(df["date"].dt.date >= start) & (df["date"].dt.date <= end)]

    if filters.get("severities"):
        df = df[df["severity"].isin(filters["severities"])]

    if filters.get("materials"):
        df = df[df["material"].isin(filters["materials"])]

    return df
