"""Session state management for the dashboard."""

import streamlit as st
import pandas as pd
import geopandas as gpd
from datetime import datetime


def init_state():
    """Initialize session state with defaults."""
    defaults = {
        "pipes_gdf": None,
        "events_df": None,
        "train_features": None,
        "test_features": None,
        "model": None,
        "predictions": None,
        "probabilities": None,
        "metrics": None,
        "data_loaded": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def set_data(pipes_gdf: gpd.GeoDataFrame, events_df: pd.DataFrame):
    """Store simulation data in session state."""
    st.session_state.pipes_gdf = pipes_gdf
    st.session_state.events_df = events_df
    st.session_state.data_loaded = True


def get_pipes() -> gpd.GeoDataFrame | None:
    return st.session_state.get("pipes_gdf")


def get_events() -> pd.DataFrame | None:
    return st.session_state.get("events_df")
