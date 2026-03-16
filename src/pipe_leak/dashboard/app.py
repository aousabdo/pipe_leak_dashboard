"""
Water Network Leak Analysis Dashboard.

Streamlit entry point that orchestrates data loading, model training,
and page rendering.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# Ensure src/ is on the path when run directly
_src = str(Path(__file__).resolve().parent.parent.parent)
if _src not in sys.path:
    sys.path.insert(0, _src)

from pipe_leak.config import SIM_CONFIG, ML_CONFIG, DASH_CONFIG, PROCESSED_DIR, MODELS_DIR
from pipe_leak.simulation.network import build_pipe_network
from pipe_leak.simulation.events import generate_leak_events
from pipe_leak.ml.features import create_feature_dataset, get_feature_columns
from pipe_leak.ml.splits import temporal_train_test_split
from pipe_leak.ml.classifiers import LeakClassifier
from pipe_leak.ml.evaluate import evaluate_predictions
from pipe_leak.ml.registry import save_model, load_latest_model
from pipe_leak.dashboard.styles import CUSTOM_CSS
from pipe_leak.dashboard.state import init_state
from pipe_leak.dashboard.components.filters import render_sidebar_filters, apply_event_filters
from pipe_leak.dashboard.pages import overview, map_view, analysis, model_perf

# Page config
st.set_page_config(
    page_title=DASH_CONFIG.title,
    page_icon=DASH_CONFIG.page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(
    f'<div class="main-header">{DASH_CONFIG.title}</div>',
    unsafe_allow_html=True,
)

init_state()


# --- Data loading / generation ---


@st.cache_resource(show_spinner="Generating pipe network and simulating leaks...")
def load_or_generate_data(seed: int, num_pipes: int, sim_years: int):
    """Generate or load simulation data."""
    from pipe_leak.config import SimulationConfig

    config = SimulationConfig(seed=seed, num_pipes=num_pipes, simulation_years=sim_years)

    # Check for cached parquet files
    pipes_path = PROCESSED_DIR / "pipes.parquet"
    events_path = PROCESSED_DIR / "events.parquet"

    if pipes_path.exists() and events_path.exists():
        import geopandas as gpd

        pipes_gdf = gpd.read_parquet(pipes_path)
        events_df = pd.read_parquet(events_path)
        return pipes_gdf, events_df

    # Generate fresh data
    pipes_gdf = build_pipe_network(config)
    events_df = generate_leak_events(pipes_gdf, config)

    # Cache to disk
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pipes_gdf.to_parquet(pipes_path)
    events_df.to_parquet(events_path)

    return pipes_gdf, events_df


@st.cache_resource(show_spinner="Training prediction model...")
def train_model(_pipes_df, _events_df, horizon_days: int):
    """Train or load the leak prediction model."""
    # Try loading existing model
    cached_model = load_latest_model()
    if cached_model is not None:
        # Re-create test data for metrics
        train_df, test_df = temporal_train_test_split(
            _pipes_df, _events_df, horizon_days=horizon_days
        )
        feature_cols = get_feature_columns(test_df)
        X_test = test_df[feature_cols].values
        y_test = test_df["target"].values
        preds, probs = cached_model.predict(test_df)
        metrics = evaluate_predictions(y_test, preds, probs)
        importance = cached_model.get_feature_importance()
        return cached_model, train_df, test_df, metrics, importance, y_test, probs

    # Train new model
    train_df, test_df = temporal_train_test_split(
        _pipes_df, _events_df, horizon_days=horizon_days
    )

    model = LeakClassifier()
    train_info = model.train(train_df, optimize=False)

    # Evaluate on test set
    preds, probs = model.predict(test_df)
    y_test = test_df["target"].values
    metrics = evaluate_predictions(y_test, preds, probs)
    importance = model.get_feature_importance()

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_model(model, {**train_info, **metrics})

    return model, train_df, test_df, metrics, importance, y_test, probs


# --- Sidebar controls ---

st.sidebar.header("Simulation Controls")
regenerate = st.sidebar.checkbox("Regenerate Data", value=False)
num_pipes = st.sidebar.slider("Number of Pipes", 200, 3000, 1000, step=100)
sim_years = st.sidebar.slider("Simulation Years", 1, 10, 5)
retrain = st.sidebar.checkbox("Retrain Model", value=False)

# If regenerate requested, clear caches
if regenerate:
    load_or_generate_data.clear()
    train_model.clear()
    # Remove cached files
    for f in PROCESSED_DIR.glob("*"):
        f.unlink()

if retrain:
    train_model.clear()
    # Remove cached models
    if MODELS_DIR.exists():
        for d in MODELS_DIR.iterdir():
            if d.is_dir():
                for f in d.iterdir():
                    f.unlink()
                d.rmdir()

# --- Load data ---

try:
    pipes_gdf, events_df = load_or_generate_data(
        SIM_CONFIG.seed, num_pipes, sim_years
    )
except Exception as e:
    st.error(f"Error generating data: {e}")
    st.stop()

# --- Train model ---

try:
    model, train_df, test_df, metrics, importance, y_test, y_prob = train_model(
        pipes_gdf, events_df, ML_CONFIG.prediction_horizon_days
    )
except ValueError as e:
    st.warning(f"Model training issue: {e}")
    model, metrics, importance, y_test, y_prob = None, None, None, None, None

# --- Predict risk for all pipes (latest cutoff) ---

risk_scores = None
if model is not None and events_df is not None and not events_df.empty:
    latest_date = pd.to_datetime(events_df["date"]).max()
    from pipe_leak.ml.features import create_feature_dataset

    pred_features = create_feature_dataset(
        pipes_gdf, events_df, latest_date, ML_CONFIG.prediction_horizon_days
    )
    _, risk_scores = model.predict(pred_features)

# --- Filters ---

filters = render_sidebar_filters(events_df, pipes_gdf)
filtered_events = apply_event_filters(events_df, filters)

# --- Page rendering ---

page_tabs = st.tabs(["Overview", "Maps", "Analysis", "Model Performance"])

with page_tabs[0]:
    overview.render(pipes_gdf, filtered_events, metrics)

with page_tabs[1]:
    map_view.render(pipes_gdf, filtered_events, risk_scores)

with page_tabs[2]:
    analysis.render(filtered_events, pipes_gdf)

with page_tabs[3]:
    model_perf.render(metrics, importance, y_test, y_prob)

# --- Diagnostic expander ---

with st.expander("Diagnostic Information"):
    st.markdown("### Pipe Network Sample")
    st.dataframe(pipes_gdf.drop(columns=["geometry"], errors="ignore").head(10))

    st.markdown("### Leak Events Sample")
    if events_df is not None and not events_df.empty:
        st.dataframe(events_df.head(10))
    else:
        st.info("No leak events.")

    if metrics:
        st.markdown("### Model Metrics")
        st.json(metrics)
