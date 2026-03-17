"""
Water Network Leak Analysis Dashboard.

Streamlit entry point — styled header, button-driven simulation & training,
reactive sidebar controls.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

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

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=DASH_CONFIG.title,
    page_icon=DASH_CONFIG.page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Gradient header
st.markdown(
    """
    <div class="dash-header">
        <h1>💧 Water Network Leak Predictor</h1>
        <p>Hydraulic simulation &bull; Weibull deterioration &bull; XGBoost risk scoring</p>
    </div>
    """,
    unsafe_allow_html=True,
)

init_state()


# ── Cached heavy functions ───────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _generate_data(seed: int, num_pipes: int, sim_years: int):
    """Generate pipe network + leak events."""
    from pipe_leak.config import SimulationConfig

    config = SimulationConfig(seed=seed, num_pipes=num_pipes, simulation_years=sim_years)
    pipes_gdf = build_pipe_network(config)
    events_df = generate_leak_events(pipes_gdf, config)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pipes_gdf.to_parquet(PROCESSED_DIR / "pipes.parquet")
    events_df.to_parquet(PROCESSED_DIR / "events.parquet")

    return pipes_gdf, events_df


@st.cache_resource(show_spinner=False)
def _train_model(_pipes_df, _events_df, horizon_days: int):
    """Train leak prediction model and return results."""
    train_df, test_df = temporal_train_test_split(
        _pipes_df, _events_df, horizon_days=horizon_days,
    )

    model = LeakClassifier()
    train_info = model.train(train_df, optimize=False)

    preds, probs = model.predict(test_df)
    y_test = test_df["target"].values
    metrics = evaluate_predictions(y_test, preds, probs)
    importance = model.get_feature_importance()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_model(model, {**train_info, **metrics})

    return model, train_df, test_df, metrics, importance, y_test, probs


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div class="sidebar-section"><h3>Simulation Parameters</h3></div>',
        unsafe_allow_html=True,
    )

    num_pipes = st.slider(
        "Number of Pipes", 200, 3000, SIM_CONFIG.num_pipes, step=100,
        help="Total pipe segments in the simulated network.",
    )
    sim_years = st.slider(
        "Simulation Years", 1, 10, SIM_CONFIG.simulation_years,
        help="How many years of leak history to generate.",
    )
    seed = st.number_input(
        "Random Seed", min_value=0, max_value=99999, value=SIM_CONFIG.seed,
        help="Change to get a different random network.",
    )

    st.markdown("---")

    # Track whether params differ from last run
    _param_key = f"{num_pipes}_{sim_years}_{seed}"
    params_changed = st.session_state.get("_last_params") != _param_key

    # ── Generate Data button ─────────────────────────────────────────────
    gen_label = "🔄 Regenerate Data" if not params_changed else "⚡ Generate Data (params changed)"
    if st.button(gen_label, type="primary", use_container_width=True):
        _generate_data.clear()
        _train_model.clear()
        for f in PROCESSED_DIR.glob("*"):
            f.unlink()
        if MODELS_DIR.exists():
            for d in MODELS_DIR.iterdir():
                if d.is_dir():
                    for f in d.iterdir():
                        f.unlink()
                    d.rmdir()
        st.session_state["_last_params"] = _param_key
        st.session_state["_force_retrain"] = True
        st.rerun()

    # ── Retrain Model button ─────────────────────────────────────────────
    if st.button("🧠 Retrain Model", use_container_width=True):
        _train_model.clear()
        if MODELS_DIR.exists():
            for d in MODELS_DIR.iterdir():
                if d.is_dir():
                    for f in d.iterdir():
                        f.unlink()
                    d.rmdir()
        st.session_state["_force_retrain"] = True
        st.rerun()


# ── Load / generate data ────────────────────────────────────────────────────

with st.spinner("Generating pipe network and simulating leaks…"):
    try:
        pipes_gdf, events_df = _generate_data(seed, num_pipes, sim_years)
        st.session_state["_last_params"] = _param_key
    except Exception as e:
        st.error(f"Data generation failed: {e}")
        st.stop()


# ── Train model ─────────────────────────────────────────────────────────────

model, metrics, importance, y_test, y_prob, risk_scores = (
    None, None, None, None, None, None,
)

with st.spinner("Training prediction model…"):
    try:
        model, train_df, test_df, metrics, importance, y_test, y_prob = _train_model(
            pipes_gdf, events_df, ML_CONFIG.prediction_horizon_days,
        )
    except ValueError as e:
        st.warning(f"Model training issue: {e}")

# Pop one-shot retrain flag
st.session_state.pop("_force_retrain", None)


# ── Predict risk scores for all pipes ───────────────────────────────────────

if model is not None and events_df is not None and not events_df.empty:
    latest_date = pd.to_datetime(events_df["date"]).max()
    pred_features = create_feature_dataset(
        pipes_gdf, events_df, latest_date, ML_CONFIG.prediction_horizon_days,
    )
    _, risk_scores = model.predict(pred_features)


# ── Sidebar filters ─────────────────────────────────────────────────────────

with st.sidebar:
    filters = render_sidebar_filters(events_df, pipes_gdf)

filtered_events = apply_event_filters(events_df, filters)


# ── Status banner ────────────────────────────────────────────────────────────

n_pipes = len(pipes_gdf)
n_events = len(events_df) if events_df is not None and not events_df.empty else 0
model_status = "trained" if model is not None else "not trained"

banner_cls = "success" if model is not None else "warning"
st.markdown(
    f'<div class="status-banner {banner_cls}">'
    f"<b>{n_pipes:,}</b> pipes &nbsp;·&nbsp; <b>{n_events:,}</b> leak events "
    f"&nbsp;·&nbsp; Model: <b>{model_status}</b>"
    f"</div>",
    unsafe_allow_html=True,
)

# ── Page tabs ────────────────────────────────────────────────────────────────

page_tabs = st.tabs([
    "📊 Overview",
    "🗺️ Maps",
    "🔬 Analysis",
    "🤖 Model Performance",
])

with page_tabs[0]:
    overview.render(pipes_gdf, filtered_events, metrics)

with page_tabs[1]:
    map_view.render(pipes_gdf, filtered_events, risk_scores)

with page_tabs[2]:
    analysis.render(filtered_events, pipes_gdf)

with page_tabs[3]:
    model_perf.render(metrics, importance, y_test, y_prob)


# ── Collapsible diagnostics ─────────────────────────────────────────────────

with st.expander("🔧 Diagnostic Information", expanded=False):
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.markdown("**Pipe Network Sample**")
        st.dataframe(
            pipes_gdf.drop(columns=["geometry"], errors="ignore").head(8),
            use_container_width=True,
            hide_index=True,
        )
    with dcol2:
        st.markdown("**Leak Events Sample**")
        if events_df is not None and not events_df.empty:
            st.dataframe(events_df.head(8), use_container_width=True, hide_index=True)
        else:
            st.info("No leak events.")

    if metrics:
        st.markdown("**Model Metrics**")
        st.json({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})
