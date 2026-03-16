# Water Network Leak Analysis Dashboard

A realistic water distribution pipe leak simulation, prediction, and analysis platform. Built with WNTR (Water Network Tool for Resilience) for hydraulic modeling, Weibull deterioration models for failure prediction, XGBoost for machine learning, and Streamlit + pydeck for interactive visualization.

## What This Does

1. **Simulates** a water distribution network as a grid of pipe segments with realistic hydraulic pressures from EPANET
2. **Generates** leak events using a physics-based Weibull deterioration model calibrated per pipe material
3. **Predicts** which pipes are most likely to fail next using an XGBoost classifier with temporal-aware features
4. **Visualizes** the network, leak history, risk heatmaps, and model performance in an interactive dashboard

## Quick Start

```bash
# Clone and install (Python 3.10+ required)
git clone <repo-url>
cd pipe_leak_dashboard
pip install -e ".[dev]"

# Option 1: Launch the dashboard (generates data + trains model automatically on first run)
make run
# or: streamlit run src/pipe_leak/dashboard/app.py

# Option 2: Run steps separately via CLI
make simulate   # generate pipe network and leak events
make train      # train the prediction model
make run        # launch the dashboard
```

## Project Structure

```
pipe_leak_dashboard/
├── src/pipe_leak/
│   ├── config.py                     # Central configuration (all defaults in one place)
│   ├── simulation/
│   │   ├── network.py                # WNTR grid network generation + hydraulic simulation
│   │   ├── deterioration.py          # Weibull failure probability model
│   │   └── events.py                 # Vectorized leak event generation
│   ├── ml/
│   │   ├── features.py               # Temporal-aware feature engineering (no leakage)
│   │   ├── splits.py                 # Temporal train/test split + expanding-window CV
│   │   ├── classifiers.py            # XGBoost classifier with scale_pos_weight
│   │   ├── evaluate.py               # Metrics: ROC-AUC, PR-AUC, Brier, calibration
│   │   └── registry.py               # Model save/load with metadata
│   ├── dashboard/
│   │   ├── app.py                    # Streamlit entry point
│   │   ├── pages/
│   │   │   ├── overview.py           # KPI cards and summary statistics
│   │   │   ├── map_view.py           # pydeck interactive maps (network, events, risk)
│   │   │   ├── analysis.py           # Leak patterns and root cause charts
│   │   │   └── model_perf.py         # Model metrics, ROC/PR curves, calibration
│   │   ├── components/
│   │   │   ├── charts.py             # Reusable Plotly chart builders
│   │   │   ├── filters.py            # Sidebar filter widgets
│   │   │   └── maps.py               # pydeck map rendering helpers
│   │   ├── state.py                  # Session state management
│   │   └── styles.py                 # CSS theming
│   ├── data/
│   │   └── schemas.py                # Pandera validation schemas
│   └── utils/
│       ├── geo.py                    # Geometry helpers (haversine, etc.)
│       └── io.py                     # Parquet data I/O utilities
├── scripts/
│   ├── run_simulation.py             # CLI: generate pipe network + leak events
│   └── train_model.py                # CLI: train and evaluate the model
├── tests/                            # pytest test suite
│   ├── conftest.py                   # Shared fixtures (sample pipes, events)
│   ├── test_simulation/              # Deterioration model, event generation
│   └── test_ml/                      # Feature leakage tests, temporal split tests
├── data/
│   ├── networks/                     # WNTR .inp network files
│   ├── processed/                    # Generated simulation data (gitignored)
│   └── external/                     # External data downloads (gitignored)
├── models/                           # Saved model artifacts (gitignored)
├── pyproject.toml                    # Dependencies, build config, tool settings
└── Makefile                          # Common commands: run, simulate, train, test, lint
```

## How It Works

### Simulation

The simulation creates a water distribution network centered on Sacramento, CA:

- **Network topology**: A grid of junctions connected by pipe segments (LineString geometries), with a reservoir and tank. Main arteries (every 5th row/col) use larger diameter pipes. All pipes get realistic operating pressures from WNTR's EPANET hydraulic simulator.
- **Spatially correlated attributes**: The network is divided into spatial zones. Pipes in the same zone share installation era, material type, and soil conditions — just like real neighborhoods built in the same decade.
- **Material assignment by era**: Pre-1960 → Cast Iron / Asbestos Cement / Steel. 1960-1985 → Ductile Iron / Steel. 1985-2005 → PVC / Ductile Iron. Post-2005 → HDPE / PVC.
- **Weibull deterioration**: Each pipe's annual failure probability is computed from `h(t) = (β/η)(t/η)^(β-1)` where β (shape) and η (scale) are material-specific. Modifiers for soil corrosivity, pressure, previous repairs, and diameter.
- **Event generation**: For each pipe and month, a Poisson-sampled event count (with seasonal modulation) determines leak occurrence. Event severity, flow rate, detection time, water loss, and repair cost are generated per event.

### Machine Learning

The prediction pipeline avoids common pitfalls:

- **No data leakage**: Given a cutoff date, features are computed only from data *before* the cutoff. The target ("did this pipe leak?") is defined only from the window *after* the cutoff.
- **Temporal splitting**: Train on earlier years, test on later years. No random shuffling that would leak future information into training.
- **Features**: Static pipe attributes (material, diameter, age, soil, pressure), historical leak metrics (count, days since last leak, avg cost — all pre-cutoff), seasonal indicators.
- **Class imbalance**: Handled via XGBoost's `scale_pos_weight` parameter, not SMOTE. Since the data is simulated, resampling synthetic data adds noise, not signal.
- **Honest evaluation**: All metrics (accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score) are reported as-is. No artificial caps. If metrics look too good, that's a signal to investigate the data.

### Dashboard

Four tabbed pages in Streamlit:

- **Overview**: KPI cards (total pipes, leak events, leak rate, costs, model AUC), pipe network summary by material.
- **Maps**: Three sub-tabs — pipe network colored by risk, historical leak events by severity, and risk prediction heatmap. All rendered with pydeck (WebGL, interactive, tooltips on hover).
- **Analysis**: Severity distribution, leaks over time, leak rate by material, leak rate by age group.
- **Model Performance**: Bar chart of metrics, top-15 feature importance, ROC curve, precision-recall curve, calibration plot, confusion matrix.

Sidebar controls: regenerate data, retrain model, filter by date range / severity / material / pipe age.

## Configuration

All defaults are centralized in `src/pipe_leak/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_pipes` | 1000 | Number of pipes in the network |
| `simulation_years` | 5 | Years of leak history to simulate |
| `center_lat/lon` | Sacramento, CA | Network placement center |
| `prediction_horizon_days` | 90 | How far ahead the model predicts |
| `weibull_params` | Per-material | β and η for each pipe material |
| `seasonal_factors` | Per-month | Monthly multipliers on failure rate |

These can be adjusted via the Streamlit sidebar or by editing `config.py` directly.

## Running Tests

```bash
make test
# or: pytest tests/ -v
```

Tests cover:
- **Deterioration model**: Failure rate increases with age, material ordering is correct, probabilities are bounded
- **Event generation**: Events have required columns, valid severities, positive costs, dates within simulation window
- **Feature engineering**: No data leakage (verified: features use only pre-cutoff data, target uses only post-cutoff data)
- **Temporal splits**: Train dates strictly before test dates, both sets have matching columns

## Dependencies

**Core:**

| Package | Purpose |
|---------|---------|
| `wntr` | Water network modeling + EPANET hydraulic simulation |
| `numpy`, `pandas`, `geopandas`, `shapely` | Data manipulation and spatial geometry |
| `xgboost`, `scikit-learn` | Machine learning classifier and evaluation |
| `lifelines` | Survival analysis (used in Phase 2) |
| `plotly` | Interactive charts |
| `streamlit`, `pydeck` | Dashboard framework + GPU-accelerated maps |
| `pandera` | Data validation schemas |
| `joblib` | Model serialization |

**Dev:** `pytest`, `pytest-cov`, `ruff`

All versions pinned in `pyproject.toml`.

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make install` | Install the package in editable mode with dev dependencies |
| `make run` | Launch the Streamlit dashboard |
| `make simulate` | Generate pipe network and leak events via CLI |
| `make train` | Train and evaluate the prediction model via CLI |
| `make test` | Run the pytest test suite |
| `make lint` | Check code style with ruff |
| `make format` | Auto-format code with ruff |
| `make clean` | Remove generated data and model files |

## Roadmap

### Phase 2: Survival Analysis + Real Data
- Cox Proportional Hazards and Random Survival Forest models (predict *time to failure*, not just yes/no)
- Real environmental data: USDA SSURGO soil corrosivity, NOAA temperature/freeze-thaw cycles, USGS elevation
- 10-year simulation for proper temporal validation
- Model comparison dashboard (classifier vs. survival models, concordance index)

### Phase 3: Interactive Dashboard + Scenario Planning
- Click-to-drill-down: select a pipe segment to see its full history and predicted survival curve
- Scenario planning: "What if we replace all Cast Iron pipes?" / "What if we double inspection frequency?"
- Cost-benefit analysis: rank pipes by (risk × cost), show ROI of proactive replacement
- Export: CSV of high-risk pipes, PDF summary reports, GeoJSON for external GIS tools
