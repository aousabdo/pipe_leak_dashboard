# Water Network Leak Analysis Dashboard

Simulates a realistic water distribution pipe network using WNTR (Water Network Tool for Resilience), generates leak events via Weibull deterioration models, predicts future leaks with XGBoost, and visualizes everything in an interactive Streamlit dashboard.

## Quick Start

```bash
# Install (Python 3.10+ required)
pip install -e ".[dev]"

# Option 1: Run the dashboard (generates data + trains model on first launch)
make run
# or: streamlit run src/pipe_leak/dashboard/app.py

# Option 2: Run simulation and training separately via CLI
make simulate   # generates pipe network and leak events
make train      # trains the prediction model
make run        # launches the dashboard
```

## Project Structure

```
src/pipe_leak/
  simulation/     # WNTR network generation, Weibull deterioration, event simulation
  ml/             # Feature engineering (temporal-aware), XGBoost classifier, evaluation
  dashboard/      # Streamlit app with pydeck maps, Plotly charts, tabbed pages
  config.py       # Central configuration
scripts/          # CLI entry points for simulation and training
tests/            # pytest test suite
data/             # Generated data (gitignored)
models/           # Saved model artifacts (gitignored)
```

## Key Design Decisions

- **WNTR hydraulic simulation**: Pipes are linear segments on a grid network with realistic pressures from EPANET, not random dots on a map.
- **Spatially correlated attributes**: Pipe material, age, and soil type are assigned by spatial zone (neighborhoods built in the same era share characteristics).
- **Weibull deterioration model**: Failure probability uses `h(t) = (beta/eta) * (t/eta)^(beta-1)` with material-specific parameters from published literature.
- **No data leakage**: Features are computed strictly from data before the prediction cutoff; the target is defined from the future window only.
- **Temporal train/test split**: No random shuffling; train on earlier years, test on later years.
- **Honest metrics**: No artificial caps on AUC or accuracy. If the model scores 0.99, investigate the data, don't hide it.
- **pydeck maps**: GPU-accelerated interactive maps replace static Folium renders.

## Running Tests

```bash
make test
# or: pytest tests/ -v
```

## Dependencies

Core: `wntr`, `numpy`, `pandas`, `geopandas`, `shapely`, `xgboost`, `scikit-learn`, `lifelines`, `plotly`, `streamlit`, `pydeck`, `pandera`, `joblib`

Dev: `pytest`, `ruff`

See `pyproject.toml` for exact versions.

## Roadmap

- **Phase 2**: Survival analysis models (Cox PH, Random Survival Forest), real environmental data (SSURGO soil, NOAA temperature, USGS elevation)
- **Phase 3**: Interactive drill-down, scenario planning ("what if we replace all cast iron?"), cost-benefit analysis, PDF/CSV export
