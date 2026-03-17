# Water Network Leak Analysis Dashboard

A realistic water distribution pipe leak simulation, prediction, and analysis platform. Built with WNTR for hydraulic modeling, Weibull deterioration models, XGBoost ML, and a modern React + FastAPI stack.

## What This Does

1. **Simulates** a water distribution network with realistic hydraulic pressures from EPANET
2. **Generates** leak events using a physics-based Weibull deterioration model per pipe material
3. **Predicts** which pipes are most likely to fail using XGBoost with temporal-aware features
4. **Visualizes** the network, leak history, risk heatmaps, and model performance in a modern React dashboard

## Quick Start

```bash
# Clone and install
git clone <repo-url>
cd pipe_leak_dashboard

# Backend (Python 3.10+ required)
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Frontend (Node 18+ required)
cd frontend && npm install --legacy-peer-deps && cd ..

# Start both (in separate terminals):
make backend    # FastAPI on http://localhost:8000
make frontend   # React dev server on http://localhost:5173
```

Open **http://localhost:5173** — click "Generate Network" in the sidebar, then "Train Model".

## Project Structure

```
pipe_leak_dashboard/
├── src/pipe_leak/
│   ├── config.py                     # Central configuration dataclasses
│   ├── api/
│   │   ├── main.py                   # FastAPI app with REST endpoints
│   │   └── state.py                  # Server-side state (data + model in memory)
│   ├── simulation/
│   │   ├── network.py                # WNTR grid network + hydraulic simulation
│   │   ├── deterioration.py          # Weibull failure probability model
│   │   └── events.py                 # Vectorized leak event generation
│   ├── ml/
│   │   ├── features.py               # Temporal feature engineering (no leakage)
│   │   ├── splits.py                 # Temporal train/test split + expanding-window CV
│   │   ├── classifiers.py            # XGBoost with scale_pos_weight
│   │   ├── evaluate.py               # ROC-AUC, PR-AUC, Brier, calibration curves
│   │   └── registry.py               # Model save/load with metadata
│   ├── data/schemas.py               # Pandera validation schemas
│   └── utils/                        # Geometry and I/O helpers
├── frontend/                         # React + Vite + Tailwind
│   ├── src/
│   │   ├── App.tsx                   # Root: state, params, rerun logic
│   │   ├── api.ts                    # API client (fetch wrapper)
│   │   ├── components/
│   │   │   ├── Header.tsx            # Gradient header with status pills
│   │   │   ├── Sidebar.tsx           # Parameter sliders + action buttons
│   │   │   ├── TabBar.tsx            # Page navigation tabs
│   │   │   ├── RerunBanner.tsx       # "Params changed" warning + rerun button
│   │   │   ├── KpiCard.tsx           # Metric card with accent border
│   │   │   ├── ChartCard.tsx         # Chart wrapper card
│   │   │   └── LoadingOverlay.tsx    # Full-screen loading spinner
│   │   └── pages/
│   │       ├── OverviewPage.tsx      # KPIs, trend, severity, material charts
│   │       ├── MapPage.tsx           # deck.gl network map with risk colors
│   │       ├── AnalysisPage.tsx      # Patterns, root causes, cost analysis
│   │       └── ModelPage.tsx         # Metrics, ROC/PR curves, confusion matrix
│   ├── index.html
│   ├── vite.config.ts
│   └── package.json
├── scripts/                          # CLI scripts for simulation and training
├── tests/                            # pytest test suite
├── pyproject.toml                    # Python dependencies and build config
└── Makefile                          # Common commands
```

## Architecture

### Backend (FastAPI)

The Python backend exposes these endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Current state (has data? has model?) |
| `/api/simulate` | POST | Generate pipe network + leak events |
| `/api/train` | POST | Train XGBoost model |
| `/api/overview` | GET | KPIs, trends, distributions |
| `/api/pipes` | GET | Pipe data with risk scores for map |
| `/api/events` | POST | Filtered leak events |
| `/api/analysis` | GET | Material, soil, cost analysis |
| `/api/model` | GET | Metrics, ROC/PR curves, feature importance |
| `/api/filters` | GET | Available filter options |

### Frontend (React + Vite + Tailwind)

- **Sidebar**: Sliders for num_pipes, sim_years, seed. Buttons for "Generate Network" and "Train Model"
- **Rerun Banner**: Appears when parameters change — prominent amber banner with "Rerun Simulation" button
- **Overview**: 8 KPI cards, area chart trend, severity/cost donuts, material bars, age histogram
- **Map**: deck.gl with dark basemap, green→yellow→red risk coloring, tooltips, risk hotspot mode
- **Analysis**: 3-tab layout (Patterns, Root Causes, Costs), stacked bar charts, material risk table
- **Model**: 7 metric badges, filled ROC/PR curves, confusion matrix, feature importance, calibration plot

## How It Works

### Simulation

- **WNTR network**: Grid of junctions with reservoir and tank. EPANET hydraulic simulation for realistic pressures.
- **Spatial correlation**: Zones share installation era, material, and soil type.
- **Weibull deterioration**: `h(t) = (β/η)(t/η)^(β-1)` with material-specific β and η.
- **Event generation**: Poisson sampling per pipe per month with seasonal modulation.

### Machine Learning

- **No data leakage**: Features from pre-cutoff data only; target from post-cutoff window only.
- **Temporal splitting**: Train on earlier years, test on later years.
- **XGBoost** with `scale_pos_weight` for class imbalance (no SMOTE).
- **Honest metrics**: No artificial caps or dummy models.

## Configuration

All defaults in `src/pipe_leak/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_pipes` | 1000 | Pipe segments in the network |
| `simulation_years` | 5 | Years of leak history |
| `center_lat/lon` | Sacramento, CA | Network placement |
| `prediction_horizon_days` | 90 | Prediction window |
| `weibull_params` | Per-material | β and η for each material |

Adjustable from the sidebar at runtime.

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make install` | Install Python package with dev dependencies |
| `make install-frontend` | Install frontend npm packages |
| `make backend` | Start FastAPI server (port 8000) |
| `make frontend` | Start Vite dev server (port 5173) |
| `make simulate` | Generate data via CLI |
| `make train` | Train model via CLI |
| `make test` | Run pytest suite |
| `make lint` | Check code style with ruff |
| `make clean` | Remove generated data and models |

## Running Tests

```bash
make test
# or: pytest tests/ -v
```

Tests cover deterioration model, event generation, feature leakage prevention, and temporal splitting.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Simulation | WNTR + EPANET |
| ML | XGBoost, scikit-learn |
| Backend | FastAPI, uvicorn |
| Frontend | React 19, Vite, Tailwind CSS v4 |
| Charts | Recharts |
| Maps | deck.gl + MapLibre GL |
| Icons | Lucide React |
