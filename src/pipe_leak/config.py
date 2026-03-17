"""Central configuration for the pipe leak dashboard."""

from dataclasses import dataclass, field
from pathlib import Path

# Project root (two levels up from this file: src/pipe_leak/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
NETWORKS_DIR = DATA_DIR / "networks"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass
class SimulationConfig:
    """Configuration for pipe network simulation."""

    seed: int = 42
    num_pipes: int = 1000
    simulation_years: int = 5
    # Network placement center (default: Sacramento, CA)
    center_lat: float = 38.5816
    center_lon: float = -121.4944
    # Weibull deterioration parameters by material (shape beta, scale eta in years)
    # Tuned so 10-25% of pipes experience a leak per year for aging infrastructure
    weibull_params: dict = field(
        default_factory=lambda: {
            "Cast Iron": {"beta": 2.5, "eta": 28},
            "Ductile Iron": {"beta": 2.2, "eta": 40},
            "Steel": {"beta": 2.0, "eta": 34},
            "Asbestos Cement": {"beta": 2.4, "eta": 26},
            "PVC": {"beta": 1.8, "eta": 55},
            "HDPE": {"beta": 1.6, "eta": 70},
        }
    )
    # Material assignment by installation decade
    material_by_era: dict = field(
        default_factory=lambda: {
            (1920, 1960): ["Cast Iron", "Asbestos Cement", "Steel"],
            (1960, 1985): ["Ductile Iron", "Steel", "Asbestos Cement"],
            (1985, 2005): ["PVC", "Ductile Iron"],
            (2005, 2030): ["HDPE", "PVC"],
        }
    )
    # Soil zones and their corrosivity index (1-10 scale)
    soil_zones: dict = field(
        default_factory=lambda: {
            "Clay": {"corrosivity": 8, "weight": 0.25},
            "Sandy": {"corrosivity": 4, "weight": 0.20},
            "Loam": {"corrosivity": 3, "weight": 0.20},
            "Rocky": {"corrosivity": 6, "weight": 0.15},
            "Silt": {"corrosivity": 5, "weight": 0.20},
        }
    )
    # Seasonal temperature factors (month -> multiplier on failure rate)
    seasonal_factors: dict = field(
        default_factory=lambda: {
            1: 1.35, 2: 1.25, 3: 1.05, 4: 0.90,
            5: 0.80, 6: 0.75, 7: 0.80, 8: 0.90,
            9: 1.00, 10: 1.10, 11: 1.25, 12: 1.40,
        }
    )


@dataclass
class MLConfig:
    """Configuration for ML pipeline."""

    prediction_horizon_days: int = 365
    test_fraction: float = 0.2  # last 20% of time for testing
    n_cv_folds: int = 5
    random_state: int = 42
    # XGBoost defaults (sensible, not exhaustive grid)
    xgb_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "eval_metric": "logloss",
        }
    )


@dataclass
class DashboardConfig:
    """Configuration for the Streamlit dashboard."""

    title: str = "Water Network Leak Analysis Dashboard"
    map_style: str = "light"
    default_zoom: int = 12
    page_icon: str = "💧"


# Default instances
SIM_CONFIG = SimulationConfig()
ML_CONFIG = MLConfig()
DASH_CONFIG = DashboardConfig()
