"""
Pipe deterioration models based on Weibull survival analysis.

Uses physically motivated Weibull hazard functions where failure rate
increases with pipe age, modified by material, soil corrosivity, pressure,
and maintenance history. Parameters are informed by water main break
rate literature (Kleiner & Rajani 2001).
"""

import numpy as np
import pandas as pd

from pipe_leak.config import SIM_CONFIG, SimulationConfig


def weibull_annual_failure_rate(
    age: np.ndarray,
    beta: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    """
    Compute annual failure rate from Weibull hazard function.

    h(t) = (beta/eta) * (t/eta)^(beta-1)

    Args:
        age: Pipe age in years.
        beta: Weibull shape parameter (>1 means increasing hazard).
        eta: Weibull scale parameter (characteristic life in years).

    Returns:
        Annual failure probability for each pipe.
    """
    age = np.maximum(age, 1.0)  # avoid zero-age edge case
    hazard_rate = (beta / eta) * (age / eta) ** (beta - 1)
    # Convert hazard rate to annual probability: P = 1 - exp(-h)
    annual_prob = 1.0 - np.exp(-hazard_rate)
    return np.clip(annual_prob, 0.0, 0.99)


def compute_failure_probabilities(
    pipes_df: pd.DataFrame,
    config: SimulationConfig | None = None,
) -> np.ndarray:
    """
    Compute per-pipe annual failure probability using Weibull model with covariates.

    The base Weibull rate is modified by:
    - Soil corrosivity (higher corrosivity -> faster deterioration)
    - Pressure ratio (higher pressure -> more stress)
    - Previous repairs (more repairs -> weaker pipe)
    - Inspection recency (recent inspection -> lower effective risk via detection)

    Args:
        pipes_df: DataFrame with columns: age, material, soil_corrosivity,
                  pressure_avg_m, prev_repairs, last_inspection_days, diameter_m.
        config: Simulation configuration.

    Returns:
        Array of annual failure probabilities, one per pipe.
    """
    cfg = config or SIM_CONFIG
    n = len(pipes_df)

    # Get Weibull parameters per pipe based on material
    beta = np.zeros(n)
    eta = np.zeros(n)
    for i, mat in enumerate(pipes_df["material"]):
        params = cfg.weibull_params.get(mat, {"beta": 2.0, "eta": 70})
        beta[i] = params["beta"]
        eta[i] = params["eta"]

    # Base failure probability from Weibull
    age = pipes_df["age"].values.astype(float)
    base_prob = weibull_annual_failure_rate(age, beta, eta)

    # Covariate modifiers (multiplicative on the hazard)
    # Soil corrosivity: normalize to [0.5, 2.0] range
    soil_corr = pipes_df["soil_corrosivity"].values.astype(float)
    soil_modifier = 0.5 + (soil_corr / 10.0) * 1.5  # corrosivity 0->0.5x, 10->2.0x

    # Pressure: higher pressure increases stress
    # Normalize around typical pressure ~30m head
    pressure = pipes_df["pressure_avg_m"].values.astype(float)
    pressure_modifier = np.clip(pressure / 30.0, 0.5, 2.0)

    # Previous repairs: each repair adds 10% to failure rate
    prev_repairs = pipes_df["prev_repairs"].values.astype(float)
    repair_modifier = 1.0 + prev_repairs * 0.10

    # Smaller diameter pipes fail more often (higher stress per unit area)
    diameter = pipes_df["diameter_m"].values.astype(float)
    diameter_modifier = np.where(diameter < 0.15, 1.3, np.where(diameter > 0.35, 0.7, 1.0))

    # Combined probability
    modified_prob = base_prob * soil_modifier * pressure_modifier * repair_modifier * diameter_modifier

    return np.clip(modified_prob, 0.001, 0.95)
