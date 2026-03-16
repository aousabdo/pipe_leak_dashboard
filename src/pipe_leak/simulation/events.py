"""
Vectorized leak event generation.

Generates leak events for a pipe network over a multi-year simulation window.
Uses the Weibull deterioration model for failure probabilities and generates
all events via vectorized numpy operations (no Python day-by-day loops).
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta

from pipe_leak.config import SIM_CONFIG, SimulationConfig
from pipe_leak.simulation.deterioration import compute_failure_probabilities


# Severity levels and their base probabilities
SEVERITY_LEVELS = ["Minor", "Moderate", "Major", "Critical"]
SEVERITY_PROBS = [0.55, 0.28, 0.12, 0.05]

# Flow rate ranges by severity (gallons per minute)
FLOW_RATE_RANGES = {
    "Minor": (0.5, 5),
    "Moderate": (5, 30),
    "Major": (30, 120),
    "Critical": (120, 500),
}

# Detection time ranges by severity (hours)
DETECTION_HOURS_RANGES = {
    "Minor": (72, 720),
    "Moderate": (24, 168),
    "Major": (6, 48),
    "Critical": (0.5, 12),
}

# Base repair cost ranges by severity (USD)
REPAIR_COST_RANGES = {
    "Minor": (1500, 6000),
    "Moderate": (6000, 18000),
    "Major": (18000, 60000),
    "Critical": (60000, 250000),
}


def generate_leak_events(
    pipes_gdf: gpd.GeoDataFrame,
    config: SimulationConfig | None = None,
) -> pd.DataFrame:
    """
    Generate leak events for the entire simulation period using vectorized operations.

    For each year in the simulation window:
    1. Compute per-pipe annual failure probability (age increases each year)
    2. Apply seasonal modulation
    3. Sample event counts from Poisson distribution
    4. Generate event attributes (severity, flow, cost, etc.)

    Args:
        pipes_gdf: GeoDataFrame from build_pipe_network().
        config: Simulation configuration.

    Returns:
        DataFrame of leak events with columns: pipe_id, date, severity,
        flow_rate_gpm, detection_hours, water_loss_gallons, repair_cost,
        plus pipe attribute columns for convenience.
    """
    cfg = config or SIM_CONFIG
    rng = np.random.default_rng(cfg.seed + 1)  # different seed from network

    start_year = 2026 - cfg.simulation_years
    all_events = []

    for year_offset in range(cfg.simulation_years):
        current_year = start_year + year_offset
        year_start = datetime(current_year, 1, 1)

        # Update age for this year
        pipes_year = pipes_gdf.copy()
        pipes_year["age"] = current_year - pipes_year["installation_year"]

        # Annual failure probability per pipe
        annual_probs = compute_failure_probabilities(pipes_year, cfg)

        # For each month, generate events with seasonal modulation
        for month in range(1, 13):
            seasonal_mult = cfg.seasonal_factors.get(month, 1.0)
            # Monthly probability = annual / 12 * seasonal factor
            monthly_probs = annual_probs * seasonal_mult / 12.0
            monthly_probs = np.clip(monthly_probs, 0, 0.5)

            # Sample how many events each pipe has this month (Poisson)
            event_counts = rng.poisson(monthly_probs)

            # Generate events for pipes that have at least one
            for pipe_idx in np.where(event_counts > 0)[0]:
                n_events = event_counts[pipe_idx]
                pipe = pipes_year.iloc[pipe_idx]

                for _ in range(n_events):
                    # Random day within the month
                    if month == 12:
                        days_in_month = 31
                    elif month in [4, 6, 9, 11]:
                        days_in_month = 30
                    elif month == 2:
                        days_in_month = 28
                    else:
                        days_in_month = 31
                    day = rng.integers(1, days_in_month + 1)
                    try:
                        event_date = datetime(current_year, month, day)
                    except ValueError:
                        event_date = datetime(current_year, month, days_in_month)

                    event = _generate_single_event(pipe, event_date, rng)
                    all_events.append(event)

    if not all_events:
        return pd.DataFrame()

    events_df = pd.DataFrame(all_events)
    events_df = events_df.sort_values("date").reset_index(drop=True)
    events_df["event_id"] = [f"E-{i:05d}" for i in range(len(events_df))]

    print(f"Generated {len(events_df)} leak events over {cfg.simulation_years} years.")
    return events_df


def _generate_single_event(
    pipe: pd.Series, event_date: datetime, rng: np.random.Generator
) -> dict:
    """Generate attributes for a single leak event."""
    # Severity
    severity = rng.choice(SEVERITY_LEVELS, p=SEVERITY_PROBS)

    # Flow rate (GPM) based on severity and diameter
    fmin, fmax = FLOW_RATE_RANGES[severity]
    base_flow = rng.uniform(fmin, fmax)
    # Scale by diameter
    diam_m = pipe.get("diameter_m", 0.15)
    diam_multiplier = max(0.5, diam_m / 0.15)  # normalized to 6-inch pipe
    flow_rate = base_flow * min(diam_multiplier, 3.0)

    # Detection time (hours) based on severity and inspection recency
    dmin, dmax = DETECTION_HOURS_RANGES[severity]
    detection_hours = rng.uniform(dmin, dmax)
    last_insp = pipe.get("last_inspection_days", 365)
    if last_insp < 30:
        detection_hours *= 0.6
    elif last_insp > 365:
        detection_hours *= 1.4

    # Water loss
    water_loss_gallons = flow_rate * 60 * detection_hours

    # Repair cost based on severity, diameter, and depth
    cmin, cmax = REPAIR_COST_RANGES[severity]
    base_cost = rng.uniform(cmin, cmax)
    depth_ft = pipe.get("depth_ft", 4.0)
    depth_factor = 1.0 + (depth_ft - 4.0) / 6.0
    size_factor = max(1.0, diam_m / 0.15)
    repair_cost = base_cost * depth_factor * min(size_factor, 2.5)

    return {
        "pipe_id": pipe["pipe_id"],
        "date": event_date,
        "latitude": pipe.get("mid_lat", 0),
        "longitude": pipe.get("mid_lon", 0),
        "severity": severity,
        "flow_rate_gpm": round(flow_rate, 2),
        "detection_hours": round(detection_hours, 1),
        "water_loss_gallons": round(water_loss_gallons, 0),
        "repair_cost": round(repair_cost, 2),
        # Pipe attributes carried through for convenience
        "material": pipe.get("material", "Unknown"),
        "diameter_category": pipe.get("diameter_category", "Unknown"),
        "diameter_m": pipe.get("diameter_m", 0.15),
        "installation_year": pipe.get("installation_year", 2000),
        "age_at_event": 2026 - pipe.get("installation_year", 2000),
        "soil_type": pipe.get("soil_type", "Unknown"),
        "pressure_avg_m": pipe.get("pressure_avg_m", 30.0),
        "depth_ft": pipe.get("depth_ft", 4.0),
        "prev_repairs": pipe.get("prev_repairs", 0),
        "traffic_load": pipe.get("traffic_load", 5),
    }
