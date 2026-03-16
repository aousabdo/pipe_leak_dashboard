"""Shared test fixtures."""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from datetime import datetime, timedelta

from pipe_leak.config import SimulationConfig


@pytest.fixture
def small_config():
    """Small simulation config for fast tests."""
    return SimulationConfig(seed=42, num_pipes=50, simulation_years=2)


@pytest.fixture
def sample_pipes():
    """Small pipe network GeoDataFrame for testing."""
    rng = np.random.default_rng(42)
    n = 30
    records = []
    for i in range(n):
        lat1 = 38.58 + rng.uniform(-0.01, 0.01)
        lon1 = -121.49 + rng.uniform(-0.01, 0.01)
        lat2 = lat1 + rng.uniform(-0.002, 0.002)
        lon2 = lon1 + rng.uniform(-0.002, 0.002)
        geom = LineString([(lon1, lat1), (lon2, lat2)])
        material = rng.choice(["Cast Iron", "PVC", "Ductile Iron", "HDPE"])
        install_year = rng.integers(1950, 2020)
        records.append({
            "pipe_id": f"P-{i:04d}",
            "start_node": f"J-{i}-0",
            "end_node": f"J-{i}-1",
            "length_m": rng.uniform(100, 300),
            "diameter_m": rng.choice([0.1, 0.15, 0.2, 0.3, 0.5]),
            "roughness": rng.uniform(80, 130),
            "mid_lat": (lat1 + lat2) / 2,
            "mid_lon": (lon1 + lon2) / 2,
            "pressure_avg_m": rng.uniform(15, 50),
            "velocity_avg_ms": rng.uniform(0.1, 2.0),
            "flowrate_avg_lps": rng.uniform(0.5, 5.0),
            "geometry": geom,
            "zone": i % 5,
            "installation_year": install_year,
            "material": material,
            "age": 2026 - install_year,
            "soil_type": rng.choice(["Clay", "Sandy", "Loam"]),
            "soil_corrosivity": rng.uniform(2, 9),
            "diameter_category": "small (2-6)" if 0.1 <= rng.random() < 0.5 else "medium (8-12)",
            "depth_ft": rng.uniform(3, 7),
            "prev_repairs": rng.integers(0, 5),
            "last_inspection_days": rng.integers(0, 1000),
            "traffic_load": rng.integers(1, 10),
        })

    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


@pytest.fixture
def sample_events(sample_pipes):
    """Sample leak events DataFrame."""
    rng = np.random.default_rng(42)
    events = []
    start = datetime(2022, 1, 1)
    # Generate ~50 events over 2 years
    for _ in range(50):
        pipe = sample_pipes.iloc[rng.integers(0, len(sample_pipes))]
        event_date = start + timedelta(days=int(rng.integers(0, 730)))
        events.append({
            "pipe_id": pipe["pipe_id"],
            "date": event_date,
            "latitude": pipe["mid_lat"],
            "longitude": pipe["mid_lon"],
            "severity": rng.choice(["Minor", "Moderate", "Major", "Critical"]),
            "flow_rate_gpm": round(rng.uniform(0.5, 100), 2),
            "detection_hours": round(rng.uniform(1, 500), 1),
            "water_loss_gallons": round(rng.uniform(100, 100000), 0),
            "repair_cost": round(rng.uniform(1000, 50000), 2),
            "material": pipe["material"],
            "diameter_category": pipe["diameter_category"],
            "diameter_m": pipe["diameter_m"],
            "installation_year": pipe["installation_year"],
            "age_at_event": 2026 - pipe["installation_year"],
            "soil_type": pipe["soil_type"],
            "pressure_avg_m": pipe["pressure_avg_m"],
            "depth_ft": pipe["depth_ft"],
            "prev_repairs": pipe["prev_repairs"],
            "traffic_load": pipe["traffic_load"],
        })

    return pd.DataFrame(events).sort_values("date").reset_index(drop=True)
