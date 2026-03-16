"""Data I/O utilities."""

import pandas as pd
import geopandas as gpd
from pathlib import Path

from pipe_leak.config import PROCESSED_DIR


def save_pipes(pipes_gdf: gpd.GeoDataFrame, name: str = "pipes"):
    """Save pipe network to parquet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pipes_gdf.to_parquet(PROCESSED_DIR / f"{name}.parquet")


def save_events(events_df: pd.DataFrame, name: str = "events"):
    """Save events to parquet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    events_df.to_parquet(PROCESSED_DIR / f"{name}.parquet")


def load_pipes(name: str = "pipes") -> gpd.GeoDataFrame | None:
    """Load pipe network from parquet."""
    path = PROCESSED_DIR / f"{name}.parquet"
    if path.exists():
        return gpd.read_parquet(path)
    return None


def load_events(name: str = "events") -> pd.DataFrame | None:
    """Load events from parquet."""
    path = PROCESSED_DIR / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None
