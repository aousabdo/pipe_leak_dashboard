"""
Temporal-aware feature engineering.

All features are computed strictly from data BEFORE a cutoff date.
The target variable is defined as whether a leak occurs in the window
AFTER the cutoff. This eliminates the data leakage present in the
original implementation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def create_feature_dataset(
    pipes_df: pd.DataFrame,
    events_df: pd.DataFrame,
    cutoff_date: datetime,
    horizon_days: int = 90,
) -> pd.DataFrame:
    """
    Create a feature dataset for ML with strict temporal partitioning.

    Features are derived only from data before cutoff_date.
    Target is whether the pipe had a leak in [cutoff_date, cutoff_date + horizon_days].

    Args:
        pipes_df: Pipe network DataFrame (static attributes).
        events_df: Leak events DataFrame with 'date' column.
        cutoff_date: The temporal split point.
        horizon_days: Number of days after cutoff for the prediction target.

    Returns:
        DataFrame with features and 'target' column (1=leak, 0=no leak).
    """
    # Start with static pipe attributes
    features = pipes_df[
        [
            "pipe_id",
            "age",
            "diameter_m",
            "pressure_avg_m",
            "velocity_avg_ms",
            "depth_ft",
            "prev_repairs",
            "last_inspection_days",
            "traffic_load",
            "soil_corrosivity",
            "length_m",
        ]
    ].copy()

    # Encode categoricals
    for col in ["material", "diameter_category", "soil_type"]:
        if col in pipes_df.columns:
            dummies = pd.get_dummies(pipes_df[col], prefix=col, dtype=int)
            features = pd.concat([features, dummies], axis=1)

    # --- Historical features (strictly before cutoff) ---
    if events_df is not None and not events_df.empty:
        events_df = events_df.copy()
        events_df["date"] = pd.to_datetime(events_df["date"])
        historical = events_df[events_df["date"] < cutoff_date]

        # Leak count before cutoff
        leak_counts = historical.groupby("pipe_id").size().reset_index(name="hist_leak_count")
        features = features.merge(leak_counts, on="pipe_id", how="left")
        features["hist_leak_count"] = features["hist_leak_count"].fillna(0)

        # Days since last leak before cutoff
        last_leak = historical.groupby("pipe_id")["date"].max().reset_index()
        last_leak["days_since_last_leak"] = (cutoff_date - last_leak["date"]).dt.days
        last_leak = last_leak[["pipe_id", "days_since_last_leak"]]
        features = features.merge(last_leak, on="pipe_id", how="left")
        features["days_since_last_leak"] = features["days_since_last_leak"].fillna(365 * 5)

        # Average repair cost before cutoff
        avg_cost = historical.groupby("pipe_id")["repair_cost"].mean().reset_index()
        avg_cost = avg_cost.rename(columns={"repair_cost": "hist_avg_repair_cost"})
        features = features.merge(avg_cost, on="pipe_id", how="left")
        features["hist_avg_repair_cost"] = features["hist_avg_repair_cost"].fillna(0)

        # Leak count in last 12 months before cutoff
        recent_cutoff = cutoff_date - timedelta(days=365)
        recent = historical[historical["date"] >= recent_cutoff]
        recent_counts = recent.groupby("pipe_id").size().reset_index(name="recent_leak_count")
        features = features.merge(recent_counts, on="pipe_id", how="left")
        features["recent_leak_count"] = features["recent_leak_count"].fillna(0)

        # --- Target: leak in [cutoff, cutoff + horizon] ---
        horizon_end = cutoff_date + timedelta(days=horizon_days)
        future = events_df[
            (events_df["date"] >= cutoff_date) & (events_df["date"] < horizon_end)
        ]
        target_pipes = set(future["pipe_id"].unique())
        features["target"] = features["pipe_id"].isin(target_pipes).astype(int)
    else:
        features["hist_leak_count"] = 0
        features["days_since_last_leak"] = 365 * 5
        features["hist_avg_repair_cost"] = 0
        features["recent_leak_count"] = 0
        features["target"] = 0

    # Add month of cutoff as a feature (seasonal signal)
    features["cutoff_month"] = cutoff_date.month

    # Drop pipe_id (not a predictive feature)
    pipe_ids = features["pipe_id"].copy()
    features = features.drop(columns=["pipe_id"])

    # Fill any remaining NaN
    features = features.fillna(0)

    # Store pipe_ids as index for later reference
    features.index = pipe_ids.values

    return features


def get_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Get the list of feature column names (excludes target)."""
    return [c for c in features_df.columns if c != "target"]
