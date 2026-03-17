"""
Temporal train/test splitting.

Provides expanding-window temporal cross-validation that prevents
future data from leaking into training sets.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Iterator

from pipe_leak.ml.features import create_feature_dataset


def temporal_train_test_split(
    pipes_df: pd.DataFrame,
    events_df: pd.DataFrame,
    test_fraction: float = 0.2,
    horizon_days: int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally: train on earlier period, test on later period.

    Args:
        pipes_df: Pipe network DataFrame.
        events_df: Leak events DataFrame.
        test_fraction: Fraction of the total time span to use for testing.
        horizon_days: Prediction horizon in days.

    Returns:
        (train_features, test_features) DataFrames each with 'target' column.
    """
    events_df = events_df.copy()
    events_df["date"] = pd.to_datetime(events_df["date"])

    date_min = events_df["date"].min()
    date_max = events_df["date"].max()
    total_days = (date_max - date_min).days

    # Cutoff: the point where we split train and test
    # Train features use history before cutoff; target is [cutoff, cutoff+horizon]
    # Test features use history before test_cutoff; target is [test_cutoff, test_cutoff+horizon]
    train_cutoff = date_min + timedelta(days=int(total_days * (1 - test_fraction) - horizon_days))
    test_cutoff = date_min + timedelta(days=int(total_days * (1 - test_fraction)))

    train_df = create_feature_dataset(pipes_df, events_df, train_cutoff, horizon_days)
    test_df = create_feature_dataset(pipes_df, events_df, test_cutoff, horizon_days)

    return train_df, test_df


def temporal_cv_splits(
    pipes_df: pd.DataFrame,
    events_df: pd.DataFrame,
    n_folds: int = 5,
    horizon_days: int = 90,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Expanding-window temporal cross-validation.

    Each fold uses all data up to a cutoff for features/training,
    and the next horizon_days for the target/validation.

    Yields:
        (train_features, val_features) tuples for each fold.
    """
    events_df = events_df.copy()
    events_df["date"] = pd.to_datetime(events_df["date"])

    date_min = events_df["date"].min()
    date_max = events_df["date"].max()
    total_days = (date_max - date_min).days

    # Reserve the first 30% for initial training history
    min_train_days = int(total_days * 0.3)
    remaining_days = total_days - min_train_days - horizon_days
    fold_step = remaining_days // n_folds

    for fold in range(n_folds):
        cutoff = date_min + timedelta(days=min_train_days + fold * fold_step)
        train_df = create_feature_dataset(pipes_df, events_df, cutoff, horizon_days)

        val_cutoff = cutoff + timedelta(days=horizon_days)
        if val_cutoff > date_max:
            break

        val_df = create_feature_dataset(pipes_df, events_df, val_cutoff, horizon_days)
        yield train_df, val_df
