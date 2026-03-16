"""Tests for feature engineering — especially no data leakage."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from pipe_leak.ml.features import create_feature_dataset, get_feature_columns


class TestCreateFeatureDataset:
    def test_no_data_leakage(self, sample_pipes, sample_events):
        """Features should only use data BEFORE the cutoff date."""
        cutoff = datetime(2023, 1, 1)
        horizon = 90

        features = create_feature_dataset(sample_pipes, sample_events, cutoff, horizon)

        # The leak count feature should only count events before cutoff
        events_before = sample_events[pd.to_datetime(sample_events["date"]) < cutoff]
        expected_counts = events_before.groupby("pipe_id").size()

        for pipe_id in features.index:
            expected = expected_counts.get(pipe_id, 0)
            actual = features.loc[pipe_id, "hist_leak_count"] if pipe_id in features.index else 0
            assert actual == expected, (
                f"Pipe {pipe_id}: expected {expected} pre-cutoff leaks, got {actual}"
            )

    def test_target_from_future_only(self, sample_pipes, sample_events):
        """Target should only reflect events in [cutoff, cutoff+horizon]."""
        cutoff = datetime(2023, 1, 1)
        horizon = 90

        features = create_feature_dataset(sample_pipes, sample_events, cutoff, horizon)

        horizon_end = cutoff + timedelta(days=horizon)
        events = sample_events.copy()
        events["date"] = pd.to_datetime(events["date"])
        future_events = events[(events["date"] >= cutoff) & (events["date"] < horizon_end)]
        expected_pipes = set(future_events["pipe_id"].unique())

        for pipe_id in features.index:
            expected_target = 1 if pipe_id in expected_pipes else 0
            actual_target = features.loc[pipe_id, "target"]
            assert actual_target == expected_target, (
                f"Pipe {pipe_id}: expected target={expected_target}, got {actual_target}"
            )

    def test_has_target_column(self, sample_pipes, sample_events):
        """Output should include 'target' column."""
        features = create_feature_dataset(
            sample_pipes, sample_events, datetime(2023, 6, 1), 90
        )
        assert "target" in features.columns

    def test_no_pipe_id_in_features(self, sample_pipes, sample_events):
        """pipe_id should not be a feature column (it's in the index)."""
        features = create_feature_dataset(
            sample_pipes, sample_events, datetime(2023, 6, 1), 90
        )
        assert "pipe_id" not in features.columns

    def test_feature_columns_helper(self, sample_pipes, sample_events):
        """get_feature_columns should exclude target."""
        features = create_feature_dataset(
            sample_pipes, sample_events, datetime(2023, 6, 1), 90
        )
        cols = get_feature_columns(features)
        assert "target" not in cols
        assert len(cols) > 5  # should have meaningful number of features


class TestTemporalIntegrity:
    def test_days_since_last_leak_correct(self, sample_pipes, sample_events):
        """days_since_last_leak should reflect time from most recent pre-cutoff leak."""
        cutoff = datetime(2023, 6, 1)
        features = create_feature_dataset(sample_pipes, sample_events, cutoff, 90)

        events = sample_events.copy()
        events["date"] = pd.to_datetime(events["date"])
        pre_cutoff = events[events["date"] < cutoff]

        for pipe_id in features.index:
            pipe_events = pre_cutoff[pre_cutoff["pipe_id"] == pipe_id]
            if len(pipe_events) > 0:
                last_date = pipe_events["date"].max()
                expected_days = (cutoff - last_date).days
                actual_days = features.loc[pipe_id, "days_since_last_leak"]
                assert actual_days == expected_days, (
                    f"Pipe {pipe_id}: expected {expected_days} days, got {actual_days}"
                )
