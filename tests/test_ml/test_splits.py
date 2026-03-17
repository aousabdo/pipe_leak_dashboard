"""Tests for temporal train/test splitting."""

import pytest
import pandas as pd
from datetime import datetime

from pipe_leak.ml.splits import temporal_train_test_split


class TestTemporalSplit:
    def test_produces_train_and_test(self, sample_pipes, sample_events):
        """Should produce non-empty train and test sets."""
        train, test = temporal_train_test_split(
            sample_pipes, sample_events, horizon_days=60
        )
        assert len(train) > 0
        assert len(test) > 0

    def test_both_have_target(self, sample_pipes, sample_events):
        """Both sets should have 'target' column."""
        train, test = temporal_train_test_split(
            sample_pipes, sample_events, horizon_days=60
        )
        assert "target" in train.columns
        assert "target" in test.columns

    def test_same_feature_columns(self, sample_pipes, sample_events):
        """Train and test should have identical column sets."""
        train, test = temporal_train_test_split(
            sample_pipes, sample_events, horizon_days=60
        )
        assert set(train.columns) == set(test.columns)
