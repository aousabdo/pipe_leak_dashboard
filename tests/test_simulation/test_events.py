"""Tests for leak event generation."""

import pytest
import pandas as pd

from pipe_leak.config import SimulationConfig
from pipe_leak.simulation.events import generate_leak_events


class TestGenerateLeakEvents:
    def test_produces_events(self, sample_pipes):
        """Should generate at least some events for a pipe network."""
        config = SimulationConfig(seed=42, num_pipes=30, simulation_years=3)
        events = generate_leak_events(sample_pipes, config)
        assert len(events) > 0, "Should generate at least some leak events"

    def test_event_columns(self, sample_pipes):
        """Events should have required columns."""
        config = SimulationConfig(seed=42, num_pipes=30, simulation_years=2)
        events = generate_leak_events(sample_pipes, config)

        required = [
            "pipe_id", "date", "severity", "flow_rate_gpm",
            "detection_hours", "water_loss_gallons", "repair_cost",
        ]
        for col in required:
            assert col in events.columns, f"Missing column: {col}"

    def test_valid_severities(self, sample_pipes):
        """All severities should be from the defined set."""
        config = SimulationConfig(seed=42, num_pipes=30, simulation_years=2)
        events = generate_leak_events(sample_pipes, config)

        valid = {"Minor", "Moderate", "Major", "Critical"}
        actual = set(events["severity"].unique())
        assert actual.issubset(valid), f"Invalid severities: {actual - valid}"

    def test_positive_costs(self, sample_pipes):
        """All repair costs should be positive."""
        config = SimulationConfig(seed=42, num_pipes=30, simulation_years=2)
        events = generate_leak_events(sample_pipes, config)

        assert (events["repair_cost"] > 0).all()
        assert (events["flow_rate_gpm"] > 0).all()

    def test_events_within_timeframe(self, sample_pipes):
        """Events should fall within the simulation years."""
        config = SimulationConfig(seed=42, num_pipes=30, simulation_years=3)
        events = generate_leak_events(sample_pipes, config)

        dates = pd.to_datetime(events["date"])
        assert dates.min().year >= 2026 - config.simulation_years
        assert dates.max().year <= 2026
