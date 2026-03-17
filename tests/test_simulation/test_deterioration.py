"""Tests for the Weibull deterioration model."""

import numpy as np
import pytest

from pipe_leak.simulation.deterioration import (
    weibull_annual_failure_rate,
    compute_failure_probabilities,
)


class TestWeibullRate:
    def test_increases_with_age(self):
        """Failure rate should increase with pipe age (beta > 1)."""
        ages = np.array([10, 30, 50, 80])
        beta = np.full_like(ages, 2.5, dtype=float)
        eta = np.full_like(ages, 55.0, dtype=float)
        rates = weibull_annual_failure_rate(ages, beta, eta)

        assert all(rates[i] < rates[i + 1] for i in range(len(rates) - 1)), (
            "Failure rate should monotonically increase with age"
        )

    def test_bounded_probability(self):
        """Probabilities should be in [0, 1)."""
        ages = np.array([1, 50, 100, 200])
        beta = np.full(4, 2.0)
        eta = np.full(4, 60.0)
        rates = weibull_annual_failure_rate(ages, beta, eta)

        assert np.all(rates >= 0)
        assert np.all(rates < 1.0)

    def test_young_pipes_low_risk(self):
        """Pipes aged 5 should have very low failure rate."""
        rate = weibull_annual_failure_rate(
            np.array([5.0]), np.array([2.5]), np.array([55.0])
        )
        assert rate[0] < 0.05, f"Young pipe rate too high: {rate[0]}"

    def test_material_ordering(self):
        """HDPE should have lower failure rate than Cast Iron at same age."""
        age = np.array([50.0, 50.0])
        # Cast Iron: beta=2.5, eta=55; HDPE: beta=1.6, eta=130
        beta = np.array([2.5, 1.6])
        eta = np.array([55.0, 130.0])
        rates = weibull_annual_failure_rate(age, beta, eta)

        assert rates[0] > rates[1], "Cast Iron should have higher failure rate than HDPE"


class TestComputeFailureProbabilities:
    def test_output_shape(self, sample_pipes):
        """Output should have one probability per pipe."""
        probs = compute_failure_probabilities(sample_pipes)
        assert len(probs) == len(sample_pipes)

    def test_output_range(self, sample_pipes):
        """All probabilities should be in [0.001, 0.95]."""
        probs = compute_failure_probabilities(sample_pipes)
        assert np.all(probs >= 0.001)
        assert np.all(probs <= 0.95)

    def test_older_pipes_higher_risk(self, sample_pipes):
        """On average, older pipes should have higher failure probability."""
        probs = compute_failure_probabilities(sample_pipes)
        median_age = sample_pipes["age"].median()
        old_mask = sample_pipes["age"] > median_age
        young_mask = sample_pipes["age"] <= median_age

        avg_old = probs[old_mask.values].mean()
        avg_young = probs[young_mask.values].mean()

        assert avg_old > avg_young, (
            f"Old pipes ({avg_old:.3f}) should have higher avg risk than young ({avg_young:.3f})"
        )
