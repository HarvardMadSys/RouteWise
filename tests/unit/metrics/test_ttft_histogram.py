"""Tests for streaming TTFT histograms."""

from __future__ import annotations

import numpy as np
import pytest

from rwsim.metrics.histogram import TtftHistogram


def test_histogram_counts_underflow_overflow_and_values() -> None:
    histogram = TtftHistogram(bin_edges_ms=np.asarray([10.0, 100.0, 1000.0]))

    histogram.add_array(np.asarray([1.0, 10.0, 50.0, 100.0, 500.0, 5000.0]))

    assert histogram.n == 6
    assert histogram.counts.tolist() == [1, 2, 2, 1]
    assert histogram.mean() == pytest.approx((1.0 + 10.0 + 50.0 + 100.0 + 500.0 + 5000.0) / 6)


def test_histogram_quantiles_track_numpy_reference_with_reasonable_precision() -> None:
    rng = np.random.default_rng(0)
    values = rng.lognormal(mean=np.log(300.0), sigma=0.7, size=20_000)
    histogram = TtftHistogram.default()

    histogram.add_array(values)

    for q in (0.5, 0.9, 0.99):
        assert histogram.quantile(q) == pytest.approx(
            float(np.percentile(values, q * 100.0)),
            rel=0.03,
        )


def test_histogram_merge_matches_one_big_histogram() -> None:
    values = np.asarray([10.0, 20.0, 100.0, 1000.0, 5000.0])
    left = TtftHistogram.default()
    right = TtftHistogram.default()
    combined = TtftHistogram.default()

    left.add_array(values[:2])
    right.add_array(values[2:])
    combined.add_array(values)

    merged = left.merge(right)

    assert merged.n == combined.n
    assert merged.counts.tolist() == combined.counts.tolist()
    assert merged.mean() == pytest.approx(combined.mean())
