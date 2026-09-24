"""Tests for streaming TTFT histograms."""

from __future__ import annotations

import numpy as np
import pytest

from rwsim.metrics.histogram import (
    HISTOGRAM_BIN_COUNT,
    HISTOGRAM_MAX_MS,
    HISTOGRAM_MIN_MS,
    TtftHistogram,
)


def test_default_histogram_contract_is_explicit() -> None:
    histogram = TtftHistogram.default()

    assert HISTOGRAM_MIN_MS == 1.0
    assert HISTOGRAM_MAX_MS == 100_000.0
    assert HISTOGRAM_BIN_COUNT == 256
    assert histogram.bin_edges_ms[0] == HISTOGRAM_MIN_MS
    assert histogram.bin_edges_ms[-1] == HISTOGRAM_MAX_MS
    assert histogram.bin_edges_ms.size == HISTOGRAM_BIN_COUNT + 1
    assert histogram.counts.size == HISTOGRAM_BIN_COUNT + 2


def test_histogram_counts_underflow_overflow_and_values() -> None:
    histogram = TtftHistogram(bin_edges_ms=np.asarray([10.0, 100.0, 1000.0]))

    histogram.add_array(np.asarray([1.0, 10.0, 50.0, 100.0, 500.0, 5000.0]))

    assert histogram.n == 6
    assert histogram.counts.tolist() == [1, 2, 2, 1]
    assert histogram.mean() == pytest.approx((1.0 + 10.0 + 50.0 + 100.0 + 500.0 + 5000.0) / 6)


def test_scalar_add_matches_batch_add() -> None:
    values = np.asarray(
        [float("-inf"), 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, float("nan")],
    )
    scalar = TtftHistogram(bin_edges_ms=np.asarray([10.0, 100.0, 1000.0]))
    batch = TtftHistogram(bin_edges_ms=np.asarray([10.0, 100.0, 1000.0]))

    for value in values:
        scalar.add(float(value))
    batch.add_array(values)

    assert scalar.n == batch.n
    assert scalar.counts.tolist() == batch.counts.tolist()
    assert scalar.mean() == pytest.approx(batch.mean())
    assert scalar.std() == pytest.approx(batch.std())
    assert scalar.min_value == pytest.approx(batch.min_value)
    assert scalar.max_value == pytest.approx(batch.max_value)


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


def test_histogram_quantiles_use_min_max_for_underflow_and_overflow() -> None:
    histogram = TtftHistogram.default()

    histogram.add_array(np.asarray([0.5, 10_000_000.0]))

    assert histogram.quantile(0.01) == pytest.approx(0.5)
    assert histogram.quantile(0.99) == pytest.approx(10_000_000.0)


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
