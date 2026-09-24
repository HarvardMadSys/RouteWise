"""Tests for shared synthetic latency distribution construction."""

from __future__ import annotations

import math

import pytest

from experiments.simulation.latency_factory import (
    SyntheticLatencySpec,
    make_synthetic_ttft,
    synthetic_latency_metadata,
)


def test_heavy_tail_mean_anchor_sets_true_mean_not_p50():
    dist = make_synthetic_ttft(
        SyntheticLatencySpec(
            family="heavy_tail",
            anchor_ms=300.0,
            anchor_kind="mean",
            lognormal_sigma=0.5,
        )
    )

    assert dist.mean() == pytest.approx(300.0)
    assert dist.p50() == pytest.approx(300.0 * math.exp(-0.5 * 0.5 * 0.5))


def test_heavy_tail_p50_anchor_remains_available_explicitly():
    dist = make_synthetic_ttft(
        SyntheticLatencySpec(
            family="heavy_tail",
            anchor_ms=300.0,
            anchor_kind="p50",
            lognormal_sigma=0.5,
        )
    )

    assert dist.p50() == pytest.approx(300.0)
    assert dist.mean() == pytest.approx(300.0 * math.exp(0.5 * 0.5 * 0.5))


def test_symmetric_families_have_same_mean_and_p50_anchor_value():
    for family in ("uniform", "normal"):
        dist = make_synthetic_ttft(
            SyntheticLatencySpec(
                family=family,
                anchor_ms=300.0,
                anchor_kind="mean",
            )
        )

        assert dist.mean() == pytest.approx(300.0)
        assert dist.p50() == pytest.approx(300.0)


def test_metadata_records_anchor_semantics_and_true_moments():
    metadata = synthetic_latency_metadata(
        SyntheticLatencySpec(
            family="lognormal",
            anchor_ms=300.0,
            anchor_kind="mean",
            lognormal_sigma=0.5,
        )
    )

    assert metadata["latency_generation_version"] == "synthetic_latency_v2"
    assert metadata["latency_family"] == "heavy_tail"
    assert metadata["latency_anchor_kind"] == "mean"
    assert metadata["latency_anchor_ms"] == 300.0
    assert metadata["latency_distribution_mean_ms"] == pytest.approx(300.0)
    assert metadata["latency_distribution_p50_ms"] == pytest.approx(
        300.0 * math.exp(-0.5 * 0.5 * 0.5)
    )
