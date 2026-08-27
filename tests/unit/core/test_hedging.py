"""Tests for the public RouteWise core hedging primitives."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import llm_routewise.core as core
from llm_routewise.core.hedging import (
    BackupCandidate,
    combined_success_probability,
    has_feasible_backup,
    select_probability_backup,
)


def _provider(name: str):
    return SimpleNamespace(name=name)


def test_core_package_exports_public_hedging_api() -> None:
    assert core.BackupCandidate is BackupCandidate
    assert core.combined_success_probability is combined_success_probability
    assert core.select_probability_backup is select_probability_backup


def test_combined_success_probability_uses_conditional_primary_and_backup() -> None:
    def primary_cdf(value_ms: float) -> float:
        if value_ms <= 500.0:
            return 0.20
        return 0.80

    def backup_cdf(value_ms: float) -> float:
        if value_ms <= 400.0:
            return 0.70
        return 1.0

    probability = combined_success_probability(
        primary_cdf,
        backup_cdf,
        elapsed_ms=500.0,
        slo_ms=1000.0,
        dispatch_overhead_ms=100.0,
    )

    # Given primary survived past elapsed: P(primary still meets SLO)=0.6/0.8,
    # P(primary misses SLO)=0.2/0.8, backup succeeds in remaining time with 0.7.
    assert probability == pytest.approx(0.75 + 0.25 * 0.70)


def test_combined_success_probability_has_zero_backup_when_no_time_remains() -> None:
    def primary_cdf(value_ms: float) -> float:
        return 0.50 if value_ms < 1000.0 else 0.80

    probability = combined_success_probability(
        primary_cdf,
        lambda _value_ms: 1.0,
        elapsed_ms=990.0,
        slo_ms=1000.0,
        dispatch_overhead_ms=50.0,
    )

    assert probability == pytest.approx((0.80 - 0.50) / (1.0 - 0.50))


def test_combined_success_probability_uses_backup_when_primary_survival_is_zero() -> None:
    probability = combined_success_probability(
        lambda _value_ms: 1.0,
        lambda value_ms: 0.75 if value_ms >= 400.0 else 0.0,
        elapsed_ms=500.0,
        slo_ms=1000.0,
        dispatch_overhead_ms=100.0,
    )

    assert probability == pytest.approx(0.75)


def test_select_probability_backup_prefers_cost_then_probability_then_latency() -> None:
    expensive = BackupCandidate(
        provider=_provider("expensive"),
        success_probability=0.999,
        marginal_cost=3.0,
        true_mean_ms=100.0,
    )
    cheap_lower_probability = BackupCandidate(
        provider=_provider("cheap_lower_probability"),
        success_probability=0.991,
        marginal_cost=1.0,
        true_mean_ms=100.0,
    )
    cheap_higher_probability = BackupCandidate(
        provider=_provider("cheap_higher_probability"),
        success_probability=0.995,
        marginal_cost=1.0,
        true_mean_ms=150.0,
    )

    selected = select_probability_backup(
        [expensive, cheap_lower_probability, cheap_higher_probability]
    )

    assert selected is cheap_higher_probability


def test_select_probability_backup_ignores_infeasible_candidates() -> None:
    infeasible = BackupCandidate(
        provider=_provider("infeasible"),
        success_probability=0.98,
        marginal_cost=0.0,
        true_mean_ms=10.0,
    )

    assert select_probability_backup([infeasible]) is None
    assert not has_feasible_backup([infeasible])
