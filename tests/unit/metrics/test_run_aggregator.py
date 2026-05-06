"""Tests for streaming run aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from rwsim.metrics import PerRequestRecord, Run, RunAggregator, Status


def test_aggregated_run_matches_exact_record_metrics_for_counts_and_costs() -> None:
    records = [
        _record("r1", ttft_ms=100.0, final_provider="api_a", primary_cost=0.01),
        _record(
            "r2",
            ttft_ms=300.0,
            final_provider="api_b",
            primary_cost=0.02,
            backup_provider="api_c",
            backup_cost=0.03,
            hedge_triggered=True,
            hedge_winner="backup",
        ),
        _record(
            "r3",
            ttft_ms=900.0,
            final_provider="api_b",
            primary_cost=0.04,
            status=Status.REJECTED,
            slo_violated=True,
        ),
    ]

    exact = Run(records=records, policy="exact")
    aggregator = RunAggregator(policy="agg", retain_records=False)
    for record in records:
        aggregator.observe(record)
    aggregated = aggregator.finalize()

    assert aggregated.records == []
    assert aggregated.total_cost_usd() == pytest.approx(exact.total_cost_usd())
    assert aggregated.mean_cost_usd() == pytest.approx(exact.mean_cost_usd())
    assert aggregated.cost_by_provider() == exact.cost_by_provider()
    assert aggregated.cost_by_tier() == exact.cost_by_tier()
    assert aggregated.provider_fractions() == exact.provider_fractions()
    assert aggregated.tier_fractions() == exact.tier_fractions()
    assert aggregated.status_breakdown() == exact.status_breakdown()
    assert aggregated.slo_violation_rate() == exact.slo_violation_rate()
    assert aggregated.hedge_rate() == exact.hedge_rate()
    assert aggregated.hedge_winner_rate() == exact.hedge_winner_rate()
    assert aggregated.mean_ttft_ms() == pytest.approx(exact.mean_ttft_ms())


def test_aggregated_run_percentiles_track_exact_record_path() -> None:
    rng = np.random.default_rng(0)
    ttft_values = rng.lognormal(mean=np.log(300.0), sigma=0.7, size=50_000)
    records = [
        _record(
            f"r{idx}",
            ttft_ms=float(value),
            final_provider="api_a",
            primary_cost=0.01,
        )
        for idx, value in enumerate(ttft_values)
    ]

    exact = Run(records=records, policy="exact")
    aggregator = RunAggregator(policy="agg", retain_records=False)
    for record in records:
        aggregator.observe(record)
    aggregated = aggregator.finalize()

    assert aggregated.p50_ms() == pytest.approx(exact.p50_ms(), rel=0.03)
    assert aggregated.p95_ms() == pytest.approx(exact.p95_ms(), rel=0.03)
    assert aggregated.p99_ms() == pytest.approx(exact.p99_ms(), rel=0.03)


def _record(
    request_id: str,
    *,
    ttft_ms: float,
    final_provider: str,
    primary_cost: float,
    backup_provider: str | None = None,
    backup_cost: float | None = None,
    hedge_triggered: bool = False,
    hedge_winner: str | None = None,
    status: Status = Status.SUCCESS,
    slo_violated: bool = False,
) -> PerRequestRecord:
    return PerRequestRecord(
        request_id=request_id,
        elapsed_sec=0.0,
        policy="p",
        prompt_tokens=100,
        completion_tokens_budget=50,
        completion_tokens_actual=50,
        primary_provider=final_provider,
        primary_tier="api",
        final_provider=final_provider,
        final_tier="api",
        backup_provider=backup_provider,
        backup_tier="api" if backup_provider else None,
        ttft_ms=ttft_ms,
        primary_local_ttft_ms=ttft_ms,
        slo_ms=500.0,
        slo_violated=slo_violated,
        total_cost_usd=primary_cost + float(backup_cost or 0.0),
        primary_cost_usd=primary_cost,
        backup_cost_usd=backup_cost,
        hedge_triggered=hedge_triggered,
        hedge_winner=hedge_winner,
        status=status,
    )
