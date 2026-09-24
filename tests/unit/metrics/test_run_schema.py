"""Tests for the canonical metrics Run schema."""

from __future__ import annotations

import pytest

from rwsim.metrics import PerRequestRecord, Run, Status


def test_run_cost_by_tier_attributes_primary_and_backup_costs() -> None:
    run = Run(
        records=[
            PerRequestRecord(
                request_id="r1",
                elapsed_sec=0.0,
                policy="p",
                prompt_tokens=100,
                completion_tokens_budget=50,
                completion_tokens_actual=40,
                primary_provider="api_a",
                primary_tier="api",
                final_provider="quota_b",
                final_tier="quota",
                backup_provider="quota_b",
                backup_tier="quota",
                ttft_ms=180.0,
                e2e_ms=400.0,
                slo_ms=200.0,
                slo_violated=False,
                total_cost_usd=0.03,
                primary_cost_usd=0.02,
                backup_cost_usd=0.01,
                hedge_triggered=True,
                hedge_winner="backup",
                status=Status.SUCCESS,
            )
        ],
        policy="p",
    )

    assert run.cost_by_tier() == {"api": 0.02, "quota": 0.01}
    assert run.cost_by_provider() == {"api_a": 0.02, "quota_b": 0.01}
    assert run.hedge_winner_rate() == {"backup": 1.0}
    assert run.mean_e2e_ms() == 400.0


def test_run_aggregates_records_without_column_constructor() -> None:
    run = Run(
        records=[
            _record("r1", ttft_ms=10.0, final_provider="a", final_tier="api"),
            _record("r2", ttft_ms=20.0, final_provider="b", final_tier="quota"),
        ],
        policy="records",
    )

    assert len(run.records) == 2
    assert run.p90_ms() == 19.0
    assert run.provider_fractions() == {"a": 0.5, "b": 0.5}


def test_run_has_no_legacy_column_surface() -> None:
    run = Run(records=[_record("r1", ttft_ms=10.0)], policy="records")

    for name in (
        "ttft_ms",
        "cost_usd",
        "provider",
        "timestamp",
        "hedge_triggered",
        "tier",
        "quota_fraction_used",
        "concurrency_utilization",
        "rejected",
    ):
        assert not hasattr(run, name)

    with pytest.raises(TypeError):
        Run(
            # type: ignore[call-arg]
            ttft_ms=[10.0],
            cost_usd=[0.1],
            provider=["api_a"],
        )


def test_slo_violation_rate_uses_record_flags_or_explicit_slo() -> None:
    run = Run(
        records=[
            _record("r1", ttft_ms=50.0, slo_violated=False),
            _record("r2", ttft_ms=200.0, status=Status.REJECTED, slo_violated=True),
        ],
        policy="records",
    )

    assert run.slo_violation_rate() == 0.5
    assert run.slo_violation_rate(100.0) == 0.5


def _record(
    request_id: str,
    *,
    ttft_ms: float,
    final_provider: str = "api_a",
    final_tier: str = "api",
    status: Status = Status.SUCCESS,
    slo_violated: bool = False,
) -> PerRequestRecord:
    return PerRequestRecord(
        request_id=request_id,
        elapsed_sec=0.0,
        policy="records",
        prompt_tokens=100,
        completion_tokens_budget=50,
        completion_tokens_actual=40,
        primary_provider=final_provider,
        primary_tier=final_tier,
        final_provider=final_provider,
        final_tier=final_tier,
        ttft_ms=ttft_ms,
        primary_local_ttft_ms=ttft_ms,
        slo_ms=100.0,
        slo_violated=slo_violated,
        total_cost_usd=0.1,
        primary_cost_usd=0.1,
        status=status,
    )
