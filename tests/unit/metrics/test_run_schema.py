"""Tests for the canonical metrics Run schema."""

from __future__ import annotations

import numpy as np

from rwsim.metrics import PerRequestRecord, Run, SimulationRun, Status


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


def test_simulation_run_legacy_columns_still_work() -> None:
    run = SimulationRun(
        policy="legacy",
        ttft_ms=np.array([10.0, 20.0]),
        cost_usd=np.array([0.1, 0.3]),
        provider=["a", "b"],
        timestamp=np.array([0.0, 1.0]),
        hedge_triggered=np.array([False, True]),
        tier=["api", "quota"],
    )

    assert len(run.records) == 2
    assert run.p90_ms() == 19.0
    assert run.provider_fractions() == {"a": 0.5, "b": 0.5}


def test_legacy_columns_only_compute_slo_violations_with_explicit_slo() -> None:
    run_without_slo = SimulationRun(
        policy="legacy",
        ttft_ms=np.array([50.0, 200.0]),
        cost_usd=np.array([0.1, 0.1]),
        provider=["a", "a"],
        timestamp=np.array([0.0, 1.0]),
        hedge_triggered=np.array([False, False]),
        rejected=np.array([False, True]),
    )

    assert [record.slo_violated for record in run_without_slo.records] == [False, False]
    assert run_without_slo.slo_violation_rate() == 0.0
    assert run_without_slo.slo_violation_rate(100.0) == 0.5

    run_with_slo = SimulationRun(
        policy="legacy",
        ttft_ms=np.array([50.0, 200.0]),
        cost_usd=np.array([0.1, 0.1]),
        provider=["a", "a"],
        timestamp=np.array([0.0, 1.0]),
        hedge_triggered=np.array([False, False]),
        rejected=np.array([False, False]),
        slo_ms=100.0,
    )

    assert [record.slo_violated for record in run_with_slo.records] == [False, True]
    assert run_with_slo.slo_violation_rate() == 0.5
