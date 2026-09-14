"""Tests for real-eval recorder mapping into canonical metrics records."""

from __future__ import annotations

import csv
import json

import pytest

from experiments.real_evaluation.policies import RoutingDecision
from experiments.real_evaluation.recorder import Recorder
from experiments.real_evaluation.transports import SingleRequestResult


def test_record_from_row_recovers_policy_hedge_identity(tmp_path) -> None:
    """A body-only policy row must reload as disabled, not guess hedging."""
    recorder = Recorder(tmp_path)
    decision = RoutingDecision(primary="primary")
    primary = SingleRequestResult(
        ttft_ms=300.0,
        e2e_ms=600.0,
        status="success",
        provider="primary",
        billed_cost_usd=0.02,
        physical_cost_usd=0.025,
    )
    recorder.write_request(
        policy="greedy_cost",
        req_id="r1",
        ctx_prompt_tokens=100,
        ctx_max_tokens=50,
        decision=decision,
        primary_result=primary,
        primary_tier="api",
        final_tier="api",
        slo_ms=2000.0,
        ts=100.0,
        hedge_algorithm="disabled",
        hedge_schedule=None,
        ctx_model="qwen/qwen3-235b",
    )

    # Reload path must recover policy identity from the persisted row instead
    # of guessing "probability_target" from backup presence (drift audit gap 2).
    reloaded = recorder._record_from_row(recorder._rows[0])
    assert reloaded.hedge_algorithm == "disabled"
    assert reloaded.hedge_schedule is None
    assert reloaded.model == "qwen/qwen3-235b"
    recorder.close()


def test_recorder_uses_user_visible_ttft_for_backup_winner(tmp_path) -> None:
    recorder = Recorder(tmp_path)
    decision = RoutingDecision(primary="primary", hedge="backup")
    primary = SingleRequestResult(
        ttft_ms=500.0,
        e2e_ms=800.0,
        status="success",
        provider="primary",
        billed_cost_usd=0.02,
        physical_cost_usd=0.025,
        start_ts=100.0,
        first_token_ts=100.5,
    )
    backup = SingleRequestResult(
        ttft_ms=120.0,
        e2e_ms=300.0,
        status="success",
        provider="backup",
        completion_tokens=20,
        billed_cost_usd=0.01,
        physical_cost_usd=0.012,
        start_ts=100.35,
        first_token_ts=100.47,
    )

    recorder.write_request(
        policy="routewise",
        req_id="req1",
        ctx_prompt_tokens=100,
        ctx_max_tokens=50,
        decision=decision,
        primary_result=primary,
        backup_result=backup,
        hedge_triggered=True,
        hedge_winner="backup",
        hedge_delay_sec=0.3,
        chosen_result=backup,
        primary_tier="api",
        backup_tier="quota",
        final_tier="quota",
        slo_ms=500.0,
        primary_cached_input_tokens=30,
        backup_cached_input_tokens=20,
        primary_routing_estimated_cost_usd=0.015,
        backup_routing_estimated_cost_usd=0.005,
        hedge_checkpoint_ts=100.31,
        backup_dispatch_ts=100.32,
        ts=101.0,
        hedge_algorithm="probability_target",
        hedge_schedule="slo_relative_checkpoints",
        ctx_model="qwen/qwen3-235b",
    )
    run = recorder.to_run()
    record = run.records[0]

    # Policy-level routing identity is persisted on the canonical record.
    assert record.model == "qwen/qwen3-235b"
    assert record.hedge_algorithm == "probability_target"
    assert record.hedge_schedule == "slo_relative_checkpoints"
    assert record.source == "real_eval"

    assert record.ttft_ms == pytest.approx(470.0)
    assert record.e2e_ms == pytest.approx(650.0)
    assert record.primary_local_ttft_ms == 500.0
    assert record.backup_local_ttft_ms == 120.0
    assert record.primary_tier == "api"
    assert record.backup_tier == "quota"
    assert record.final_tier == "quota"
    assert record.slo_ms == 500.0
    assert record.slo_violated is False
    assert run.cost_by_tier() == {"api": 0.02, "quota": 0.01}
    assert record.total_cost_usd == 0.03
    assert record.physical_cost_usd == pytest.approx(0.037)
    assert record.primary_physical_cost_usd == pytest.approx(0.025)
    assert record.backup_physical_cost_usd == pytest.approx(0.012)
    assert record.metadata["real_primary_cached_input_tokens"] == 30
    assert record.metadata["real_backup_cached_input_tokens"] == 20
    assert record.metadata["real_hedge_checkpoint_ts"] == pytest.approx(100.31)
    assert record.metadata["real_primary_start_ts"] == pytest.approx(100.0)
    assert record.metadata["real_backup_dispatch_ts"] == pytest.approx(100.32)
    assert record.metadata["real_backup_start_ts"] == pytest.approx(100.35)
    assert record.metadata["real_primary_first_token_ts"] == pytest.approx(100.5)
    assert record.metadata["real_backup_first_token_ts"] == pytest.approx(100.47)
    assert record.metadata["real_actual_dispatch_overhead_ms"] == pytest.approx(50.0)
    assert record.metadata["real_checkpoint_dispatch_overhead_ms"] == pytest.approx(40.0)
    assert record.metadata["real_backup_dispatch_overhead_ms"] == pytest.approx(30.0)

    with (tmp_path / "requests.csv").open(newline="") as fh:
        csv_row = next(csv.DictReader(fh))
    assert csv_row["model"] == "qwen/qwen3-235b"
    assert csv_row["hedge_algorithm"] == "probability_target"
    assert csv_row["hedge_schedule"] == "slo_relative_checkpoints"
    assert csv_row["hedge_checkpoint_ts"] == "100.310000"
    assert csv_row["primary_start_ts"] == "100.000000"
    assert csv_row["backup_dispatch_ts"] == "100.320000"
    assert csv_row["backup_start_ts"] == "100.350000"
    assert csv_row["primary_first_token_ts"] == "100.500000"
    assert csv_row["backup_first_token_ts"] == "100.470000"
    assert csv_row["actual_dispatch_overhead_ms"] == "50.000"
    assert csv_row["checkpoint_dispatch_overhead_ms"] == "40.000"
    assert csv_row["backup_dispatch_overhead_ms"] == "30.000"

    summary_path = recorder.write_summary(
        slo_ms=500.0,
        fixed_cost_usd_by_policy={"routewise": 0.07},
    )
    summary = json.loads(summary_path.read_text())
    assert summary["routewise"]["ttft_ms_p50"] == 470.0
    assert summary["routewise"]["e2e_ms_p50"] == 650.0
    assert summary["routewise"]["total_cost_usd"] == 0.10
    assert summary["routewise"]["mean_cost_usd"] == 0.10
    assert summary["routewise"]["total_physical_cost_usd"] == 0.037
    assert summary["routewise"]["slo_violation_rate"] == 0.0
    assert summary["routewise"]["actual_dispatch_overhead_ms_n"] == 1
    assert summary["routewise"]["actual_dispatch_overhead_ms_mean"] == 50.0
    assert summary["routewise"]["actual_dispatch_overhead_ms_p50"] == 50.0
    assert summary["routewise"]["actual_dispatch_overhead_ms_p90"] == 50.0
    assert summary["routewise"]["actual_dispatch_overhead_ms_p99"] == 50.0
    assert summary["routewise"]["checkpoint_dispatch_overhead_ms_n"] == 1
    assert summary["routewise"]["checkpoint_dispatch_overhead_ms_mean"] == 40.0
    assert summary["routewise"]["checkpoint_dispatch_overhead_ms_p50"] == 40.0
    assert summary["routewise"]["checkpoint_dispatch_overhead_ms_p90"] == 40.0
    assert summary["routewise"]["checkpoint_dispatch_overhead_ms_p99"] == 40.0
    assert summary["routewise"]["backup_dispatch_overhead_ms_n"] == 1
    assert summary["routewise"]["backup_dispatch_overhead_ms_mean"] == 30.0
    assert summary["routewise"]["backup_dispatch_overhead_ms_p50"] == 30.0
    assert summary["routewise"]["backup_dispatch_overhead_ms_p90"] == 30.0
    assert summary["routewise"]["backup_dispatch_overhead_ms_p99"] == 30.0
    recorder.close()


def test_recorder_persists_decision_state_and_generation_ids(tmp_path) -> None:
    recorder = Recorder(tmp_path)
    decision = RoutingDecision(
        primary="OR_A",
        hedge="Featherless_SC",
        lp_weights={"OR_A": 0.7, "Featherless_SC": 0.3},
        c_eff_map={"OR_A": 0.0004, "Featherless_SC": 0.0},
        latency_objective_ms={"OR_A": 650.0, "Featherless_SC": 400.0},
        c_min_usd=0.0,
        predicted_output_tokens=42,
        candidates=("OR_A", "Featherless_SC"),
        quota_fraction_used={"MiniMax_Plus_SQ": 0.25},
        concurrency_in_flight={"Featherless_SC": 1},
        hedge_success_probability=0.9,
    )
    primary = SingleRequestResult(
        ttft_ms=900.0, e2e_ms=1200.0, status="canceled", provider="OR_A", generation_id="gen-p"
    )
    backup = SingleRequestResult(
        ttft_ms=300.0,
        e2e_ms=500.0,
        status="success",
        provider="Featherless_SC",
        generation_id="gen-b",
    )
    recorder.write_request(
        policy="budget_range_alpha50_hedge",
        req_id="r1",
        ctx_prompt_tokens=100,
        ctx_max_tokens=50,
        decision=decision,
        primary_result=primary,
        backup_result=backup,
        hedge_triggered=True,
        hedge_winner="backup",
        chosen_result=backup,
        slo_ms=3000.0,
        ts=100.0,
        hedge_algorithm="probability_target",
        hedge_schedule="slo_relative_checkpoints",
    )
    recorder.close()

    with (tmp_path / "requests.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    row = rows[0]
    assert json.loads(row["latency_objective_ms"]) == {"OR_A": 650.0, "Featherless_SC": 400.0}
    assert json.loads(row["c_eff"]) == {"OR_A": 0.0004, "Featherless_SC": 0.0}
    assert row["c_min_usd"] == "0.00000000"
    assert row["predicted_output_tokens"] == "42"
    assert json.loads(row["candidates"]) == ["OR_A", "Featherless_SC"]
    assert json.loads(row["quota_fraction_used"]) == {"MiniMax_Plus_SQ": 0.25}
    assert json.loads(row["concurrency_in_flight"]) == {"Featherless_SC": 1}
    assert row["hedge_success_probability"] == "0.900000"
    assert row["primary_generation_id"] == "gen-p"
    assert row["backup_generation_id"] == "gen-b"


def test_recorder_leaves_decision_state_empty_for_baselines(tmp_path) -> None:
    recorder = Recorder(tmp_path)
    recorder.write_request(
        policy="greedy_cost",
        req_id="r1",
        ctx_prompt_tokens=100,
        ctx_max_tokens=50,
        decision=RoutingDecision(primary="OR_A"),
        primary_result=SingleRequestResult(
            ttft_ms=300.0, e2e_ms=600.0, status="success", provider="OR_A"
        ),
        slo_ms=3000.0,
        ts=100.0,
    )
    recorder.close()

    with (tmp_path / "requests.csv").open() as handle:
        row = next(csv.DictReader(handle))
    for column in (
        "latency_objective_ms",
        "c_eff",
        "c_min_usd",
        "predicted_output_tokens",
        "candidates",
        "quota_fraction_used",
        "concurrency_in_flight",
        "hedge_success_probability",
        "primary_generation_id",
        "backup_generation_id",
    ):
        assert row[column] == ""
