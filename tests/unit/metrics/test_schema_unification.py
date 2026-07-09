"""Cross-source checks for the canonical RouteWise request schema."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import pytest

from routewise.metrics import PerRequestRecord, Status

CANONICAL_ROUTEWISE_REQUEST_FIELDS = {
    "request_id",
    "policy",
    "model",
    "source",
    "timestamp_sec",
    "prompt_tokens",
    "completion_tokens_actual",
    "primary_provider",
    "primary_tier",
    "backup_provider",
    "backup_tier",
    "final_provider",
    "final_tier",
    "ttft_ms",
    "e2e_ms",
    "slo_ms",
    "slo_violated",
    "total_cost_usd",
    "physical_cost_usd",
    "routing_estimated_cost_usd",
    "hedge_triggered",
    "hedge_delay_ms",
    "hedge_winner",
    "hedge_algorithm",
    "hedge_schedule",
    "lp_weights",
    "lp_budget_usd",
    "lp_status",
    "status",
    "error_class",
}


def test_per_request_record_declares_canonical_routewise_fields() -> None:
    """SIM and REAL-EVAL records expose the canonical request fields directly."""
    field_names = {field.name for field in fields(PerRequestRecord)}

    assert CANONICAL_ROUTEWISE_REQUEST_FIELDS <= field_names


@pytest.mark.parametrize(
    ("source", "payload"),
    [
        ("sim", lambda: _record_payload(_sim_record())),
        ("real_eval", lambda: _record_payload(_real_eval_record())),
        ("prod", lambda: _prod_payload(_prod_api_log_row())),
    ],
)
def test_sim_real_eval_and_prod_payloads_share_canonical_routewise_contract(
    source: str,
    payload: Any,
) -> None:
    """Fixtures from all three sources normalize to the same canonical keys."""
    data = payload()

    assert data["source"] == source
    assert CANONICAL_ROUTEWISE_REQUEST_FIELDS <= set(data)
    assert isinstance(data["request_id"], str)
    assert isinstance(data["policy"], str)
    assert isinstance(data["model"], str)
    assert data["source"] in {"sim", "real_eval", "prod"}
    assert data["primary_provider"]
    assert data["primary_tier"] in {"api", "quota", "concurrency"}
    assert isinstance(data["hedge_triggered"], bool)
    assert data["hedge_algorithm"] in {"probability_target", "disabled"}
    if data["hedge_algorithm"] == "disabled":
        assert data["hedge_schedule"] is None
        assert data["hedge_triggered"] is False
    else:
        assert data["hedge_schedule"] == "slo_relative_checkpoints"
    assert data["lp_status"] in {
        "feasible",
        "single_candidate",
        "all_over_budget",
        "no_candidates",
        None,
    }
    if data["lp_weights"] is not None:
        assert all(isinstance(key, str) for key in data["lp_weights"])
        assert all(isinstance(value, float | int) for value in data["lp_weights"].values())
    if data["routing_estimated_cost_usd"] is not None:
        assert data["routing_estimated_cost_usd"] >= 0.0
    if data["physical_cost_usd"] is not None:
        assert data["physical_cost_usd"] >= 0.0


def _record_payload(record: PerRequestRecord) -> dict[str, Any]:
    return {field: getattr(record, field) for field in CANONICAL_ROUTEWISE_REQUEST_FIELDS}


def _prod_payload(api_log_row: dict[str, Any]) -> dict[str, Any]:
    routewise = api_log_row["metadata"]["routewise"]
    status = Status.SUCCESS if api_log_row["status_code"] < 400 else Status.ERROR
    return {
        "request_id": api_log_row["request_id"],
        "policy": routewise["policy"],
        "model": api_log_row["model_id"],
        "source": "prod",
        "timestamp_sec": api_log_row["timestamp"],
        "prompt_tokens": api_log_row["prompt_tokens"],
        "completion_tokens_actual": api_log_row["completion_tokens"],
        "primary_provider": routewise["primary_provider"],
        "primary_tier": routewise["primary_tier"],
        "backup_provider": routewise["backup_provider"],
        "backup_tier": routewise["backup_tier"],
        "final_provider": api_log_row["provider"],
        "final_tier": routewise.get("final_tier"),
        "ttft_ms": api_log_row["ttft_ms"],
        "e2e_ms": api_log_row["latency_ms"],
        "slo_ms": routewise.get("slo_ms"),
        "slo_violated": bool(routewise.get("slo_violated", False)),
        "total_cost_usd": api_log_row["cost_usd"],
        "physical_cost_usd": api_log_row["upstream_cost_usd"],
        "routing_estimated_cost_usd": routewise["routing_estimated_cost_usd"],
        "hedge_triggered": routewise["hedge_triggered"],
        "hedge_delay_ms": routewise.get("hedge_delay_ms"),
        "hedge_winner": routewise["hedge_winner"],
        "hedge_algorithm": routewise["hedge_algorithm"],
        "hedge_schedule": routewise["hedge_schedule"],
        "lp_weights": routewise["lp_weights"],
        "lp_budget_usd": routewise["lp_budget_usd"],
        "lp_status": routewise["lp_status"],
        "status": status,
        "error_class": api_log_row["error"],
    }


def _sim_record() -> PerRequestRecord:
    return PerRequestRecord(
        request_id="sim-1",
        elapsed_sec=0.0,
        policy="routewise",
        prompt_tokens=100,
        completion_tokens_budget=50,
        completion_tokens_actual=42,
        primary_provider="chutes",
        primary_tier="quota",
        final_provider="chutes",
        final_tier="quota",
        model="qwen/qwen3-235b",
        source="sim",
        ttft_ms=420.0,
        e2e_ms=None,
        slo_ms=3000.0,
        slo_violated=False,
        total_cost_usd=0.001,
        primary_cost_usd=0.001,
        physical_cost_usd=None,
        routing_estimated_cost_usd=0.001,
        primary_routing_estimated_cost_usd=0.001,
        hedge_triggered=False,
        hedge_algorithm="probability_target",
        hedge_schedule="slo_relative_checkpoints",
        lp_weights={"chutes": 0.8, "openrouter": 0.2},
        lp_budget_usd=0.0015,
        lp_status="feasible",
    )


def _real_eval_record() -> PerRequestRecord:
    return PerRequestRecord(
        request_id="real-1",
        elapsed_sec=1.0,
        policy="routewise",
        prompt_tokens=100,
        completion_tokens_budget=50,
        completion_tokens_actual=42,
        primary_provider="chutes",
        primary_tier="quota",
        final_provider="openrouter",
        final_tier="api",
        backup_provider="openrouter",
        backup_tier="api",
        model="qwen/qwen3-235b",
        source="real_eval",
        timestamp_sec=1_795_000_000.0,
        ttft_ms=640.0,
        e2e_ms=900.0,
        slo_ms=3000.0,
        slo_violated=False,
        total_cost_usd=0.002,
        primary_cost_usd=0.001,
        backup_cost_usd=0.001,
        physical_cost_usd=0.0024,
        primary_physical_cost_usd=0.0012,
        backup_physical_cost_usd=0.0012,
        routing_estimated_cost_usd=0.0018,
        primary_routing_estimated_cost_usd=0.0008,
        backup_routing_estimated_cost_usd=0.001,
        hedge_triggered=True,
        hedge_delay_ms=750.0,
        hedge_winner="backup",
        hedge_algorithm="probability_target",
        hedge_schedule="slo_relative_checkpoints",
        lp_weights={"chutes": 0.8, "openrouter": 0.2},
        lp_budget_usd=0.0015,
        lp_status="feasible",
    )


def _prod_api_log_row() -> dict[str, Any]:
    """Representative hybridInference api_logs row with routewise metadata."""
    return {
        "request_id": "prod-1",
        "timestamp": 1_795_000_000.0,
        "model_id": "qwen/qwen3-235b",
        "provider": "chutes:https://chutes.example/v1",
        "ttft_ms": 610.0,
        "latency_ms": 1100.0,
        "prompt_tokens": 100,
        "completion_tokens": 42,
        "cost_usd": 0.002,
        "upstream_cost_usd": 0.0024,
        "status_code": 200,
        "error": None,
        "metadata": {
            "routewise": {
                "policy": "routewise",
                "primary_provider": "chutes:https://chutes.example/v1",
                "primary_tier": "quota",
                "backup_provider": None,
                "backup_tier": None,
                "hedge_triggered": False,
                "hedge_winner": None,
                "hedge_algorithm": "disabled",
                "hedge_schedule": None,
                "lp_weights": {"chutes:https://chutes.example/v1": 1.0},
                "lp_budget_usd": 0.0015,
                "lp_status": "feasible",
                "routing_estimated_cost_usd": None,
            }
        },
    }
