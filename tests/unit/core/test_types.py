"""Public RouteWise core type contracts."""

from __future__ import annotations

import llm_routewise.core as public_core
from llm_routewise.core.hedging import CheckpointBackupDispatch, CheckpointBackupSelector
from llm_routewise.core.types import HedgeDispatch, RoutingDecision


def test_routewise_core_exports_decision_types() -> None:
    assert public_core.RoutingDecision is RoutingDecision
    assert public_core.HedgeDispatch is HedgeDispatch


def test_routewise_core_exports_checkpoint_backup_contracts() -> None:
    assert public_core.CheckpointBackupDispatch is CheckpointBackupDispatch
    assert public_core.CheckpointBackupSelector is CheckpointBackupSelector

    released = False

    def release() -> None:
        nonlocal released
        released = True

    dispatch = public_core.CheckpointBackupDispatch(
        backup="backup",
        elapsed_sec=0.75,
        success_probability=0.99,
        release=release,
        metadata={"reason": "probability_target"},
    )

    assert dispatch.backup == "backup"
    assert dispatch.elapsed_sec == 0.75
    assert dispatch.success_probability == 0.99
    assert dispatch.metadata == {"reason": "probability_target"}
    assert dispatch.release is not None
    dispatch.release()
    assert released is True


def test_hybrid_inference_consumer_contract_remains_exported() -> None:
    """Keep the production router's current core imports available after rename."""
    expected = {
        "HEDGE_SUCCESS_TARGET",
        "BackupCandidate",
        "BudgetLPCandidate",
        "CheckpointBackupDispatch",
        "CheckpointBackupSelector",
        "combined_success_probability",
        "cost_tiebroken_objective",
        "hedge_checkpoints_for_slo",
        "quota_effective_cost",
        "select_probability_backup",
        "solve_budget_lp",
    }

    assert expected <= set(public_core.__all__)
    assert all(hasattr(public_core, name) for name in expected)


def test_routing_decision_exposes_canonical_and_legacy_checkpoint_names() -> None:
    decision = RoutingDecision(
        primary_provider="primary",
        hedge_checkpoints_sec=[0.5, 1.0],
        metadata={"policy": "routewise"},
    )

    assert decision.primary_provider == "primary"
    assert decision.hedge_checkpoints_sec == (0.5, 1.0)
    assert decision.hedge_checkpoints == (0.5, 1.0)
    assert decision.metadata == {"policy": "routewise"}


def test_routing_decision_accepts_legacy_checkpoint_keyword() -> None:
    decision = RoutingDecision(primary_provider="primary", hedge_checkpoints=(0.25,))

    assert decision.hedge_checkpoints_sec == (0.25,)
    assert decision.hedge_checkpoints == (0.25,)


def test_routing_decision_rejects_ambiguous_checkpoint_keywords() -> None:
    try:
        RoutingDecision(
            primary_provider="primary",
            hedge_checkpoints_sec=(0.25,),
            hedge_checkpoints=(0.5,),
        )
    except TypeError as exc:
        assert "pass only one" in str(exc)
    else:
        raise AssertionError("expected TypeError")


def test_hedge_dispatch_carries_backup_provider_and_metadata() -> None:
    dispatch = HedgeDispatch(
        backup_provider="backup",
        metadata={"reason": "probability_target"},
    )

    assert dispatch.backup_provider == "backup"
    assert dispatch.metadata == {"reason": "probability_target"}


def test_hedge_dispatch_preserves_legacy_positional_metadata() -> None:
    dispatch = HedgeDispatch("backup", {"reason": "probability_target"})

    assert dispatch.backup_provider == "backup"
    assert dispatch.metadata == {"reason": "probability_target"}
