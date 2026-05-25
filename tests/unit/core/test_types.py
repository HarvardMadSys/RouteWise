"""Public RouteWise core type contracts."""

from __future__ import annotations

import routewise.core as public_core
from rwsim import schemas
from rwsim.core.types import HedgeDispatch, RoutingDecision


def test_routewise_core_exports_decision_types() -> None:
    assert public_core.RoutingDecision is RoutingDecision
    assert public_core.HedgeDispatch is HedgeDispatch


def test_rwsim_schemas_reexports_public_decision_types() -> None:
    assert schemas.RoutingDecision is RoutingDecision
    assert schemas.HedgeDispatch is HedgeDispatch


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
