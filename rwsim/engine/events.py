"""Engine event records used for debugging and metrics."""

from __future__ import annotations

from dataclasses import dataclass

from rwsim.schemas import Request, RoutingDecision, RoutingOutcome


@dataclass(frozen=True)
class RequestEvent:
    """A request entered the simulation loop."""

    request: Request


@dataclass(frozen=True)
class DecisionEvent:
    """A policy emitted a routing decision."""

    request: Request
    decision: RoutingDecision


@dataclass(frozen=True)
class OutcomeEvent:
    """A routing decision was executed."""

    request: Request
    decision: RoutingDecision
    outcome: RoutingOutcome


__all__ = ["DecisionEvent", "OutcomeEvent", "RequestEvent"]
