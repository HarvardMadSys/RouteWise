"""Flat policy interface for the RouteWise simulator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from routewise.schemas import HedgeDispatch, Request, RoutingDecision, RoutingOutcome

if TYPE_CHECKING:
    from routewise.sim.engine.state import SimulationState


class Policy(Protocol):
    """A routing policy decides, learns, and may re-check in-flight hedges."""

    def route(self, request: Request, state: SimulationState) -> RoutingDecision:
        """Return the initial routing decision for one request."""
        ...

    def tick(
        self,
        request: Request,
        decision: RoutingDecision,
        elapsed: float,
        state: SimulationState,
    ) -> HedgeDispatch | None:
        """Optionally dispatch a hedge at an in-flight checkpoint."""
        ...

    def observe(
        self,
        request: Request,
        decision: RoutingDecision,
        outcome: RoutingOutcome,
    ) -> None:
        """Update policy-owned learning state after execution."""
        ...


class NoOpTickMixin:
    """Default no-op implementation for policies without in-flight actions."""

    def tick(
        self,
        request: Request,
        decision: RoutingDecision,
        elapsed: float,
        state: SimulationState,
    ) -> HedgeDispatch | None:
        del request, decision, elapsed, state
        return None


class NoOpObserveMixin:
    """Default no-op implementation for stateless policies."""

    def observe(
        self,
        request: Request,
        decision: RoutingDecision,
        outcome: RoutingOutcome,
    ) -> None:
        del request, decision, outcome


__all__ = ["NoOpObserveMixin", "NoOpTickMixin", "Policy"]
