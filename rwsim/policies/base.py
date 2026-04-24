"""Base interfaces for composable policy stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from rwsim.engine.state import SimulationState
from rwsim.schemas import Request, RoutingDecision


@dataclass
class PolicyStageContext:
    """Mutable metadata shared across policy stages for one request."""

    request: Request
    state: SimulationState
    estimates: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ValueEstimator(Protocol):
    """Predict request value or token cost before routing."""

    def estimate(self, context: PolicyStageContext) -> Mapping[str, Any]:
        """Return stage estimates for the current request."""
        ...


class CostRouter(Protocol):
    """Choose a primary provider from price and capacity state."""

    def route(self, context: PolicyStageContext) -> RoutingDecision:
        """Return the initial routing decision."""
        ...


class LatencyRouter(Protocol):
    """Filter or refine the primary provider using latency state."""

    def refine(
        self,
        decision: RoutingDecision,
        context: PolicyStageContext,
    ) -> RoutingDecision:
        """Return a latency-adjusted routing decision."""
        ...


class Hedger(Protocol):
    """Attach optional backup routing behavior."""

    def hedge(
        self,
        decision: RoutingDecision,
        context: PolicyStageContext,
    ) -> RoutingDecision:
        """Return a decision with hedge metadata if applicable."""
        ...


@dataclass
class PipelinePolicy:
    """Composable policy made from value, cost, latency, and hedge stages."""

    cost_router: CostRouter
    value_estimator: ValueEstimator | None = None
    latency_router: LatencyRouter | None = None
    hedger: Hedger | None = None

    def route(self, request: Request, state: SimulationState) -> RoutingDecision:
        """Run the policy pipeline for one request."""
        context = PolicyStageContext(request=request, state=state)
        if self.value_estimator is not None:
            context.estimates.update(dict(self.value_estimator.estimate(context)))

        decision = self.cost_router.route(context)

        if self.latency_router is not None:
            decision = self.latency_router.refine(decision, context)

        if self.hedger is not None:
            decision = self.hedger.hedge(decision, context)

        return decision


__all__ = [
    "CostRouter",
    "Hedger",
    "LatencyRouter",
    "PipelinePolicy",
    "PolicyStageContext",
    "ValueEstimator",
]
