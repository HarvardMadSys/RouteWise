"""Shared simulation loop target.

This module defines the ownership boundary. Some current strategies still run
their own loops; migrated strategies should move to this engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from rwsim.engine.state import SimulationState
from rwsim.schemas import Request, RoutingDecision, RoutingOutcome, SimulationResult


class Policy(Protocol):
    """Policy interface: choose a route, do not execute it."""

    def route(self, request: Request, state: SimulationState) -> RoutingDecision:
        """Return a routing decision for one request."""
        ...


class Executor(Protocol):
    """Execution interface owned by the engine."""

    def execute(
        self,
        decision: RoutingDecision,
        request: Request,
        state: SimulationState,
    ) -> RoutingOutcome:
        """Execute a decision and return the realized outcome."""
        ...


class MetricsRecorder(Protocol):
    """Metrics sink for per-request records."""

    def record(
        self,
        request: Request,
        decision: RoutingDecision,
        outcome: RoutingOutcome,
        state: SimulationState,
    ) -> None:
        """Record one request result."""
        ...

    def result(self) -> SimulationResult:
        """Return the aggregate simulation result."""
        ...


@dataclass
class Simulator:
    """Minimal shared request loop for migrated policies."""

    executor: Executor
    metrics: MetricsRecorder

    def run(
        self,
        requests: Iterable[Request],
        policy: Policy,
        state: SimulationState | None = None,
    ) -> SimulationResult:
        """Run one policy over one request stream."""
        sim_state = state or SimulationState()
        for request in requests:
            sim_state.current_time = float(request.timestamp)
            decision = policy.route(request, sim_state)
            outcome = self.executor.execute(decision, request, sim_state)
            self.metrics.record(request, decision, outcome, sim_state)
        return self.metrics.result()


__all__ = ["Executor", "MetricsRecorder", "Policy", "Simulator"]
