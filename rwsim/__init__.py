"""Canonical package entrypoint for the RouteWise simulator.

Strategy exports are loaded lazily so dependency-light modules such as
``rwsim.schemas`` can be imported before the scientific stack is installed.
"""

from __future__ import annotations

_RUNNER_EXPORTS = {
    "LATENCY_STRATEGIES",
    "TIERED_STRATEGIES",
    "run_registered_strategy",
}

_SCHEMA_EXPORTS = {
    "DistributionConfig",
    "MetricRecord",
    "ProviderConfig",
    "ProviderTier",
    "Request",
    "RoutingDecision",
    "RoutingOutcome",
    "ScenarioConfig",
    "ShiftEvent",
    "SimulationResult",
    "WorkloadConfig",
}

__all__ = sorted(_RUNNER_EXPORTS | _SCHEMA_EXPORTS)


def __getattr__(name: str):
    """Resolve public exports lazily."""
    if name in _RUNNER_EXPORTS:
        from . import runner

        return getattr(runner, name)
    if name in _SCHEMA_EXPORTS:
        from . import schemas

        return getattr(schemas, name)
    raise AttributeError(f"module 'rwsim' has no attribute {name!r}")
