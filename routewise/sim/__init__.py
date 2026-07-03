"""Canonical package entrypoint for the RouteWise simulator."""

from __future__ import annotations

_RUNNER_EXPORTS = {"POLICIES", "run_policy"}

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
    "WorkloadConfig",
    "HedgeDispatch",
    "SimulationResult",
}

__all__ = sorted(_RUNNER_EXPORTS | _SCHEMA_EXPORTS)


def __getattr__(name: str):
    """Resolve public exports lazily."""
    if name in _RUNNER_EXPORTS:
        from . import runner

        return getattr(runner, name)
    if name in _SCHEMA_EXPORTS:
        from routewise import schemas

        return getattr(schemas, name)
    raise AttributeError(f"module 'routewise.sim' has no attribute {name!r}")
