"""Canonical strategy registry exports."""

from experiment.scripts.simulate.synthetic._core.strategies import (
    LATENCY_STRATEGIES,
    STRATEGY_REGISTRY,
    TIERED_STRATEGIES,
    run_registered_strategy,
)

__all__ = [
    "LATENCY_STRATEGIES",
    "STRATEGY_REGISTRY",
    "TIERED_STRATEGIES",
    "run_registered_strategy",
]
