"""Canonical runner exports for the synthetic simulator."""

from experiment.scripts.simulate.synthetic._core.strategies import (
    LATENCY_STRATEGIES,
    TIERED_STRATEGIES,
    run_registered_strategy,
)

__all__ = [
    "LATENCY_STRATEGIES",
    "TIERED_STRATEGIES",
    "run_registered_strategy",
]
