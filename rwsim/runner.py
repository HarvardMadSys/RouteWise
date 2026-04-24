"""Canonical runner exports for the synthetic simulator."""

from rwsim.strategies.registry import (
    LATENCY_STRATEGIES,
    TIERED_STRATEGIES,
    run_registered_strategy,
)

__all__ = [
    "LATENCY_STRATEGIES",
    "TIERED_STRATEGIES",
    "run_registered_strategy",
]
