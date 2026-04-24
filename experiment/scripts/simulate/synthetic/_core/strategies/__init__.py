"""Legacy compatibility exports for the strategy registry."""

from rwsim.strategies.registry import (
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
