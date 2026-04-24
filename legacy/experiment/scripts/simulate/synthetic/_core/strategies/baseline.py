"""Legacy compatibility exports for baseline strategies."""

from rwsim.strategies.baseline import (
    BASELINE_STRATEGIES,
    run_cheapest_fixed,
    run_fastest_fixed,
    run_oracle_per_window,
    run_round_robin,
)

__all__ = [
    "BASELINE_STRATEGIES",
    "run_cheapest_fixed",
    "run_fastest_fixed",
    "run_oracle_per_window",
    "run_round_robin",
]
