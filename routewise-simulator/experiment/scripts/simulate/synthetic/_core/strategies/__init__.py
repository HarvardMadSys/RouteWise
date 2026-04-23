"""Strategy registry for the synthetic simulator."""

from __future__ import annotations

from .baseline import BASELINE_STRATEGIES
from .lp import LP_STRATEGIES
from .tiered import TIERED_STRATEGIES
from .v2 import V2_STRATEGIES

LATENCY_STRATEGIES = [
    "cheapest_fixed",
    "fastest_fixed",
    "round_robin",
    "lp_mix",
    "lp_hedge",
    "lp_explorer",
    "lp_explorer_no_probe",
    "v2_only",
    "v2_p50_hedge",
    "v2_explorer",
    "v2_explorer_no_probe",
    "oracle_per_window",
]

STRATEGY_REGISTRY = {
    **BASELINE_STRATEGIES,
    **LP_STRATEGIES,
    **V2_STRATEGIES,
    **TIERED_STRATEGIES,
}


def run_registered_strategy(
    scenario,
    requests,
    strategy: str,
    seed: int = 42,
):
    """Dispatch one strategy through the shared registry."""
    try:
        strategy_fn = STRATEGY_REGISTRY[strategy]
    except KeyError as exc:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. Choose from {sorted(STRATEGY_REGISTRY)}"
        ) from exc
    return strategy_fn(scenario, requests, seed=seed)


__all__ = [
    "LATENCY_STRATEGIES",
    "STRATEGY_REGISTRY",
    "TIERED_STRATEGIES",
    "run_registered_strategy",
]
