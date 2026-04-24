"""Tiered-provider extension to the synthetic simulator.

This package adds multi-tier (S_A, S_Q, S_C) routing capability so we can
evaluate the joint cross-tier routing proposal against the two-layer
baseline in a controlled environment.

Entry points:
    run_tiered_scenario(scenario, requests, seeds=None, strategies=None)
    make_tiered_scenarios() -> dict[str, TieredScenarioConfig]
"""

from .providers import (
    ConcurrencyState,
    ProviderTier,
    QuotaState,
    TieredProvider,
)
from .runner import run_tiered_scenario
from .scenarios import TieredScenarioConfig, make_tiered_scenarios
from .shadow_price import (
    calibrate_envelopes,
    concurrency_shadow_price,
    effective_cost,
    quota_shadow_price,
)
from .strategies import TIERED_STRATEGIES, StrategyRun, run_tiered_strategy

__all__ = [
    "ConcurrencyState",
    "ProviderTier",
    "QuotaState",
    "StrategyRun",
    "TIERED_STRATEGIES",
    "TieredProvider",
    "TieredScenarioConfig",
    "calibrate_envelopes",
    "concurrency_shadow_price",
    "effective_cost",
    "make_tiered_scenarios",
    "quota_shadow_price",
    "run_tiered_scenario",
    "run_tiered_strategy",
]
