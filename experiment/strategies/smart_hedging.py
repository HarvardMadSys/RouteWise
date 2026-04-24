"""Compatibility wrapper for hedger policy stages."""

from __future__ import annotations

from rwsim.policies.hedgers.smart_economic import (
    BackupSelectionMethod,
    HedgingParams,
    HedgingResult,
    HedgingStrategy,
    SmartHedger,
    compute_conditional_expectation,
    compute_expected_latency,
    find_optimal_hedge_time_economic,
    find_optimal_hedge_time_residual,
    find_optimal_hedge_time_survival,
    find_percentile_hedge_time,
    get_cdf_for_hedging,
    get_cheapest_viable,
    get_fastest_provider,
    get_lp_backup,
    get_survival_for_hedging,
    select_backup,
    smart_hedge_economic,
    smart_hedge_residual,
    smart_hedge_survival,
)

__all__ = [
    "BackupSelectionMethod",
    "HedgingParams",
    "HedgingResult",
    "HedgingStrategy",
    "SmartHedger",
    "compute_conditional_expectation",
    "compute_expected_latency",
    "find_optimal_hedge_time_economic",
    "find_optimal_hedge_time_residual",
    "find_optimal_hedge_time_survival",
    "find_percentile_hedge_time",
    "get_cdf_for_hedging",
    "get_cheapest_viable",
    "get_fastest_provider",
    "get_lp_backup",
    "get_survival_for_hedging",
    "select_backup",
    "smart_hedge_economic",
    "smart_hedge_residual",
    "smart_hedge_survival",
]
