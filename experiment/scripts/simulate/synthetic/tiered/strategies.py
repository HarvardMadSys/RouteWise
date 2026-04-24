"""Legacy compatibility wrapper for tiered strategy runners.

The canonical implementation lives in :mod:`rwsim.strategies.tiered_impl`.
"""

from rwsim.strategies.tiered_impl import (
    BackupSelectionMethod,
    HedgingParams,
    HedgingStrategy,
    SmartHedger,
    StrategyRun,
    TIERED_STRATEGIES,
    V2Router,
    _joint_select_p50band,
    _joint_select_slo_safe,
    _lognormal_p95,
    _pick_cross_tier_backup,
    _pick_tier,
    _provider_p95_at,
    _reset_all_state,
    _run_joint,
    _run_two_layer,
    _sample_service_time,
    _sample_ttft,
    run_tiered_strategy,
)

__all__ = [
    "BackupSelectionMethod",
    "HedgingParams",
    "HedgingStrategy",
    "SmartHedger",
    "StrategyRun",
    "TIERED_STRATEGIES",
    "V2Router",
    "_joint_select_p50band",
    "_joint_select_slo_safe",
    "_lognormal_p95",
    "_pick_cross_tier_backup",
    "_pick_tier",
    "_provider_p95_at",
    "_reset_all_state",
    "_run_joint",
    "_run_two_layer",
    "_sample_service_time",
    "_sample_ttft",
    "run_tiered_strategy",
]
