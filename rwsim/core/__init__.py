"""Public RouteWise core APIs.

This package is intentionally limited to pure algorithm helpers and shared
data contracts. Simulator, real-eval, and production harnesses own mutable
state and side effects.
"""

from rwsim.core.cost import (
    ROUTEWISE_CONCURRENCY_CURVE,
    ROUTEWISE_QUOTA_CURVE,
    SCARCITY_CURVES,
    EffectiveCostTier,
    ScarcityCurve,
    concurrency_effective_cost,
    effective_cost,
    quota_effective_cost,
    scarcity_price,
)
from rwsim.core.hedging import (
    DISPATCH_OVERHEAD_MS,
    HEDGE_SUCCESS_TARGET,
    BackupCandidate,
    CheckpointBackupDispatch,
    CheckpointBackupSelector,
    combined_success_probability,
    has_feasible_backup,
    hedge_checkpoints_for_slo,
    select_probability_backup,
)
from rwsim.core.latency_profile import (
    DEFAULT_PROFILE_WINDOW_SEC,
    RollingLatencyProfile,
)
from rwsim.core.lp import (
    LP_EPS,
    BudgetLPCandidate,
    BudgetLPResult,
    cost_tiebroken_objective,
    solve_budget_lp,
)
from rwsim.core.types import HedgeDispatch, RoutingDecision

__all__ = [
    "DEFAULT_PROFILE_WINDOW_SEC",
    "DISPATCH_OVERHEAD_MS",
    "HEDGE_SUCCESS_TARGET",
    "LP_EPS",
    "ROUTEWISE_CONCURRENCY_CURVE",
    "ROUTEWISE_QUOTA_CURVE",
    "SCARCITY_CURVES",
    "BackupCandidate",
    "BudgetLPCandidate",
    "BudgetLPResult",
    "CheckpointBackupDispatch",
    "CheckpointBackupSelector",
    "EffectiveCostTier",
    "HedgeDispatch",
    "RollingLatencyProfile",
    "RoutingDecision",
    "ScarcityCurve",
    "combined_success_probability",
    "concurrency_effective_cost",
    "cost_tiebroken_objective",
    "effective_cost",
    "has_feasible_backup",
    "hedge_checkpoints_for_slo",
    "quota_effective_cost",
    "scarcity_price",
    "select_probability_backup",
    "solve_budget_lp",
]
