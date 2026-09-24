"""Public RouteWise core APIs.

This package is intentionally limited to pure algorithm helpers and shared
data contracts. Simulator, real-eval, and production harnesses own mutable
state and side effects.
"""

from llm_routewise.core.beliefs import (
    DEFAULT_ERROR_PENALTY_MS,
    DEFAULT_UNPROFILED_PENALTY_MS,
    BeliefMode,
    LatencyBeliefs,
)
from llm_routewise.core.cost import (
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
from llm_routewise.core.hedging import (
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
from llm_routewise.core.latency_profile import (
    DEFAULT_PROFILE_WINDOW_SEC,
    RollingLatencyProfile,
)
from llm_routewise.core.lp import (
    LP_EPS,
    BudgetLPCandidate,
    BudgetLPResult,
    cost_tiebroken_objective,
    solve_budget_lp,
)
from llm_routewise.core.pricing import (
    cache_discounted_cost_per_million_usd,
    cache_discounted_token_cost_usd,
    capped_cached_input_tokens,
)
from llm_routewise.core.provider_view import ProviderView
from llm_routewise.core.router import (
    CheckpointResult,
    LPStatus,
    RouteResult,
    RouteWiseRouter,
    argmax_weight_sampler,
    latest_safe_dispatch_gate,
)
from llm_routewise.core.types import HedgeDispatch, RoutingDecision

__all__ = [
    "DEFAULT_ERROR_PENALTY_MS",
    "DEFAULT_PROFILE_WINDOW_SEC",
    "DEFAULT_UNPROFILED_PENALTY_MS",
    "DISPATCH_OVERHEAD_MS",
    "HEDGE_SUCCESS_TARGET",
    "LP_EPS",
    "ROUTEWISE_CONCURRENCY_CURVE",
    "ROUTEWISE_QUOTA_CURVE",
    "SCARCITY_CURVES",
    "BackupCandidate",
    "BeliefMode",
    "BudgetLPCandidate",
    "BudgetLPResult",
    "CheckpointBackupDispatch",
    "CheckpointBackupSelector",
    "CheckpointResult",
    "EffectiveCostTier",
    "HedgeDispatch",
    "LPStatus",
    "LatencyBeliefs",
    "ProviderView",
    "RollingLatencyProfile",
    "RouteResult",
    "RouteWiseRouter",
    "RoutingDecision",
    "ScarcityCurve",
    "argmax_weight_sampler",
    "cache_discounted_cost_per_million_usd",
    "cache_discounted_token_cost_usd",
    "capped_cached_input_tokens",
    "combined_success_probability",
    "concurrency_effective_cost",
    "cost_tiebroken_objective",
    "effective_cost",
    "has_feasible_backup",
    "hedge_checkpoints_for_slo",
    "latest_safe_dispatch_gate",
    "quota_effective_cost",
    "scarcity_price",
    "select_probability_backup",
    "solve_budget_lp",
]
