"""Public RouteWise core APIs.

This package is intentionally limited to pure algorithm helpers and shared
data contracts. Simulator, real-eval, and production harnesses own mutable
state and side effects.
"""

from rwsim.core.hedging import (
    DISPATCH_OVERHEAD_MS,
    HEDGE_SUCCESS_TARGET,
    BackupCandidate,
    combined_success_probability,
    has_feasible_backup,
    hedge_checkpoints_for_slo,
    latest_safe_hedge_delay_sec,
    select_probability_backup,
)
from rwsim.core.lp import (
    LP_EPS,
    BudgetLPCandidate,
    BudgetLPResult,
    cost_tiebroken_objective,
    normalize_weights,
    solve_budget_lp,
    solve_simplex_lp,
)

__all__ = [
    "DISPATCH_OVERHEAD_MS",
    "HEDGE_SUCCESS_TARGET",
    "LP_EPS",
    "BackupCandidate",
    "BudgetLPCandidate",
    "BudgetLPResult",
    "combined_success_probability",
    "cost_tiebroken_objective",
    "has_feasible_backup",
    "hedge_checkpoints_for_slo",
    "latest_safe_hedge_delay_sec",
    "normalize_weights",
    "select_probability_backup",
    "solve_budget_lp",
    "solve_simplex_lp",
]
