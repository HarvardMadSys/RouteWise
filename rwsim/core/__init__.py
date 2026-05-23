"""Public RouteWise core APIs.

This package is intentionally limited to pure algorithm helpers and shared
data contracts. Simulator, real-eval, and production harnesses own mutable
state and side effects.
"""

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
    "LP_EPS",
    "BudgetLPCandidate",
    "BudgetLPResult",
    "cost_tiebroken_objective",
    "normalize_weights",
    "solve_budget_lp",
    "solve_simplex_lp",
]
