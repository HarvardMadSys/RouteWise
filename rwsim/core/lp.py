"""Shared RouteWise simplex LP solver.

The RouteWise body router solves a small linear program:

    minimize    objective · weights
    subject to  cost · weights <= budget
                sum(weights) = 1
                weights >= 0

With one budget inequality plus the simplex constraints, an optimum is either
a pure provider or a two-provider mixture on the budget boundary. Enumerating
those candidates is exact for this LP and avoids a scipy runtime dependency in
policy code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

LP_EPS: float = 1e-9
_COST_TIEBREAK_MS: float = 1e-3


@dataclass(frozen=True)
class BudgetLPCandidate:
    """Provider snapshot for the public named LP API."""

    name: str
    objective: float
    effective_cost: float
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BudgetLPResult:
    """Result from solving the RouteWise budget LP."""

    feasible: bool
    weights: dict[str, float]
    budget: float
    objective: float | None = None
    expected_cost: float | None = None
    status: str = "infeasible"


def solve_simplex_lp(
    objective: Sequence[float],
    *,
    upper_constraint: Sequence[float],
    upper_bound: float,
    eps: float = LP_EPS,
) -> tuple[bool, tuple[float, ...] | None]:
    """Solve the RouteWise simplex LP and return the optimal vector.

    This low-level API preserves the shape used by the historical SIM and
    REAL-EVAL helpers. Prefer ``solve_budget_lp`` for new named-provider code.
    """
    n = len(objective)
    if n == 0 or n != len(upper_constraint):
        return False, None
    if not math.isfinite(float(upper_bound)):
        return False, None

    obj = tuple(float(value) for value in objective)
    costs = tuple(float(value) for value in upper_constraint)
    if any(not math.isfinite(value) for value in obj):
        return False, None
    if any(not math.isfinite(value) for value in costs):
        return False, None

    best_vector: tuple[float, ...] | None = None
    best_key: tuple[float, float, int, tuple[float, ...]] | None = None

    def consider(vector: tuple[float, ...]) -> None:
        nonlocal best_key, best_vector
        expected_cost = _dot(costs, vector)
        if expected_cost > float(upper_bound) + eps:
            return
        value = _dot(obj, vector)
        support = sum(1 for part in vector if part > eps)
        rounded = tuple(round(float(part), 12) for part in vector)
        key = (value, expected_cost, support, rounded)
        if best_key is None or key < best_key:
            best_key = key
            best_vector = vector

    for index, cost in enumerate(costs):
        if cost <= float(upper_bound) + eps:
            consider(tuple(1.0 if position == index else 0.0 for position in range(n)))

    for left in range(n):
        left_cost = costs[left]
        for right in range(left + 1, n):
            right_cost = costs[right]
            span = left_cost - right_cost
            if abs(span) <= eps:
                continue
            left_weight = (float(upper_bound) - right_cost) / span
            right_weight = 1.0 - left_weight
            if left_weight < -eps or right_weight < -eps:
                continue
            left_weight = min(max(left_weight, 0.0), 1.0)
            right_weight = min(max(right_weight, 0.0), 1.0)
            vector = tuple(
                left_weight
                if position == left
                else right_weight
                if position == right
                else 0.0
                for position in range(n)
            )
            consider(vector)

    if best_vector is None:
        return False, None
    return True, best_vector


def solve_budget_lp(
    candidates: Sequence[BudgetLPCandidate],
    *,
    budget: float,
    eps: float = LP_EPS,
) -> BudgetLPResult:
    """Solve the RouteWise budget LP for named provider candidates."""
    names = [candidate.name for candidate in candidates]
    objective = [candidate.objective for candidate in candidates]
    costs = [candidate.effective_cost for candidate in candidates]
    success, vector = solve_simplex_lp(
        objective,
        upper_constraint=costs,
        upper_bound=budget,
        eps=eps,
    )
    if not success or vector is None:
        return BudgetLPResult(
            feasible=False,
            weights={},
            budget=float(budget),
        )

    weights = normalize_weights(names, vector, eps=eps)
    if not weights:
        return BudgetLPResult(
            feasible=False,
            weights={},
            budget=float(budget),
        )

    return BudgetLPResult(
        feasible=True,
        weights=weights,
        budget=float(budget),
        objective=_dot(tuple(objective), vector),
        expected_cost=_dot(tuple(costs), vector),
        status="feasible",
    )


def cost_tiebroken_objective(
    latency_objective_ms: Sequence[float],
    effective_costs: Sequence[float],
    *,
    eps: float = LP_EPS,
) -> list[float]:
    """Prefer lower effective cost when LP latency objectives are equal."""
    if len(latency_objective_ms) != len(effective_costs):
        raise ValueError("latency objective and cost arrays must have the same length")
    if len(latency_objective_ms) == 0:
        return []

    latencies = [float(value) for value in latency_objective_ms]
    costs = [float(value) for value in effective_costs]
    cost_min = min(costs)
    cost_span = max(costs) - cost_min
    if cost_span <= eps:
        return latencies

    return [
        latency + _COST_TIEBREAK_MS * ((cost - cost_min) / cost_span)
        for latency, cost in zip(latencies, costs, strict=True)
    ]


def normalize_weights(
    names: Sequence[str],
    vector: Sequence[float],
    *,
    eps: float = LP_EPS,
) -> dict[str, float]:
    """Convert an LP vector into a normalized provider weight map."""
    weights = {
        name: float(vector[idx])
        for idx, name in enumerate(names)
        if float(vector[idx]) > eps
    }
    total = sum(weights.values())
    if total <= 0.0:
        return {}
    return {name: value / total for name, value in weights.items()}


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return float(sum(float(a) * float(b) for a, b in zip(left, right, strict=True)))
