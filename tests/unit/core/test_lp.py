"""Tests for the public RouteWise core LP API."""

from __future__ import annotations

import random

import pytest

from rwsim.core.lp import (
    BudgetLPCandidate,
    cost_tiebroken_objective,
    normalize_weights,
    solve_budget_lp,
    solve_simplex_lp,
)


def _dot(left: list[float], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def test_solve_simplex_lp_selects_best_pure_candidate() -> None:
    success, vector = solve_simplex_lp(
        [300.0, 100.0, 500.0],
        upper_constraint=[1.0, 2.0, 3.0],
        upper_bound=2.0,
    )

    assert success
    assert vector == (0.0, 1.0, 0.0)


def test_solve_simplex_lp_uses_budget_boundary_mix() -> None:
    success, vector = solve_simplex_lp(
        [100.0, 300.0],
        upper_constraint=[3.0, 1.0],
        upper_bound=2.0,
    )

    assert success
    assert vector == pytest.approx((0.5, 0.5))


def test_solve_simplex_lp_reports_infeasible_when_all_candidates_exceed_budget() -> None:
    success, vector = solve_simplex_lp(
        [100.0, 300.0],
        upper_constraint=[3.0, 4.0],
        upper_bound=2.0,
    )

    assert not success
    assert vector is None


def test_solve_budget_lp_returns_named_weights_and_values() -> None:
    result = solve_budget_lp(
        [
            BudgetLPCandidate("fast_expensive", objective=100.0, effective_cost=3.0),
            BudgetLPCandidate("slow_cheap", objective=300.0, effective_cost=1.0),
        ],
        budget=2.0,
    )

    assert result.feasible
    assert result.status == "feasible"
    assert result.weights == pytest.approx(
        {
            "fast_expensive": 0.5,
            "slow_cheap": 0.5,
        }
    )
    assert result.objective == pytest.approx(200.0)
    assert result.expected_cost == pytest.approx(2.0)


def test_cost_tiebroken_objective_prefers_lower_cost_for_equal_latency() -> None:
    objective = cost_tiebroken_objective(
        [100.0, 100.0],
        [10.0, 1.0],
    )

    assert objective[1] < objective[0]


def test_normalize_weights_drops_zero_support() -> None:
    assert normalize_weights(["a", "b", "c"], (0.0, 0.25, 0.25)) == {
        "b": 0.5,
        "c": 0.5,
    }


def test_solve_simplex_lp_matches_scipy_objective_on_random_small_cases() -> None:
    linprog = pytest.importorskip("scipy.optimize").linprog
    rng = random.Random(7)

    for _ in range(100):
        n = rng.randint(2, 6)
        objective = [rng.uniform(10.0, 1000.0) for _ in range(n)]
        costs = [rng.uniform(0.1, 10.0) for _ in range(n)]
        budget = rng.uniform(min(costs), max(costs))

        success, vector = solve_simplex_lp(
            objective,
            upper_constraint=costs,
            upper_bound=budget,
        )
        assert success
        assert vector is not None

        result = linprog(
            c=objective,
            A_ub=[costs],
            b_ub=[budget],
            A_eq=[[1.0] * n],
            b_eq=[1.0],
            bounds=[(0.0, 1.0)] * n,
            method="highs",
        )
        assert result.success
        assert _dot(objective, vector) == pytest.approx(float(result.fun), abs=1e-7)
        assert _dot(costs, vector) <= budget + 1e-7
        assert sum(vector) == pytest.approx(1.0)
