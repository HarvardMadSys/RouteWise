"""Tests for the public RouteWise core LP API."""

from __future__ import annotations

import random

import pytest

import rwsim.core as core
from rwsim.core.lp import (
    BudgetLPCandidate,
    BudgetLPResult,
    cost_tiebroken_objective,
    solve_budget_lp,
)


def _dot(left: list[float], weights: dict[str, float], names: list[str]) -> float:
    return sum(value * weights.get(names[idx], 0.0) for idx, value in enumerate(left))


def test_core_package_exports_public_lp_api() -> None:
    assert core.BudgetLPCandidate is BudgetLPCandidate
    assert core.BudgetLPResult is BudgetLPResult
    assert core.cost_tiebroken_objective is cost_tiebroken_objective
    assert core.solve_budget_lp is solve_budget_lp
    assert not hasattr(core, "normalize_weights")
    assert not hasattr(core, "solve_simplex_lp")


def test_solve_budget_lp_selects_best_pure_candidate() -> None:
    result = solve_budget_lp(
        [
            BudgetLPCandidate("slow_cheap", objective=300.0, effective_cost=1.0),
            BudgetLPCandidate("fast_mid", objective=100.0, effective_cost=2.0),
            BudgetLPCandidate("slow_expensive", objective=500.0, effective_cost=3.0),
        ],
        budget=2.0,
    )

    assert result.feasible
    assert result.weights == {"fast_mid": 1.0}


def test_solve_budget_lp_uses_budget_boundary_mix() -> None:
    result = solve_budget_lp(
        [
            BudgetLPCandidate("fast_expensive", objective=100.0, effective_cost=3.0),
            BudgetLPCandidate("slow_cheap", objective=300.0, effective_cost=1.0),
        ],
        budget=2.0,
    )

    assert result.feasible
    assert result.weights == pytest.approx(
        {"fast_expensive": 0.5, "slow_cheap": 0.5}
    )


def test_solve_budget_lp_skips_equal_cost_mix_and_uses_best_pure() -> None:
    result = solve_budget_lp(
        [
            BudgetLPCandidate("slow", objective=300.0, effective_cost=2.0),
            BudgetLPCandidate("fast", objective=100.0, effective_cost=2.0),
        ],
        budget=2.0,
    )

    assert result.feasible
    assert result.weights == {"fast": 1.0}


def test_solve_budget_lp_treats_exact_budget_cost_as_feasible() -> None:
    result = solve_budget_lp(
        [
            BudgetLPCandidate("slow_cheap", objective=300.0, effective_cost=1.0),
            BudgetLPCandidate("fast_exact", objective=100.0, effective_cost=2.0),
        ],
        budget=2.0,
    )

    assert result.feasible
    assert result.weights == {"fast_exact": 1.0}


def test_solve_budget_lp_reports_infeasible_when_all_candidates_exceed_budget() -> None:
    result = solve_budget_lp(
        [
            BudgetLPCandidate("fast", objective=100.0, effective_cost=3.0),
            BudgetLPCandidate("slow", objective=300.0, effective_cost=4.0),
        ],
        budget=2.0,
    )

    assert not result.feasible
    assert result.weights == {}


def test_solve_budget_lp_rejects_duplicate_candidate_names() -> None:
    result = solve_budget_lp(
        [
            BudgetLPCandidate("same", objective=100.0, effective_cost=1.0),
            BudgetLPCandidate("same", objective=300.0, effective_cost=2.0),
        ],
        budget=2.0,
    )

    assert not result.feasible
    assert result.weights == {}
    assert result.status == "invalid"


@pytest.mark.parametrize(
    ("candidates", "budget"),
    [
        ([], 1.0),
        ([BudgetLPCandidate("bad_objective", objective=float("nan"), effective_cost=1.0)], 2.0),
        ([BudgetLPCandidate("bad_cost", objective=100.0, effective_cost=float("inf"))], 2.0),
        ([BudgetLPCandidate("bad_budget", objective=100.0, effective_cost=1.0)], float("inf")),
    ],
)
def test_solve_budget_lp_rejects_invalid_inputs(
    candidates: list[BudgetLPCandidate],
    budget: float,
) -> None:
    result = solve_budget_lp(
        candidates,
        budget=budget,
    )

    assert not result.feasible
    assert result.weights == {}


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


def test_solve_budget_lp_matches_scipy_objective_on_random_small_cases() -> None:
    linprog = pytest.importorskip("scipy.optimize").linprog
    rng = random.Random(7)

    for _ in range(100):
        n = rng.randint(2, 6)
        names = [f"p{idx}" for idx in range(n)]
        objective = [rng.uniform(10.0, 1000.0) for _ in range(n)]
        costs = [rng.uniform(0.1, 10.0) for _ in range(n)]
        budget = rng.uniform(min(costs), max(costs))

        lp_result = solve_budget_lp(
            [
                BudgetLPCandidate(name, objective=obj, effective_cost=cost)
                for name, obj, cost in zip(names, objective, costs, strict=True)
            ],
            budget=budget,
        )
        assert lp_result.feasible

        scipy_result = linprog(
            c=objective,
            A_ub=[costs],
            b_ub=[budget],
            A_eq=[[1.0] * n],
            b_eq=[1.0],
            bounds=[(0.0, 1.0)] * n,
            method="highs",
        )
        assert scipy_result.success
        assert _dot(objective, lp_result.weights, names) == pytest.approx(
            float(scipy_result.fun),
            abs=1e-7,
        )
        assert _dot(costs, lp_result.weights, names) <= budget + 1e-7
        assert sum(lp_result.weights.values()) == pytest.approx(1.0)
