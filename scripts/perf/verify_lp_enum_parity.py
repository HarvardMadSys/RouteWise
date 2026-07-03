"""Parity check: RouteWise public budget LP vs scipy.optimize.linprog.

Runs both solvers on the same LP instances drawn from realistic
RouteWise parameters and compares objective value, expected cost,
and weight vector. This is a manual parity/performance smoke test for
routewise.core.lp.solve_budget_lp.

Usage:
    uv run python scripts/perf/verify_lp_enum_parity.py
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import linprog

from routewise.core.lp import (
    BudgetLPCandidate,
    cost_tiebroken_objective,
    solve_budget_lp,
)


def _reference_linprog(
    objective: list[float],
    upper_constraint: list[float],
    upper_bound: float,
) -> tuple[bool, np.ndarray | None]:
    """Old solver path — kept verbatim from pre-swap routewise.py."""
    n = len(objective)
    result = linprog(
        c=objective,
        A_ub=[upper_constraint],
        b_ub=[upper_bound],
        A_eq=[np.ones(n)],
        b_eq=[1.0],
        bounds=[(0.0, 1.0) for _ in range(n)],
        method="highs",
    )
    if not result.success:
        return False, None
    return True, result.x


def _make_routewise_lp(
    rng: np.random.Generator,
    n_providers: int = 3,
) -> tuple[list[float], list[float], float]:
    """One LP instance shaped like RouteWise §1.1 cost-layer routing."""
    # Cost ratios in the (1, 2, 4) ballpark, USD per million tokens.
    base_costs = np.sort(rng.uniform(0.5e-6, 5e-6, size=n_providers))
    # Latency objectives near 300 ms with realistic spread.
    latencies = rng.uniform(150.0, 600.0, size=n_providers)
    # alpha sweep value — pick uniformly to cover the full sweep.
    p_value = float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0]))
    c_min = float(base_costs.min())
    c_max = float(base_costs.max())
    budget = c_min + p_value * (c_max - c_min)
    objective = cost_tiebroken_objective(latencies.tolist(), base_costs.tolist())
    return objective, base_costs.tolist(), budget


def _make_edge_cases() -> list[tuple[list[float], list[float], float]]:
    """Hand-rolled corner cases."""
    cases: list[tuple[list[float], list[float], float]] = []

    # Two providers with identical cost (degenerate budget edge).
    cases.append(
        (
            cost_tiebroken_objective([300.0, 200.0, 250.0], [1e-6, 1e-6, 4e-6]),
            [1e-6, 1e-6, 4e-6],
            1e-6,
        )
    )

    # Two providers with identical latency objective (forces cost tiebreak).
    cases.append(
        (
            cost_tiebroken_objective([300.0, 300.0, 200.0], [1e-6, 2e-6, 4e-6]),
            [1e-6, 2e-6, 4e-6],
            1.5e-6,
        )
    )

    # All providers identical.
    cases.append(
        (
            cost_tiebroken_objective([300.0, 300.0, 300.0], [2e-6, 2e-6, 2e-6]),
            [2e-6, 2e-6, 2e-6],
            2e-6,
        )
    )

    # Budget exactly equals c_min (p=0).
    cases.append(
        (
            cost_tiebroken_objective([500.0, 300.0, 200.0], [1e-6, 2e-6, 4e-6]),
            [1e-6, 2e-6, 4e-6],
            1e-6,
        )
    )

    # Budget exactly equals c_max (p=1).
    cases.append(
        (
            cost_tiebroken_objective([500.0, 300.0, 200.0], [1e-6, 2e-6, 4e-6]),
            [1e-6, 2e-6, 4e-6],
            4e-6,
        )
    )

    # Cheap provider also fastest — budget irrelevant.
    cases.append(
        (
            cost_tiebroken_objective([100.0, 300.0, 500.0], [1e-6, 2e-6, 4e-6]),
            [1e-6, 2e-6, 4e-6],
            2e-6,
        )
    )

    # Expensive provider fastest, mid budget — typical mix scenario.
    cases.append(
        (
            cost_tiebroken_objective([500.0, 300.0, 150.0], [1e-6, 2e-6, 4e-6]),
            [1e-6, 2e-6, 4e-6],
            2.5e-6,
        )
    )

    # 5-provider variant.
    cases.append(
        (
            cost_tiebroken_objective(
                [500.0, 400.0, 300.0, 200.0, 150.0],
                [1e-6, 1.5e-6, 2e-6, 3e-6, 4e-6],
            ),
            [1e-6, 1.5e-6, 2e-6, 3e-6, 4e-6],
            2e-6,
        )
    )

    return cases


def _check(
    name: str,
    objective: list[float],
    upper_constraint: list[float],
    upper_bound: float,
) -> tuple[bool, str]:
    names = [f"p{idx}" for idx in range(len(objective))]
    enum = solve_budget_lp(
        [
            BudgetLPCandidate(name, objective=obj, effective_cost=cost)
            for name, obj, cost in zip(names, objective, upper_constraint, strict=True)
        ],
        budget=upper_bound,
    )
    enum_ok = enum.feasible
    enum_w = np.asarray([enum.weights.get(name, 0.0) for name in names])
    ref_ok, ref_w = _reference_linprog(objective, upper_constraint, upper_bound)

    if enum_ok != ref_ok:
        return False, f"{name}: feasibility mismatch enum={enum_ok} ref={ref_ok}"
    if not enum_ok:
        return True, f"{name}: both infeasible"

    obj_arr = np.asarray(objective)
    cost_arr = np.asarray(upper_constraint)
    enum_obj = float(np.dot(obj_arr, enum_w))
    ref_obj = float(np.dot(obj_arr, ref_w))
    enum_cost = float(np.dot(cost_arr, enum_w))
    ref_cost = float(np.dot(cost_arr, ref_w))

    obj_gap = abs(enum_obj - ref_obj) / (abs(ref_obj) + 1e-12)
    cost_diff = abs(enum_cost - ref_cost) / (abs(ref_cost) + 1e-18)

    # Correctness checks: objective must match (LP optimum is unique up to
    # weight degeneracy at multiple optima) and enum must respect the budget.
    # cost_diff is informational only — distinct optima can have different
    # expected costs while sharing the optimal objective value.
    if obj_gap > 1e-6:
        return False, (
            f"{name}: objective mismatch enum={enum_obj:.6f} "
            f"ref={ref_obj:.6f} rel_gap={obj_gap:.2e} "
            f"weights enum={enum_w} ref={ref_w}"
        )
    if enum_cost > upper_bound + 1e-9:
        return False, (
            f"{name}: enum violates budget enum_cost={enum_cost:.6e} "
            f"budget={upper_bound:.6e}"
        )
    return True, (
        f"{name}: obj_rel={obj_gap:.2e} cost_diff={cost_diff:.2e} "
        f"enum_w={np.round(enum_w, 4)} ref_w={np.round(ref_w, 4)}"
    )


def main() -> None:
    rng = np.random.default_rng(2026)
    failures: list[str] = []
    passes = 0

    print("=" * 70)
    print("Edge cases:")
    for idx, (obj, cost, bnd) in enumerate(_make_edge_cases()):
        ok, msg = _check(f"edge_{idx}", obj, cost, bnd)
        print(("  PASS  " if ok else "  FAIL  ") + msg)
        if ok:
            passes += 1
        else:
            failures.append(msg)

    print()
    print("Random instances (3-provider):")
    for idx in range(50):
        obj, cost, bnd = _make_routewise_lp(rng, n_providers=3)
        ok, msg = _check(f"r3_{idx:02d}", obj, cost, bnd)
        if ok:
            passes += 1
        else:
            print("  FAIL  " + msg)
            failures.append(msg)
    print(f"  3-provider: {50 - len([f for f in failures if 'r3_' in f])}/50 passed")

    print()
    print("Random instances (5-provider):")
    for idx in range(50):
        obj, cost, bnd = _make_routewise_lp(rng, n_providers=5)
        ok, msg = _check(f"r5_{idx:02d}", obj, cost, bnd)
        if ok:
            passes += 1
        else:
            print("  FAIL  " + msg)
            failures.append(msg)
    print(f"  5-provider: {50 - len([f for f in failures if 'r5_' in f])}/50 passed")

    print()
    print("Performance bench (3-provider, 10000 calls):")
    rng = np.random.default_rng(7)
    obj, cost, bnd = _make_routewise_lp(rng, n_providers=3)
    n_iter = 10_000

    t0 = time.perf_counter()
    for _ in range(n_iter):
        solve_budget_lp(
            [
                BudgetLPCandidate(f"p{idx}", objective=value, effective_cost=cost[idx])
                for idx, value in enumerate(obj)
            ],
            budget=bnd,
        )
    enum_us = (time.perf_counter() - t0) * 1e6 / n_iter

    t0 = time.perf_counter()
    for _ in range(n_iter):
        _reference_linprog(obj, cost, bnd)
    linprog_us = (time.perf_counter() - t0) * 1e6 / n_iter

    print(f"  enum     {enum_us:8.2f} us/call")
    print(f"  linprog  {linprog_us:8.2f} us/call")
    print(f"  speedup  {linprog_us / max(enum_us, 1e-9):8.1f}x")

    print()
    print("=" * 70)
    if failures:
        print(f"FAILED: {len(failures)} mismatches, {passes} passes")
        raise SystemExit(1)
    print(f"OK: {passes} parity checks all passed")


if __name__ == "__main__":
    main()
