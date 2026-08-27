"""Tests for the public RouteWise core effective-cost API."""

from __future__ import annotations

import math

import pytest

import llm_routewise.core as core
from llm_routewise.core.cost import (
    SCARCITY_CURVES,
    EffectiveCostTier,
    ScarcityCurve,
    concurrency_effective_cost,
    effective_cost,
    quota_effective_cost,
    scarcity_price,
)


def test_core_package_exports_public_cost_api() -> None:
    assert core.SCARCITY_CURVES is SCARCITY_CURVES
    assert core.EffectiveCostTier is EffectiveCostTier
    assert core.ScarcityCurve is ScarcityCurve
    assert core.concurrency_effective_cost is concurrency_effective_cost
    assert core.effective_cost is effective_cost
    assert core.quota_effective_cost is quota_effective_cost
    assert core.scarcity_price is scarcity_price


def test_effective_cost_api_tier_returns_request_cost() -> None:
    assert effective_cost(
        "api",
        request_cost_usd=0.00123,
        L=0.0001,
        U=0.01,
    ) == pytest.approx(0.00123)


def test_effective_cost_quota_tier_uses_routewise_quota_curve() -> None:
    expected = scarcity_price("exp_lu", 0.5, L=0.0001, U=0.01)

    assert effective_cost(
        "quota",
        quota_fraction_used=0.5,
        L=0.0001,
        U=0.01,
    ) == pytest.approx(expected)
    assert quota_effective_cost(0.5, L=0.0001, U=0.01) == pytest.approx(expected)


def test_missing_quota_snapshot_returns_zero() -> None:
    assert effective_cost("quota", quota_fraction_used=None, L=0.0001, U=0.01) == 0.0
    assert quota_effective_cost(None, L=0.0001, U=0.01) == 0.0


def test_effective_cost_concurrency_tier_is_zero_by_default() -> None:
    assert (
        effective_cost(
            "concurrency",
            concurrency_utilization=0.75,
            L=0.0001,
            U=0.01,
        )
        == 0.0
    )
    assert concurrency_effective_cost(0.75, L=0.0001, U=0.01) == 0.0


def test_effective_cost_concurrency_tier_supports_ablation_curve() -> None:
    assert effective_cost(
        "concurrency",
        concurrency_utilization=0.75,
        L=0.0001,
        U=0.01,
        concurrency_curve="util_linear_u",
    ) == pytest.approx(0.0075)


def test_unknown_effective_cost_tier_raises() -> None:
    with pytest.raises(ValueError, match="unknown effective-cost tier"):
        effective_cost("other", L=0.0001, U=0.01)  # type: ignore[arg-type]


def test_registered_curve_set_is_intentional() -> None:
    assert SCARCITY_CURVES == (
        "exp_lu",
        "linear_lu",
        "util_linear_u",
        "constant_0",
        "constant_l",
        "constant_u",
    )


def test_exp_lu_matches_current_quota_formula_and_high_clamp() -> None:
    assert scarcity_price("exp_lu", 0.0, L=1.0, U=1000.0) == pytest.approx(1.0)
    assert scarcity_price("exp_lu", 0.5, L=1.0, U=1000.0) == pytest.approx(math.sqrt(1000.0))
    assert scarcity_price("exp_lu", 1.0, L=1.0, U=1000.0) == pytest.approx(math.pow(1000.0, 0.9999))


def test_linear_lu_spans_exact_l_to_u() -> None:
    assert scarcity_price("linear_lu", 0.0, L=1.0, U=1000.0) == pytest.approx(1.0)
    assert scarcity_price("linear_lu", 0.5, L=1.0, U=1000.0) == pytest.approx(500.5)
    assert scarcity_price("linear_lu", 1.0, L=1.0, U=1000.0) == pytest.approx(1000.0)


def test_util_linear_u_matches_former_concurrency_formula() -> None:
    assert scarcity_price("util_linear_u", 0.0, L=0.0, U=10.0) == pytest.approx(0.0)
    assert scarcity_price("util_linear_u", 0.5, L=0.0, U=10.0) == pytest.approx(5.0)
    assert scarcity_price("util_linear_u", 1.0, L=0.0, U=10.0) == pytest.approx(10.0)


@pytest.mark.parametrize("x", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_constant_curves_ignore_scarcity(x: float) -> None:
    assert scarcity_price("constant_0", x, L=2.5, U=100.0) == pytest.approx(0.0)
    assert scarcity_price("constant_l", x, L=2.5, U=100.0) == pytest.approx(2.5)
    assert scarcity_price("constant_u", x, L=2.5, U=100.0) == pytest.approx(100.0)


def test_clamps_below_zero_to_zero_scarcity() -> None:
    assert scarcity_price("exp_lu", -0.5, L=1.0, U=1000.0) == pytest.approx(1.0)
    assert scarcity_price("linear_lu", -0.5, L=1.0, U=1000.0) == pytest.approx(1.0)
    assert scarcity_price("util_linear_u", -0.5, L=0.0, U=10.0) == pytest.approx(0.0)


def test_clamps_above_one_per_curve_semantics() -> None:
    assert scarcity_price("exp_lu", 1.5, L=1.0, U=1000.0) == pytest.approx(
        scarcity_price("exp_lu", 0.9999, L=1.0, U=1000.0)
    )
    assert scarcity_price("linear_lu", 1.5, L=1.0, U=1000.0) == pytest.approx(1000.0)
    assert scarcity_price("util_linear_u", 1.5, L=0.0, U=10.0) == pytest.approx(10.0)


@pytest.mark.parametrize(
    "curve",
    ["exp_lu", "linear_lu", "util_linear_u"],
)
def test_variable_curves_are_monotone(curve: str) -> None:
    values = [scarcity_price(curve, x, L=1.0, U=100.0) for x in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert values == sorted(values)


@pytest.mark.parametrize(
    ("L", "U"),
    [(0.0, 1.0), (-1.0, 1.0), (1.0, 1.0), (2.0, 1.0), (1.0, 0.0)],
)
@pytest.mark.parametrize("curve", ["exp_lu", "linear_lu"])
def test_lu_curves_reject_invalid_envelopes(curve: str, L: float, U: float) -> None:
    with pytest.raises(ValueError, match=rf"{curve} requires 0 < L < U"):
        scarcity_price(curve, 0.5, L=L, U=U)


@pytest.mark.parametrize("U", [0.0, -1.0])
@pytest.mark.parametrize("curve", ["util_linear_u", "constant_u"])
def test_u_only_curves_reject_nonpositive_u(curve: str, U: float) -> None:
    with pytest.raises(ValueError, match=rf"{curve} requires U > 0"):
        scarcity_price(curve, 0.5, L=1.0, U=U)


def test_constant_l_rejects_nonpositive_l() -> None:
    with pytest.raises(ValueError, match="constant_l requires L > 0"):
        scarcity_price("constant_l", 0.5, L=0.0, U=10.0)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_rejects_nonfinite_inputs(bad: float) -> None:
    with pytest.raises(ValueError, match="requires finite"):
        scarcity_price("linear_lu", bad, L=1.0, U=10.0)
    with pytest.raises(ValueError, match="requires finite"):
        scarcity_price("linear_lu", 0.5, L=bad, U=10.0)
    with pytest.raises(ValueError, match="requires finite"):
        scarcity_price("linear_lu", 0.5, L=1.0, U=bad)


def test_unknown_curve_raises() -> None:
    with pytest.raises(ValueError, match="unknown scarcity curve"):
        scarcity_price("not_a_curve", 0.5, L=1.0, U=10.0)  # type: ignore[arg-type]
