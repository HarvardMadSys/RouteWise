"""Unit tests for effective-cost ablation scarcity curves."""

from __future__ import annotations

import math

import pytest

from experiments.ablations.effective_cost.curves import SCARCITY_CURVES, scarcity_price


def test_registered_curve_set_is_intentional() -> None:
    assert SCARCITY_CURVES == (
        "exp_lu",
        "linear_lu",
        "legacy_linear_u",
        "constant_l",
        "constant_u",
    )


def test_exp_lu_matches_current_quota_formula_and_high_clamp() -> None:
    assert scarcity_price("exp_lu", 0.0, L=1.0, U=1000.0) == pytest.approx(1.0)
    assert scarcity_price("exp_lu", 0.5, L=1.0, U=1000.0) == pytest.approx(
        math.sqrt(1000.0)
    )
    assert scarcity_price("exp_lu", 1.0, L=1.0, U=1000.0) == pytest.approx(
        math.pow(1000.0, 0.9999)
    )


def test_linear_lu_spans_exact_L_to_U() -> None:
    assert scarcity_price("linear_lu", 0.0, L=1.0, U=1000.0) == pytest.approx(1.0)
    assert scarcity_price("linear_lu", 0.5, L=1.0, U=1000.0) == pytest.approx(500.5)
    assert scarcity_price("linear_lu", 1.0, L=1.0, U=1000.0) == pytest.approx(1000.0)


def test_legacy_linear_u_matches_current_concurrency_formula() -> None:
    assert scarcity_price("legacy_linear_u", 0.0, L=0.0, U=10.0) == pytest.approx(0.0)
    assert scarcity_price("legacy_linear_u", 0.5, L=0.0, U=10.0) == pytest.approx(5.0)
    assert scarcity_price("legacy_linear_u", 1.0, L=0.0, U=10.0) == pytest.approx(10.0)


@pytest.mark.parametrize("x", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_constant_curves_ignore_scarcity(x: float) -> None:
    assert scarcity_price("constant_l", x, L=2.5, U=100.0) == pytest.approx(2.5)
    assert scarcity_price("constant_u", x, L=2.5, U=100.0) == pytest.approx(100.0)


def test_clamps_below_zero_to_zero_scarcity() -> None:
    assert scarcity_price("exp_lu", -0.5, L=1.0, U=1000.0) == pytest.approx(1.0)
    assert scarcity_price("linear_lu", -0.5, L=1.0, U=1000.0) == pytest.approx(1.0)
    assert scarcity_price("legacy_linear_u", -0.5, L=0.0, U=10.0) == pytest.approx(0.0)


def test_clamps_above_one_per_curve_semantics() -> None:
    assert scarcity_price("exp_lu", 1.5, L=1.0, U=1000.0) == pytest.approx(
        scarcity_price("exp_lu", 0.9999, L=1.0, U=1000.0)
    )
    assert scarcity_price("linear_lu", 1.5, L=1.0, U=1000.0) == pytest.approx(1000.0)
    assert scarcity_price("legacy_linear_u", 1.5, L=0.0, U=10.0) == pytest.approx(10.0)


@pytest.mark.parametrize(
    "curve",
    ["exp_lu", "linear_lu", "legacy_linear_u"],
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
@pytest.mark.parametrize("curve", ["legacy_linear_u", "constant_u"])
def test_u_only_curves_reject_nonpositive_U(curve: str, U: float) -> None:
    with pytest.raises(ValueError, match=rf"{curve} requires U > 0"):
        scarcity_price(curve, 0.5, L=1.0, U=U)


def test_constant_l_rejects_nonpositive_L() -> None:
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
