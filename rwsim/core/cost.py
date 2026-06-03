"""Shared RouteWise effective-cost primitives.

This module is intentionally pure: it does not import provider, policy,
simulator, or real-eval types. Harnesses extract request cost and scarcity
signals from their own mutable state, then delegate the common math here.
"""

from __future__ import annotations

import math
from typing import Literal

ScarcityCurve = Literal[
    "exp_lu",
    "linear_lu",
    "util_linear_u",
    "constant_l",
    "constant_u",
]
EffectiveCostTier = Literal["api", "quota", "concurrency"]

SCARCITY_CURVES: tuple[ScarcityCurve, ...] = (
    "exp_lu",
    "linear_lu",
    "util_linear_u",
    "constant_l",
    "constant_u",
)

ROUTEWISE_QUOTA_CURVE: ScarcityCurve = "exp_lu"
ROUTEWISE_CONCURRENCY_CURVE: ScarcityCurve | None = None
_EXP_HIGH_CLAMP = 0.9999


def effective_cost(
    tier: EffectiveCostTier,
    *,
    request_cost_usd: float = 0.0,
    quota_fraction_used: float | None = None,
    concurrency_utilization: float | None = None,
    L: float,
    U: float,
    quota_curve: ScarcityCurve = ROUTEWISE_QUOTA_CURVE,
    concurrency_curve: ScarcityCurve | None = ROUTEWISE_CONCURRENCY_CURVE,
) -> float:
    """Return the RouteWise piecewise effective cost for one provider snapshot.

    ``tier`` is a normalized provider tier name. The caller owns all state reads:
    API request cost, quota usage, and concurrency utilization are scalar
    snapshots, not mutable provider objects.
    """
    if tier == "api":
        return float(request_cost_usd)
    if tier == "quota":
        return quota_effective_cost(
            quota_fraction_used,
            L=L,
            U=U,
            curve=quota_curve,
        )
    if tier == "concurrency":
        return concurrency_effective_cost(
            concurrency_utilization,
            L=L,
            U=U,
            curve=concurrency_curve,
        )
    raise ValueError(f"unknown effective-cost tier {tier!r}")


def quota_effective_cost(
    quota_fraction_used: float | None,
    *,
    L: float,
    U: float,
    curve: ScarcityCurve = ROUTEWISE_QUOTA_CURVE,
) -> float:
    """Return opportunity cost for a quota-backed provider snapshot."""
    if quota_fraction_used is None:
        return 0.0
    return scarcity_price(curve, quota_fraction_used, L=L, U=U)


def concurrency_effective_cost(
    concurrency_utilization: float | None,
    *,
    L: float,
    U: float,
    curve: ScarcityCurve | None = ROUTEWISE_CONCURRENCY_CURVE,
) -> float:
    """Return opportunity cost for a concurrency-backed provider snapshot.

    The main RouteWise path treats reusable concurrency slots as
    availability-only, so ``curve=None`` returns zero. Ablations can pass a
    scarcity curve to price utilization explicitly.
    """
    if curve is None:
        return 0.0
    if concurrency_utilization is None:
        return 0.0
    return scarcity_price(curve, concurrency_utilization, L=L, U=U)


def scarcity_price(
    curve: ScarcityCurve,
    x: float,
    *,
    L: float,
    U: float,
) -> float:
    """Return the shadow price for one scarcity fraction.

    ``x`` is the normalized scarcity signal: quota fraction used for S_Q or
    weighted concurrency utilization for S_C. The L/U envelope is measured in
    API-equivalent request dollars.
    """
    _validate_finite(x=x, L=L, U=U)
    if curve == "exp_lu":
        _validate_lu(curve, L=L, U=U)
        scarcity = _clamp(x, high=_EXP_HIGH_CLAMP)
        return L * math.pow(U / L, scarcity)
    if curve == "linear_lu":
        _validate_lu(curve, L=L, U=U)
        scarcity = _clamp(x)
        return L + scarcity * (U - L)
    if curve == "util_linear_u":
        _validate_positive("U", U, curve=curve)
        return U * _clamp(x)
    if curve == "constant_l":
        _validate_positive("L", L, curve=curve)
        return L
    if curve == "constant_u":
        _validate_positive("U", U, curve=curve)
        return U
    raise ValueError(f"unknown scarcity curve {curve!r}")


def _clamp(value: float, *, high: float = 1.0) -> float:
    return min(max(float(value), 0.0), high)


def _validate_finite(*, x: float, L: float, U: float) -> None:
    if not all(math.isfinite(float(value)) for value in (x, L, U)):
        raise ValueError(f"scarcity_price requires finite x, L, and U; got x={x}, L={L}, U={U}")


def _validate_lu(curve: str, *, L: float, U: float) -> None:
    if not (0.0 < L < U):
        raise ValueError(f"{curve} requires 0 < L < U; got L={L}, U={U}")


def _validate_positive(name: str, value: float, *, curve: str) -> None:
    if not value > 0.0:
        raise ValueError(f"{curve} requires {name} > 0; got {name}={value}")


__all__ = [
    "ROUTEWISE_CONCURRENCY_CURVE",
    "ROUTEWISE_QUOTA_CURVE",
    "SCARCITY_CURVES",
    "EffectiveCostTier",
    "ScarcityCurve",
    "concurrency_effective_cost",
    "effective_cost",
    "quota_effective_cost",
    "scarcity_price",
]
