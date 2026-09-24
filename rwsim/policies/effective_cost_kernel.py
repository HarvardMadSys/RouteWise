"""Shared scarcity-price curves for RouteWise effective-cost routing.

This module is intentionally pure: it does not import provider, policy, or
simulator engine types. Callers extract the scarcity signal from their own
state objects, then delegate the curve math here.
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

SCARCITY_CURVES: tuple[ScarcityCurve, ...] = (
    "exp_lu",
    "linear_lu",
    "util_linear_u",
    "constant_l",
    "constant_u",
)

_EXP_HIGH_CLAMP = 0.9999


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
    "SCARCITY_CURVES",
    "ScarcityCurve",
    "scarcity_price",
]
