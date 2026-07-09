"""Tests for the public RouteWise core effective-cost API."""

from __future__ import annotations

import pytest

import routewise.core as core
from routewise.core.cost import (
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
    assert effective_cost(
        "concurrency",
        concurrency_utilization=0.75,
        L=0.0001,
        U=0.01,
    ) == 0.0
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
