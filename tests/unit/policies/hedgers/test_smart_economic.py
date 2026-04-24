"""Canonical unit tests for the SMART_ECONOMIC hedger."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

from rwsim.policies.hedgers.smart_economic import (
    find_optimal_hedge_time_economic,
    smart_hedge_economic,
)
from rwsim.policies.latency_routers.online_lp import ProviderProfile
from rwsim.world.distributions import LogNormal


NOW = 1000.0
SLO_SEC = 3.0
DISPATCH_OVERHEAD_SEC = 0.05


def _profile_from_lognormal(
    provider: str,
    dist: LogNormal,
    *,
    sample_count: int = 10_000,
    now: float = NOW,
) -> ProviderProfile:
    """Build a deterministic empirical profile from lognormal quantiles."""
    normal = NormalDist()
    profile = ProviderProfile(provider=provider)
    for index in range(sample_count):
        quantile = (index + 0.5) / sample_count
        value_sec = math.exp(dist.mu + dist.sigma * normal.inv_cdf(quantile))
        profile.add_sample(now - 1.0, value_sec * 1000.0, None)
    return profile


def _profiles_from_lognormals(
    primary: LogNormal,
    backup: LogNormal,
) -> dict[str, ProviderProfile]:
    return {
        "primary": _profile_from_lognormal("primary", primary),
        "backup": _profile_from_lognormal("backup", backup),
    }


def _canonical_economic_product(
    primary: LogNormal,
    backup: LogNormal,
    *,
    elapsed_sec: float,
    slo_sec: float = SLO_SEC,
    dispatch_overhead_sec: float = DISPATCH_OVERHEAD_SEC,
) -> float:
    """Return the left side of ALGORITHMS.md Hedger eq. 5."""
    remaining = slo_sec - elapsed_sec - dispatch_overhead_sec
    if remaining <= 0.0:
        return 0.0

    primary_survival_at_slo = 1.0 - primary.cdf(slo_sec)
    primary_survival_at_elapsed = 1.0 - primary.cdf(elapsed_sec)
    p_viol = (
        1.0
        if primary_survival_at_elapsed < 1e-6
        else primary_survival_at_slo / primary_survival_at_elapsed
    )
    return p_viol * backup.cdf(remaining)


def _canonical_first_trigger(
    primary: LogNormal,
    backup: LogNormal,
    *,
    cost_ratio: float,
    resolution_sec: float,
    slo_sec: float = SLO_SEC,
    dispatch_overhead_sec: float = DISPATCH_OVERHEAD_SEC,
) -> float:
    """Return the earliest grid time satisfying ALGORITHMS.md Hedger eq. 5."""
    for hedge_time in np.arange(0.0, slo_sec, resolution_sec):
        product = _canonical_economic_product(
            primary,
            backup,
            elapsed_sec=float(hedge_time),
            slo_sec=slo_sec,
            dispatch_overhead_sec=dispatch_overhead_sec,
        )
        if product > cost_ratio:
            return float(hedge_time)
    return float("inf")


def test_eq5_decision_matches_lognormal_cdf_on_both_sides() -> None:
    primary = LogNormal(math.log(2.0), 0.5)
    backup = LogNormal(math.log(0.4), 0.2)
    profiles = _profiles_from_lognormals(primary, backup)
    elapsed_sec = 2.0

    product = _canonical_economic_product(primary, backup, elapsed_sec=elapsed_sec)
    assert 0.35 < product < 0.50

    assert smart_hedge_economic(
        "primary",
        "backup",
        elapsed_sec,
        SLO_SEC,
        profiles,
        NOW,
        cost_ratio=product * 0.5,
        dispatch_overhead_sec=DISPATCH_OVERHEAD_SEC,
    )
    assert not smart_hedge_economic(
        "primary",
        "backup",
        elapsed_sec,
        SLO_SEC,
        profiles,
        NOW,
        cost_ratio=min(0.99, product * 1.5),
        dispatch_overhead_sec=DISPATCH_OVERHEAD_SEC,
    )


def test_cost_ratio_monotonicity_against_eq5() -> None:
    primary = LogNormal(math.log(1.2), 0.75)
    backup = LogNormal(math.log(0.25), 0.3)
    profiles = _profiles_from_lognormals(primary, backup)
    elapsed_sec = 1.5

    product = _canonical_economic_product(primary, backup, elapsed_sec=elapsed_sec)
    cost_ratios = [product * factor for factor in (0.25, 0.75, 1.25, 2.0)]
    decisions = [
        smart_hedge_economic(
            "primary",
            "backup",
            elapsed_sec,
            SLO_SEC,
            profiles,
            NOW,
            cost_ratio=cost_ratio,
            dispatch_overhead_sec=DISPATCH_OVERHEAD_SEC,
        )
        for cost_ratio in cost_ratios
    ]

    assert decisions == [True, True, False, False]


def test_backup_usefulness_required_by_eq5() -> None:
    primary = LogNormal(math.log(2.0), 0.5)
    backup = LogNormal(math.log(10.0), 0.1)
    profiles = _profiles_from_lognormals(primary, backup)
    elapsed_sec = 2.0

    product = _canonical_economic_product(primary, backup, elapsed_sec=elapsed_sec)
    assert product < 1e-15

    assert not smart_hedge_economic(
        "primary",
        "backup",
        elapsed_sec,
        SLO_SEC,
        profiles,
        NOW,
        cost_ratio=0.01,
        dispatch_overhead_sec=DISPATCH_OVERHEAD_SEC,
    )


def test_no_time_left_blocks_hedge_even_for_fast_backup() -> None:
    primary = LogNormal(math.log(2.0), 0.5)
    backup = LogNormal(math.log(0.1), 0.1)
    profiles = _profiles_from_lognormals(primary, backup)
    elapsed_sec = SLO_SEC - DISPATCH_OVERHEAD_SEC + 0.001

    assert _canonical_economic_product(primary, backup, elapsed_sec=elapsed_sec) == 0.0
    assert not smart_hedge_economic(
        "primary",
        "backup",
        elapsed_sec,
        SLO_SEC,
        profiles,
        NOW,
        cost_ratio=0.01,
        dispatch_overhead_sec=DISPATCH_OVERHEAD_SEC,
    )


def test_find_optimal_hedge_time_is_earliest_eq5_grid_trigger() -> None:
    primary = LogNormal(math.log(2.0), 0.5)
    backup = LogNormal(math.log(0.4), 0.2)
    profiles = _profiles_from_lognormals(primary, backup)
    resolution_sec = 0.1
    cost_ratio = 0.25

    expected = _canonical_first_trigger(
        primary,
        backup,
        cost_ratio=cost_ratio,
        resolution_sec=resolution_sec,
    )
    observed = find_optimal_hedge_time_economic(
        "primary",
        "backup",
        profiles,
        NOW,
        SLO_SEC,
        cost_ratio=cost_ratio,
        dispatch_overhead_sec=DISPATCH_OVERHEAD_SEC,
        resolution_sec=resolution_sec,
    )

    assert expected == 1.3
    assert observed == expected
