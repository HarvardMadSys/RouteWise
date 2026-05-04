"""Unit tests for tiered hedge backup selection."""

from __future__ import annotations

import numpy as np

from experiments.simulation.lp_budget_eval import (
    RecentViolationTracker,
    RollingLatencyProfile,
    _pick_probability_target_backup,
    _select_budget_body,
)
from rwsim.schemas import Request
from rwsim.strategies.tiered_impl import _pick_cross_tier_backup
from rwsim.world.capacity import ConcurrencyState, ProviderTier, QuotaState
from rwsim.world.distributions import LogNormal
from rwsim.world.providers import TieredProvider


NOW = 10.0
TTFT = LogNormal(0.0, 0.1)
TPS = LogNormal(4.0, 0.1)


def _api_provider(name: str) -> TieredProvider:
    return TieredProvider(
        name=name,
        cost_per_token=0.001,
        ttft_dist=TTFT,
        tps_dist=TPS,
        tier=ProviderTier.S_A,
    )


def _priced_api_provider(name: str, cost_per_token: float) -> TieredProvider:
    return TieredProvider(
        name=name,
        cost_per_token=cost_per_token,
        ttft_dist=TTFT,
        tps_dist=TPS,
        tier=ProviderTier.S_A,
    )


def _full_quota_provider(name: str) -> TieredProvider:
    return TieredProvider(
        name=name,
        cost_per_token=0.0,
        ttft_dist=TTFT,
        tps_dist=TPS,
        tier=ProviderTier.S_Q,
        quota=QuotaState(size=1, used=1),
    )


def _full_concurrency_provider(name: str) -> TieredProvider:
    concurrency = ConcurrencyState(limit=1)
    concurrency.admit(request_id=1, now=NOW, service_time_sec=100.0)
    return TieredProvider(
        name=name,
        cost_per_token=0.0,
        ttft_dist=TTFT,
        tps_dist=TPS,
        tier=ProviderTier.S_C,
        concurrency=concurrency,
    )


def _available_quota_provider(name: str) -> TieredProvider:
    return TieredProvider(
        name=name,
        cost_per_token=0.0,
        ttft_dist=TTFT,
        tps_dist=TPS,
        tier=ProviderTier.S_Q,
        quota=QuotaState(size=1, used=0),
    )


def test_cross_tier_backup_returns_none_when_only_unavailable_backups_exist() -> None:
    primary = _api_provider("api-primary")
    providers = [
        primary,
        _full_quota_provider("quota-full"),
        _full_concurrency_provider("concurrency-full"),
    ]

    assert _pick_cross_tier_backup(providers, primary, NOW) is None


def test_cross_tier_backup_ignores_unavailable_fastest_candidate() -> None:
    primary = _api_provider("api-primary")
    slow_available = _available_quota_provider("quota-available")
    providers = [
        primary,
        _full_quota_provider("quota-full-fast"),
        slow_available,
    ]

    assert _pick_cross_tier_backup(providers, primary, NOW) == slow_available


def test_probability_target_cross_tier_scope_rejects_same_tier_backup() -> None:
    primary = _api_provider("api-primary")
    same_tier_backup = _api_provider("api-backup")
    request = Request(id=1, timestamp=NOW, request_tokens=10, response_tokens=10, total_tokens=20)

    backup, mode, random_prob = _pick_probability_target_backup(
        [primary, same_tier_backup],
        primary,
        request,
        now=NOW,
        slo_ms=1000.0,
        rng=np.random.default_rng(0),
        violation_tracker=RecentViolationTracker(),
        allow_random_backup=False,
        backup_scope="cross_tier",
    )

    assert backup is None
    assert mode == "no_backup"
    assert random_prob == 0.0


def test_probability_target_default_scope_allows_same_tier_backup() -> None:
    primary = _api_provider("api-primary")
    same_tier_backup = _api_provider("api-backup")
    request = Request(id=1, timestamp=NOW, request_tokens=10, response_tokens=10, total_tokens=20)

    backup, mode, random_prob = _pick_probability_target_backup(
        [primary, same_tier_backup],
        primary,
        request,
        now=NOW,
        slo_ms=1000.0,
        rng=np.random.default_rng(0),
        violation_tracker=RecentViolationTracker(),
        allow_random_backup=False,
    )

    assert backup == same_tier_backup
    assert mode == "safe_cheapest"
    assert random_prob == 0.0


def test_rolling_profile_hides_future_samples_until_observation_time() -> None:
    profile = RollingLatencyProfile(window_sec=100.0)

    profile.add_sample(12.0, 120.0)
    profile.add_sample(10.0, 100.0)

    assert profile.values(11.0) == [100.0]
    assert sorted(profile.values(12.0)) == [100.0, 120.0]


def test_range_budget_uses_feasible_cost_envelope() -> None:
    providers = [
        _priced_api_provider("cheap", 0.001),
        _priced_api_provider("middle", 0.002),
        _priced_api_provider("expensive", 0.004),
    ]
    request = Request(id=1, timestamp=NOW, request_tokens=10, response_tokens=10, total_tokens=20)
    profiles = {
        provider.name: RollingLatencyProfile(window_sec=100.0)
        for provider in providers
    }

    outcome = _select_budget_body(
        providers,
        request,
        now=NOW,
        slo_ms=1000.0,
        U=1.0,
        L=1.0,
        profiles=profiles,
        variant="budget_range_p25",
    )

    costs = list(outcome.c_eff.values())
    expected_budget = min(costs) + 0.25 * (max(costs) - min(costs))
    assert np.isclose(outcome.budget_tau, expected_budget)
