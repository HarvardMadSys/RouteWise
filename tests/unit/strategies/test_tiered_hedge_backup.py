"""Unit tests for tiered hedge backup selection."""

from __future__ import annotations

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
