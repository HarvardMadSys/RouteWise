"""Canonical unit tests for tiered cost-router primitives."""

from __future__ import annotations

import pytest
import math

from rwsim.policies.cost_routers.tiered import (
    cheapest_effective_provider,
    pick_available_tier,
    pick_two_layer_provider,
)
from rwsim.schemas import Request
from rwsim.world.capacity import ConcurrencyState, ProviderTier, QuotaState
from rwsim.world.distributions import LogNormal
from rwsim.world.providers import TieredProvider
from rwsim.world.shadow_price import (
    calibrate_envelopes,
    concurrency_shadow_price,
    quota_shadow_price,
)


NOW = 10.0
U = 1.0
L = 0.001
REQUEST = Request(id=1, timestamp=NOW, request_tokens=80, response_tokens=20, total_tokens=100)
TTFT = LogNormal(0.0, 0.1)
TPS = LogNormal(4.0, 0.1)


def _api_provider(name: str, cost_per_token: float) -> TieredProvider:
    return TieredProvider(
        name=name,
        cost_per_token=cost_per_token,
        ttft_dist=TTFT,
        tps_dist=TPS,
        tier=ProviderTier.S_A,
    )


def _quota_provider(
    name: str,
    *,
    used: int = 0,
    size: int = 10,
    ttft_dist: LogNormal = TTFT,
) -> TieredProvider:
    return TieredProvider(
        name=name,
        cost_per_token=0.0,
        ttft_dist=ttft_dist,
        tps_dist=TPS,
        tier=ProviderTier.S_Q,
        quota=QuotaState(size=size, used=used),
    )


def _concurrency_provider(
    name: str,
    *,
    active_slots: int = 0,
    limit: int = 10,
) -> TieredProvider:
    concurrency = ConcurrencyState(limit=limit)
    for request_id in range(active_slots):
        concurrency.admit(request_id=request_id, now=NOW, service_time_sec=100.0)
    return TieredProvider(
        name=name,
        cost_per_token=0.0,
        ttft_dist=TTFT,
        tps_dist=TPS,
        tier=ProviderTier.S_C,
        concurrency=concurrency,
    )


def test_shadow_price_endpoints() -> None:
    quota_empty = _quota_provider("quota-empty", used=0)
    quota_full = _quota_provider("quota-full", used=10)
    concurrency_empty = _concurrency_provider("concurrency-empty", active_slots=0)
    concurrency_full = _concurrency_provider("concurrency-full", active_slots=10)

    assert quota_shadow_price(quota_empty, NOW, U=U, L=L) == pytest.approx(L)
    assert quota_shadow_price(quota_full, NOW, U=U, L=L) == pytest.approx(U, rel=1e-3)
    assert concurrency_shadow_price(concurrency_empty, NOW, U=U) == pytest.approx(0.0)
    assert concurrency_shadow_price(concurrency_full, NOW, U=U) == pytest.approx(U)


def test_shadow_prices_are_monotone() -> None:
    quota_prices = [
        quota_shadow_price(_quota_provider(f"quota-{used}", used=used), NOW, U=U, L=L)
        for used in range(0, 11)
    ]
    concurrency_prices = [
        concurrency_shadow_price(
            _concurrency_provider(f"concurrency-{active}", active_slots=active),
            NOW,
            U=U,
        )
        for active in range(0, 11)
    ]

    assert quota_prices == sorted(quota_prices)
    assert concurrency_prices == sorted(concurrency_prices)


def test_cost_envelope_scaling_preserves_argmin_and_scales_costs() -> None:
    providers = [
        _api_provider("api-cheap", 0.001),
        _api_provider("api-expensive", 0.002),
        _quota_provider("quota", used=0),
        _concurrency_provider("concurrency", active_slots=5),
    ]
    L1, U1 = calibrate_envelopes(providers, typical_tokens=REQUEST.total_tokens)
    selected1, costs1 = cheapest_effective_provider(providers, REQUEST, NOW, U=U1, L=L1)

    scale = 7.3
    scaled_providers = [
        _api_provider("api-cheap", 0.001 * scale),
        _api_provider("api-expensive", 0.002 * scale),
        _quota_provider("quota", used=0),
        _concurrency_provider("concurrency", active_slots=5),
    ]
    L2, U2 = calibrate_envelopes(scaled_providers, typical_tokens=REQUEST.total_tokens)
    selected2, costs2 = cheapest_effective_provider(scaled_providers, REQUEST, NOW, U=U2, L=L2)

    assert selected1.name == selected2.name
    assert L2 == pytest.approx(L1 * scale)
    assert U2 == pytest.approx(U1 * scale)
    assert set(costs1) == set(costs2)
    for provider_name, cost in costs1.items():
        assert costs2[provider_name] == pytest.approx(cost * scale)


def test_cheapest_effective_provider_filters_infeasible_capacity() -> None:
    exhausted_quota = _quota_provider("quota-exhausted", used=1, size=1)
    saturated_concurrency = _concurrency_provider("concurrency-full", active_slots=1, limit=1)
    api = _api_provider("api", 0.01)

    selected, costs = cheapest_effective_provider(
        [exhausted_quota, saturated_concurrency, api],
        REQUEST,
        NOW,
        U=U,
        L=L,
    )

    assert selected.name == "api"
    assert costs == {"api": pytest.approx(api.marginal_cost(REQUEST.total_tokens, NOW))}


def test_two_layer_filters_unavailable_providers_inside_selected_tier() -> None:
    unavailable_fast_quota = _quota_provider(
        "unavailable-fast-quota",
        used=1,
        size=1,
        ttft_dist=LogNormal(math.log(50.0), 0.1),
    )
    available_slow_quota = _quota_provider(
        "available-slow-quota",
        used=0,
        size=1,
        ttft_dist=LogNormal(math.log(200.0), 0.1),
    )
    api = _api_provider("api", 0.01)

    providers = [unavailable_fast_quota, available_slow_quota, api]

    assert pick_available_tier(providers, NOW) == ProviderTier.S_Q
    assert pick_two_layer_provider(providers, NOW).name == "available-slow-quota"
