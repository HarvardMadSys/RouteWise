"""Tests for provider input/output token pricing."""

from __future__ import annotations

import pytest

from rwsim.schemas import Request
from rwsim.world.capacity import ProviderTier, QuotaState
from rwsim.world.distributions import Uniform
from rwsim.world.providers import TieredProvider


def _provider(**kwargs) -> TieredProvider:
    return TieredProvider(
        name="api",
        cost_per_token=kwargs.pop("cost_per_token", 1e-6),
        ttft_dist=Uniform(1.0, 2.0),
        tps_dist=Uniform(1.0, 2.0),
        tier=kwargs.pop("tier", ProviderTier.S_A),
        **kwargs,
    )


def test_provider_uses_split_input_output_prices_for_requests():
    provider = _provider(input_cost_per_token=1e-6, output_cost_per_token=5e-6)
    request = Request(id=1, timestamp=0.0, request_tokens=100, response_tokens=20, total_tokens=120)

    assert provider.marginal_cost_for_request(request, 0.0) == pytest.approx(200e-6)


def test_provider_applies_cached_input_price_to_cached_tokens_only():
    provider = _provider(
        input_cost_per_token=1e-6,
        output_cost_per_token=5e-6,
        cached_input_cost_per_token=0.2e-6,
    )
    request = Request(id=1, timestamp=0.0, request_tokens=100, response_tokens=20, total_tokens=120)

    assert provider.marginal_cost_for_request(
        request,
        0.0,
        cached_input_tokens=60,
    ) == pytest.approx(152e-6)


def test_provider_preserves_legacy_blended_price_without_split_prices():
    provider = _provider(cost_per_token=2e-6)
    request = Request(id=1, timestamp=0.0, request_tokens=100, response_tokens=20, total_tokens=120)

    assert provider.marginal_cost_for_request(request, 0.0) == pytest.approx(240e-6)
    assert provider.marginal_cost(120, 0.0) == pytest.approx(240e-6)


def test_provider_ignores_cached_tokens_for_legacy_blended_price():
    provider = _provider(cost_per_token=2e-6, cached_input_cost_per_token=0.2e-6)
    request = Request(id=1, timestamp=0.0, request_tokens=100, response_tokens=20, total_tokens=120)

    assert provider.marginal_cost_for_request(
        request,
        0.0,
        cached_input_tokens=60,
    ) == pytest.approx(240e-6)


def test_subscription_providers_have_zero_marginal_request_cost():
    provider = _provider(
        input_cost_per_token=1e-6,
        output_cost_per_token=5e-6,
        tier=ProviderTier.S_Q,
    )
    request = Request(id=1, timestamp=0.0, request_tokens=100, response_tokens=20, total_tokens=120)

    assert provider.marginal_cost_for_request(request, 0.0) == 0.0


def test_quota_reset_can_anchor_to_trace_start():
    quota = QuotaState(size=1, window_sec=10.0)

    quota.reset(window_start=5.0)
    quota.charge(14.0)

    assert not quota.can_admit(14.0)
    assert quota.can_admit(15.0)
