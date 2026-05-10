"""Tests for provider-local prefix-cache cost helpers."""

from __future__ import annotations

import pytest

from rwsim.engine.simulator import Simulator
from rwsim.engine.state import SimulationState
from rwsim.policies.baselines import BaselinePolicy
from rwsim.policies.prefix_cache import (
    cache_aware_marginal_cost,
    cached_input_tokens,
    record_prefix_cache_dispatch,
)
from rwsim.policies.routewise import effective_cost
from rwsim.schemas import Request
from rwsim.world.capacity import ProviderTier
from rwsim.world.distributions import Uniform
from rwsim.world.providers import TieredProvider
from rwsim.world.scenarios import ScenarioConfig


def _provider(
    name: str,
    *,
    input_cost: float = 1e-6,
    output_cost: float = 5e-6,
    cached_input_cost: float | None = 0.2e-6,
    p50_ms: float = 100.0,
) -> TieredProvider:
    return TieredProvider(
        name=name,
        cost_per_token=input_cost,
        input_cost_per_token=input_cost,
        output_cost_per_token=output_cost,
        cached_input_cost_per_token=cached_input_cost,
        ttft_dist=Uniform(p50_ms, p50_ms + 1.0),
        tps_dist=Uniform(1000.0, 1001.0),
        tier=ProviderTier.S_A,
    )


def _request(
    request_id: int,
    *,
    request_tokens: int,
    response_tokens: int,
    session_id: str = "s1",
    timestamp: float = 0.0,
) -> Request:
    return Request(
        id=request_id,
        timestamp=timestamp,
        request_tokens=request_tokens,
        response_tokens=response_tokens,
        total_tokens=request_tokens + response_tokens,
        metadata={"session_id": session_id},
    )


def test_cached_input_tokens_are_provider_local_and_length_based():
    left = _provider("left")
    right = _provider("right")
    state = SimulationState.from_providers({left.name: left, right.name: right})
    state.metadata["prefix_cache_enabled"] = True
    first = _request(1, request_tokens=100, response_tokens=20)
    second = _request(2, request_tokens=150, response_tokens=10)

    assert cached_input_tokens(left, second, state) == 0

    record_prefix_cache_dispatch(left, first, state, response_tokens=20)

    assert state.provider_prefix_cache[left.name]["s1"] == 120
    assert cached_input_tokens(left, second, state) == 120
    assert cached_input_tokens(right, second, state) == 0


def test_cache_disabled_state_returns_cold_and_does_not_update():
    provider = _provider("api")
    state = SimulationState.from_providers({provider.name: provider})
    request = _request(1, request_tokens=100, response_tokens=20)

    record_prefix_cache_dispatch(provider, request, state, response_tokens=20)

    assert state.provider_prefix_cache == {}
    assert cached_input_tokens(provider, request, state) == 0
    assert cache_aware_marginal_cost(provider, request, state, now=0.0) == pytest.approx(200e-6)


def test_routewise_effective_cost_uses_cached_api_cost_when_state_is_provided():
    provider = _provider("api")
    state = SimulationState.from_providers({provider.name: provider})
    state.metadata["prefix_cache_enabled"] = True
    state.provider_prefix_cache[provider.name] = {"s1": 60}
    request = _request(1, request_tokens=100, response_tokens=20)

    assert effective_cost(
        provider,
        request,
        0.0,
        U=1.0,
        L=0.001,
        state=state,
    ) == pytest.approx(152e-6)


def test_greedy_cost_can_choose_cached_provider_over_cheaper_cold_provider():
    cold = _provider("cold", input_cost=1e-6, output_cost=5e-6, cached_input_cost=0.2e-6)
    cached = _provider(
        "cached",
        input_cost=4e-6,
        output_cost=20e-6,
        cached_input_cost=0.8e-6,
    )
    state = SimulationState.from_providers({cold.name: cold, cached.name: cached})
    state.metadata["prefix_cache_enabled"] = True
    state.provider_prefix_cache[cached.name] = {"s1": 1000}
    request = _request(1, request_tokens=1000, response_tokens=0)

    decision = BaselinePolicy("greedy_cost").route(request, state)

    assert decision.primary_provider == "cached"


def test_simulator_bills_cached_input_after_prior_dispatch():
    provider = _provider("api")
    scenario = ScenarioConfig(
        name="prefix-cache",
        description="prefix cache billing test",
        providers=[provider],
        primary_slo_ms=2000.0,
        metadata={"prefix_cache_enabled": True},
    )
    requests = [
        _request(1, request_tokens=100, response_tokens=20, timestamp=0.0),
        _request(2, request_tokens=150, response_tokens=10, timestamp=1.0),
    ]

    run = Simulator(scenario, seed=1).run(
        requests,
        BaselinePolicy("greedy_cost"),
        policy_name="greedy_cost",
    )

    assert run.total_cost_usd() == pytest.approx(304e-6)
    assert run.records[0].metadata["primary_cached_input_tokens"] == 0
    assert run.records[1].metadata["primary_cached_input_tokens"] == 120
