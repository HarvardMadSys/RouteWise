"""Tests for trace-driven prefix-cache cost helpers."""

from __future__ import annotations

import pytest

from llm_routewise.capacity import ProviderTier
from llm_routewise.schemas import Request
from llm_routewise.sim.engine.simulator import Simulator
from llm_routewise.sim.engine.state import SimulationState
from llm_routewise.sim.policies.baselines import BaselinePolicy
from llm_routewise.sim.policies.prefix_cache import (
    cache_aware_marginal_cost,
    cached_input_tokens,
)
from llm_routewise.sim.policies.routewise import effective_cost
from llm_routewise.sim.world.distributions import Uniform
from llm_routewise.sim.world.providers import TieredProvider
from llm_routewise.sim.world.scenarios import ScenarioConfig


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
    cache_read_tokens: int | None = None,
) -> Request:
    metadata: dict[str, object] = {"session_id": session_id}
    if cache_read_tokens is not None:
        metadata["cache_read_tokens"] = cache_read_tokens
    return Request(
        id=request_id,
        timestamp=timestamp,
        request_tokens=request_tokens,
        response_tokens=response_tokens,
        total_tokens=request_tokens + response_tokens,
        metadata=metadata,
    )


def test_missing_trace_field_is_treated_as_cold_miss():
    provider = _provider("api")
    state = SimulationState.from_providers({provider.name: provider})
    state.metadata["prefix_cache_enabled"] = True
    request = _request(1, request_tokens=150, response_tokens=20)

    assert cached_input_tokens(provider, request, state) == 0
    assert cache_aware_marginal_cost(provider, request, state, now=0.0) == pytest.approx(250e-6)


def test_trace_observed_cache_read_tokens_drive_discount():
    provider = _provider("api")
    state = SimulationState.from_providers({provider.name: provider})
    state.metadata["prefix_cache_enabled"] = True
    request = _request(1, request_tokens=150, response_tokens=20, cache_read_tokens=80)

    assert cached_input_tokens(provider, request, state) == 80
    assert cache_aware_marginal_cost(provider, request, state, now=0.0) == pytest.approx(186e-6)


def test_trace_observed_cache_read_tokens_are_capped_and_zero_is_authoritative():
    provider = _provider("api")
    state = SimulationState.from_providers({provider.name: provider})
    state.metadata["prefix_cache_enabled"] = True
    capped = _request(1, request_tokens=150, response_tokens=20, cache_read_tokens=500)
    cold = _request(2, request_tokens=150, response_tokens=20, cache_read_tokens=0)

    assert cached_input_tokens(provider, capped, state) == 150
    assert cached_input_tokens(provider, cold, state) == 0


def test_cache_disabled_state_returns_cold():
    provider = _provider("api")
    state = SimulationState.from_providers({provider.name: provider})
    request = _request(1, request_tokens=100, response_tokens=20, cache_read_tokens=90)

    assert cached_input_tokens(provider, request, state) == 0
    assert cache_aware_marginal_cost(provider, request, state, now=0.0) == pytest.approx(200e-6)


def test_routewise_effective_cost_uses_trace_reported_cache_hit():
    provider = _provider("api")
    state = SimulationState.from_providers({provider.name: provider})
    state.metadata["prefix_cache_enabled"] = True
    request = _request(1, request_tokens=100, response_tokens=20, cache_read_tokens=60)

    assert effective_cost(
        provider,
        request,
        0.0,
        U=1.0,
        L=0.001,
        state=state,
    ) == pytest.approx(152e-6)


def test_simulator_bills_cached_input_from_trace_field():
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
        _request(
            2,
            request_tokens=150,
            response_tokens=10,
            timestamp=1.0,
            cache_read_tokens=120,
        ),
    ]

    run = Simulator(scenario, seed=1).run(
        requests,
        BaselinePolicy("greedy_cost"),
        policy_name="greedy_cost",
    )

    assert run.records[0].metadata["primary_cached_input_tokens"] == 0
    assert run.records[1].metadata["primary_cached_input_tokens"] == 120
