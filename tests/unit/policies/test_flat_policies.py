"""Flat policy contract tests."""

from __future__ import annotations

import pytest

from rwsim.engine.state import SimulationState
from rwsim.policies import build_policy
from rwsim.policies.routewise import RouteWisePolicy
from rwsim.schemas import Request, RoutingDecision, RoutingOutcome
from rwsim.world.capacity import ProviderTier, QuotaState
from rwsim.world.distributions import Uniform
from rwsim.world.providers import TieredProvider


def _cost_latency_tradeoff_providers() -> list[TieredProvider]:
    return [
        TieredProvider(
            name="cheap_slow",
            cost_per_token=1e-6,
            ttft_dist=Uniform(500.0, 1500.0),
            tps_dist=Uniform(100.0, 200.0),
            tier=ProviderTier.S_A,
        ),
        TieredProvider(
            name="mid",
            cost_per_token=2e-6,
            ttft_dist=Uniform(150.0, 450.0),
            tps_dist=Uniform(100.0, 200.0),
            tier=ProviderTier.S_A,
        ),
        TieredProvider(
            name="fast_expensive",
            cost_per_token=4e-6,
            ttft_dist=Uniform(50.0, 150.0),
            tps_dist=Uniform(100.0, 200.0),
            tier=ProviderTier.S_A,
        ),
    ]


def test_baseline_policy_has_noop_tick_and_observe():
    providers = _cost_latency_tradeoff_providers()
    state = SimulationState.from_providers({provider.name: provider for provider in providers})
    request = Request(id=1, timestamp=0.0, request_tokens=100, response_tokens=50, total_tokens=150)
    policy = build_policy("greedy_cost", seed=1)

    decision = policy.route(request, state)

    assert isinstance(decision, RoutingDecision)
    assert policy.tick(request, decision, 0.25, state) is None
    policy.observe(
        request,
        decision,
        RoutingOutcome(
            request_id=request.id,
            primary_provider=decision.primary_provider,
            final_provider=decision.primary_provider,
            ttft_ms=100.0,
            cost_usd=0.0,
        ),
    )


def test_routewise_declares_in_flight_hedge_checkpoints():
    providers = _cost_latency_tradeoff_providers()
    state = SimulationState.from_providers({provider.name: provider for provider in providers})
    request = Request(id=1, timestamp=0.0, request_tokens=100, response_tokens=50, total_tokens=150)
    policy = build_policy(
        "routewise",
        presets={
            "routewise": {
                "policy": "RouteWisePolicy",
                "params": {
                    "hedging": "probability_target",
                    "explorer": True,
                    "cost_envelope": (1e-6, 1e-3),
                },
            }
        },
        seed=1,
    )

    decision = policy.route(request, state)

    assert decision.primary_provider in {provider.name for provider in providers}
    assert decision.hedge_checkpoints
    assert decision.hedge_checkpoints == tuple(sorted(decision.hedge_checkpoints))


def test_routewise_uses_cost_tiebreak_when_latency_objective_is_equal():
    providers = [
        TieredProvider(
            name="expensive",
            cost_per_token=4e-6,
            ttft_dist=Uniform(150.0, 450.0),
            tps_dist=Uniform(100.0, 200.0),
            tier=ProviderTier.S_A,
        ),
        TieredProvider(
            name="cheap",
            cost_per_token=1e-6,
            ttft_dist=Uniform(150.0, 450.0),
            tps_dist=Uniform(100.0, 200.0),
            tier=ProviderTier.S_A,
        ),
        TieredProvider(
            name="mid",
            cost_per_token=2e-6,
            ttft_dist=Uniform(150.0, 450.0),
            tps_dist=Uniform(100.0, 200.0),
            tier=ProviderTier.S_A,
        ),
    ]
    state = SimulationState.from_providers({provider.name: provider for provider in providers})
    request = Request(id=1, timestamp=0.0, request_tokens=100, response_tokens=50, total_tokens=150)
    policy = RouteWisePolicy(
        hedging=False,
        explorer=False,
        p=1.0,
        seed=7,
        cost_envelope=(1e-6, 1e-3),
    )

    decision = policy.route(request, state)

    assert decision.primary_provider == "cheap"
    assert decision.metadata["weights"] == {"cheap": 1.0}


def test_routewise_requires_explicit_cost_envelope():
    with pytest.raises(ValueError, match="requires an explicit cost_envelope"):
        RouteWisePolicy(hedging=False, explorer=False, p=0.75, seed=7)


def test_routewise_fixed_cost_envelope_keeps_quota_price_request_independent():
    providers = _quota_and_api_providers()
    providers[0].quota.used = 5
    state = SimulationState.from_providers({provider.name: provider for provider in providers})
    short = Request(id=1, timestamp=0.0, request_tokens=1, response_tokens=1, total_tokens=2)
    long = Request(id=2, timestamp=0.0, request_tokens=1000, response_tokens=1000, total_tokens=2000)
    policy = RouteWisePolicy(
        hedging=False,
        explorer=False,
        p=0.0,
        seed=7,
        cost_envelope=(1e-4, 1e-2),
    )

    short_decision = policy.route(short, state)
    long_decision = policy.route(long, state)

    assert short_decision.metadata["L"] == long_decision.metadata["L"] == 1e-4
    assert short_decision.metadata["U"] == long_decision.metadata["U"] == 1e-2
    assert short_decision.metadata["c_eff"]["quota"] == long_decision.metadata["c_eff"]["quota"]
    assert short_decision.metadata["c_eff"]["api"] < long_decision.metadata["c_eff"]["api"]


def test_routewise_prefers_quota_for_high_value_request_with_fixed_envelope():
    providers = _quota_and_api_providers()
    state = SimulationState.from_providers({provider.name: provider for provider in providers})
    short = Request(id=1, timestamp=0.0, request_tokens=1, response_tokens=1, total_tokens=2)
    long = Request(id=2, timestamp=0.0, request_tokens=1000, response_tokens=1000, total_tokens=2000)
    policy = RouteWisePolicy(
        hedging=False,
        explorer=False,
        p=0.0,
        seed=7,
        cost_envelope=(1e-4, 1e-2),
    )

    short_decision = policy.route(short, state)
    long_decision = policy.route(long, state)

    assert short_decision.primary_provider == "api"
    assert long_decision.primary_provider == "quota"


def _quota_and_api_providers() -> list[TieredProvider]:
    return [
        TieredProvider(
            name="quota",
            cost_per_token=0.0,
            input_cost_per_token=0.0,
            output_cost_per_token=0.0,
            ttft_dist=Uniform(150.0, 450.0),
            tps_dist=Uniform(100.0, 200.0),
            tier=ProviderTier.S_Q,
            quota=QuotaState(size=10),
        ),
        TieredProvider(
            name="api",
            cost_per_token=1e-6,
            input_cost_per_token=1e-6,
            output_cost_per_token=5e-6,
            ttft_dist=Uniform(150.0, 450.0),
            tps_dist=Uniform(100.0, 200.0),
            tier=ProviderTier.S_A,
        ),
    ]
