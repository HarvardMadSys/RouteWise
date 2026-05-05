"""Flat policy contract tests."""

from __future__ import annotations

from rwsim.engine.state import SimulationState
from rwsim.policies import build_policy
from rwsim.policies.routewise import RouteWisePolicy
from rwsim.schemas import Request, RoutingDecision, RoutingOutcome
from rwsim.world.capacity import ProviderTier
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
    policy = build_policy("routewise", seed=1)

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
    policy = RouteWisePolicy(hedging=False, explorer=False, p=1.0, seed=7)

    decision = policy.route(request, state)

    assert decision.primary_provider == "cheap"
    assert decision.metadata["weights"] == {"cheap": 1.0}
