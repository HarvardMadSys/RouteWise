"""Flat policy contract tests."""

from __future__ import annotations

from experiments.simulation.eval_grid import LatencyFamily, ProviderSetup, make_scenario
from rwsim.engine.state import SimulationState
from rwsim.policies import build_policy
from rwsim.schemas import Request, RoutingDecision, RoutingOutcome


def test_baseline_policy_has_noop_tick_and_observe():
    scenario = make_scenario(ProviderSetup.COST_LATENCY_TRADEOFF, LatencyFamily.UNIFORM)
    state = SimulationState.from_providers({provider.name: provider for provider in scenario.providers})
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
    scenario = make_scenario(ProviderSetup.COST_LATENCY_TRADEOFF, LatencyFamily.UNIFORM)
    state = SimulationState.from_providers({provider.name: provider for provider in scenario.providers})
    request = Request(id=1, timestamp=0.0, request_tokens=100, response_tokens=50, total_tokens=150)
    policy = build_policy("routewise", seed=1)

    decision = policy.route(request, state)

    assert decision.primary_provider in {provider.name for provider in scenario.providers}
    assert decision.hedge_checkpoints
    assert decision.hedge_checkpoints == tuple(sorted(decision.hedge_checkpoints))
