"""Flat policy contract tests."""

from __future__ import annotations

import copy
from itertools import pairwise

import pytest

from rwsim.core.hedging import BackupCandidate, hedge_checkpoints_for_slo
from rwsim.core.lp import cost_tiebroken_objective, normalize_weights, solve_simplex_lp
from rwsim.engine.state import SimulationState
from rwsim.policies import build_policy, routewise as routewise_module
from rwsim.policies.routewise import (
    RollingLatencyProfile,
    RouteWisePolicy,
    _same_cost_shortcut_weights,
    concurrency_shadow_price,
)
from rwsim.schemas import Request, RoutingDecision, RoutingOutcome
from rwsim.world.capacity import ConcurrencyState, ProviderTier, QuotaState
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


def test_greedy_cost_ties_break_concurrency_before_quota():
    """When S_C and S_Q both have 0 marginal cost, prefer S_C (perishable).

    The slow concurrency provider must beat the fast quota provider — this is
    the joint-tier behaviour that distinguishes tier-aware greedy_cost from
    the older latency-tiebreak implementation.
    """
    providers = [
        TieredProvider(
            name="slow_concurrency",
            cost_per_token=0.0,
            input_cost_per_token=0.0,
            output_cost_per_token=0.0,
            ttft_dist=Uniform(900.0, 1100.0),
            tps_dist=Uniform(100.0, 200.0),
            tier=ProviderTier.S_C,
            concurrency=ConcurrencyState(limit=4),
        ),
        TieredProvider(
            name="fast_quota",
            cost_per_token=0.0,
            input_cost_per_token=0.0,
            output_cost_per_token=0.0,
            ttft_dist=Uniform(90.0, 110.0),
            tps_dist=Uniform(100.0, 200.0),
            tier=ProviderTier.S_Q,
            quota=QuotaState(size=10),
        ),
        TieredProvider(
            name="paid_api",
            cost_per_token=1e-6,
            input_cost_per_token=1e-6,
            output_cost_per_token=5e-6,
            ttft_dist=Uniform(50.0, 100.0),
            tps_dist=Uniform(100.0, 200.0),
            tier=ProviderTier.S_A,
        ),
    ]
    state = SimulationState.from_providers({provider.name: provider for provider in providers})
    request = Request(id=1, timestamp=0.0, request_tokens=100, response_tokens=50, total_tokens=150)
    policy = build_policy("greedy_cost", seed=1)

    decision = policy.route(request, state)

    assert decision.primary_provider == "slow_concurrency"


def test_greedy_cost_falls_back_to_quota_when_concurrency_saturated():
    """If S_C has no available slot, greedy_cost picks S_Q over the paid API."""
    concurrency = ConcurrencyState(limit=1)
    concurrency.admit(1, 0.0, 60.0)
    providers = [
        TieredProvider(
            name="full_concurrency",
            cost_per_token=0.0,
            input_cost_per_token=0.0,
            output_cost_per_token=0.0,
            ttft_dist=Uniform(90.0, 110.0),
            tps_dist=Uniform(100.0, 200.0),
            tier=ProviderTier.S_C,
            concurrency=concurrency,
        ),
        TieredProvider(
            name="quota",
            cost_per_token=0.0,
            input_cost_per_token=0.0,
            output_cost_per_token=0.0,
            ttft_dist=Uniform(900.0, 1100.0),
            tps_dist=Uniform(100.0, 200.0),
            tier=ProviderTier.S_Q,
            quota=QuotaState(size=10),
        ),
        TieredProvider(
            name="paid_api",
            cost_per_token=1e-6,
            input_cost_per_token=1e-6,
            output_cost_per_token=5e-6,
            ttft_dist=Uniform(50.0, 100.0),
            tps_dist=Uniform(100.0, 200.0),
            tier=ProviderTier.S_A,
        ),
    ]
    state = SimulationState.from_providers({provider.name: provider for provider in providers})
    request = Request(id=1, timestamp=0.0, request_tokens=100, response_tokens=50, total_tokens=150)
    policy = build_policy("greedy_cost", seed=1)

    decision = policy.route(request, state)

    assert decision.primary_provider == "quota"


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


def test_routewise_hedge_checkpoints_spaced_by_2_5_percent_of_slo() -> None:
    checkpoints = hedge_checkpoints_for_slo(2000.0)

    assert checkpoints[0] == pytest.approx(0.5)
    assert checkpoints[-1] == pytest.approx(1.8)
    assert len(checkpoints) == 27
    assert all(
        right - left == pytest.approx(0.05)
        for left, right in pairwise(checkpoints)
    )


def test_routewise_hedging_selects_backup_by_success_probability():
    primary = TieredProvider(
        name="primary",
        cost_per_token=1e-6,
        ttft_dist=Uniform(3000.0, 4000.0),
        tps_dist=Uniform(100.0, 200.0),
        tier=ProviderTier.S_A,
    )
    cheap_but_too_slow = TieredProvider(
        name="cheap_but_too_slow",
        cost_per_token=1e-6,
        ttft_dist=Uniform(1400.0, 3000.0),
        tps_dist=Uniform(100.0, 200.0),
        tier=ProviderTier.S_A,
    )
    expensive_but_safe = TieredProvider(
        name="expensive_but_safe",
        cost_per_token=10e-6,
        ttft_dist=Uniform(100.0, 200.0),
        tps_dist=Uniform(100.0, 200.0),
        tier=ProviderTier.S_A,
    )
    providers = [primary, cheap_but_too_slow, expensive_but_safe]
    state = SimulationState.from_providers({provider.name: provider for provider in providers})
    request = Request(id=1, timestamp=0.0, request_tokens=100, response_tokens=50, total_tokens=150)
    policy = RouteWisePolicy(
        hedging="probability_target",
        explorer=False,
        p=0.75,
        seed=7,
        slo_ms=2000.0,
        cost_envelope=(1e-6, 1e-3),
    )
    decision = RoutingDecision(
        primary_provider="primary",
        hedge_checkpoints=(0.5,),
    )
    state.now = 0.5

    dispatch = policy.tick(request, decision, 0.5, state)

    assert dispatch is not None
    assert dispatch.backup_provider == "expensive_but_safe"


def test_routewise_explorer_does_not_randomize_backup_selection():
    policy = RouteWisePolicy(
        hedging="probability_target",
        explorer=True,
        p=0.75,
        seed=7,
        slo_ms=2000.0,
        cost_envelope=(1e-6, 1e-3),
    )
    infeasible = BackupCandidate(
        provider=TieredProvider(
            name="cheap_but_too_slow",
            cost_per_token=1e-6,
            ttft_dist=Uniform(1400.0, 3000.0),
            tps_dist=Uniform(100.0, 200.0),
            tier=ProviderTier.S_A,
        ),
        success_probability=0.50,
        marginal_cost=1e-6,
        true_mean_ms=2200.0,
    )
    feasible = BackupCandidate(
        provider=TieredProvider(
            name="expensive_but_safe",
            cost_per_token=10e-6,
            ttft_dist=Uniform(100.0, 200.0),
            tps_dist=Uniform(100.0, 200.0),
            tier=ProviderTier.S_A,
        ),
        success_probability=0.995,
        marginal_cost=10e-6,
        true_mean_ms=150.0,
    )
    before = copy.deepcopy(policy.rng.bit_generator.state)

    selected = policy._select_backup_candidate([infeasible, feasible])

    assert selected is feasible
    assert policy.rng.bit_generator.state == before


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


@pytest.mark.parametrize(
    ("latencies", "costs", "p_value"),
    [
        ([100.0, 300.0, 1000.0], [1.0, 1.0, 1.0], 0.75),
        ([100.0, 100.0, 1000.0], [1.0, 1.0, 1.0], 0.75),
        ([100.0, 100.0, 100.0], [1.0, 1.0 + 1e-10, 1.0 + 5e-10], 0.0),
    ],
)
def test_routewise_same_cost_shortcut_matches_lp_tiebreak(
    latencies: list[float],
    costs: list[float],
    p_value: float,
) -> None:
    names = ["fast", "medium", "slow"]
    objective = cost_tiebroken_objective(latencies, costs)
    c_min = min(costs)
    c_max = max(costs)
    success, vector = solve_simplex_lp(
        objective=objective,
        upper_constraint=costs,
        upper_bound=c_min + p_value * (c_max - c_min),
    )

    assert success
    assert vector is not None
    assert _same_cost_shortcut_weights(names, objective=objective, costs=costs) == (
        normalize_weights(names, vector)
    )


def test_routewise_same_cost_path_skips_lp_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    providers = [
        TieredProvider(
            name="slow",
            cost_per_token=1e-6,
            ttft_dist=Uniform(900.0, 1100.0),
            tps_dist=Uniform(100.0, 200.0),
            tier=ProviderTier.S_A,
        ),
        TieredProvider(
            name="fast",
            cost_per_token=1e-6,
            ttft_dist=Uniform(90.0, 110.0),
            tps_dist=Uniform(100.0, 200.0),
            tier=ProviderTier.S_A,
        ),
    ]
    state = SimulationState.from_providers({provider.name: provider for provider in providers})
    request = Request(id=1, timestamp=0.0, request_tokens=100, response_tokens=50, total_tokens=150)
    policy = RouteWisePolicy(
        hedging=False,
        explorer=False,
        p=0.75,
        seed=7,
        cost_envelope=(1e-6, 1e-3),
    )

    def fail_solve_lp(*args, **kwargs):
        raise AssertionError("solve_simplex_lp should not run for same-cost providers")

    monkeypatch.setattr(routewise_module, "solve_simplex_lp", fail_solve_lp)

    decision = policy.route(request, state)

    assert decision.primary_provider == "fast"
    assert decision.metadata["weights"] == {"fast": 1.0}


def test_rolling_latency_profile_does_not_use_future_observations() -> None:
    profile = RollingLatencyProfile(window_sec=100.0)

    profile.add_sample(10.0, 1000.0)
    profile.add_sample(2.0, 100.0)

    assert profile.mean(1.0) is None
    assert profile.cdf(500.0, 1.0) is None
    assert profile.mean(3.0) == pytest.approx(100.0)
    assert profile.cdf(500.0, 3.0) == pytest.approx(1.0)
    assert profile.mean(11.0) == pytest.approx(550.0)
    assert profile.cdf(500.0, 11.0) == pytest.approx(0.5)


def test_rolling_latency_profile_cdf_state_is_lazy() -> None:
    profile = RollingLatencyProfile(window_sec=100.0)

    for index in range(100):
        profile.add_sample(float(index), float(index))

    assert profile._cdf_pending == []
    assert profile._cdf_active == []
    assert profile._cdf_active_count == 0
    assert profile.mean(99.0) == pytest.approx(49.5)
    assert profile._cdf_pending == []
    assert profile._cdf_active == []

    assert profile.cdf(49.0, 99.0) == pytest.approx(0.5)
    assert profile._cdf_active_count == 100
    profile.add_sample(101.0, 101.0)
    assert len(profile._cdf_pending) == 1


def test_rolling_latency_profile_expires_out_of_order_observations() -> None:
    profile = RollingLatencyProfile(window_sec=5.0)

    profile.add_sample(10.0, 1000.0)
    profile.add_sample(2.0, 100.0)
    profile.add_sample(12.0, 300.0)
    profile.add_sample(4.0, 200.0)

    assert profile.mean(6.0) == pytest.approx(150.0)
    assert profile.cdf(150.0, 6.0) == pytest.approx(0.5)
    assert profile.mean(11.0) == pytest.approx(1000.0)
    assert profile.cdf(500.0, 11.0) == pytest.approx(0.0)
    assert profile.mean(13.0) == pytest.approx(650.0)
    assert profile.cdf(500.0, 13.0) == pytest.approx(0.5)


def test_routewise_configured_latency_profile_ignores_observations() -> None:
    providers = [
        TieredProvider(
            name="slow",
            cost_per_token=1e-6,
            ttft_dist=Uniform(900.0, 1100.0),
            tps_dist=Uniform(100.0, 200.0),
            tier=ProviderTier.S_A,
        ),
        TieredProvider(
            name="fast",
            cost_per_token=1e-6,
            ttft_dist=Uniform(90.0, 110.0),
            tps_dist=Uniform(100.0, 200.0),
            tier=ProviderTier.S_A,
        ),
    ]
    state = SimulationState.from_providers({provider.name: provider for provider in providers})
    request = Request(id=1, timestamp=0.0, request_tokens=100, response_tokens=50, total_tokens=150)
    policy = RouteWisePolicy(
        hedging=False,
        explorer=False,
        p=0.75,
        seed=7,
        cost_envelope=(1e-6, 1e-3),
        latency_profile_mode="configured",
    )

    decision = policy.route(request, state)
    policy.observe(
        request,
        decision,
        RoutingOutcome(
            request_id=request.id,
            primary_provider=decision.primary_provider,
            final_provider=decision.primary_provider,
            ttft_ms=10_000.0,
            cost_usd=0.0,
            metadata={
                "primary_observed_at": 0.0,
                "primary_ttft_ms": 10_000.0,
            },
        ),
    )
    next_decision = policy.route(request, state)

    assert decision.primary_provider == "fast"
    assert next_decision.primary_provider == "fast"
    assert policy.profiles == {}


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


def test_routewise_concurrency_shadow_price_is_zero():
    provider = TieredProvider(
        name="concurrency",
        cost_per_token=0.0,
        ttft_dist=Uniform(150.0, 450.0),
        tps_dist=Uniform(100.0, 200.0),
        tier=ProviderTier.S_C,
        concurrency=ConcurrencyState(limit=4),
    )

    empty_price = concurrency_shadow_price(provider, 0.0, U=1e-2, L=1e-4)
    provider.concurrency.admit(1, 0.0, 60.0)
    provider.concurrency.admit(2, 0.0, 60.0)
    loaded_price = concurrency_shadow_price(provider, 1.0, U=1e-2, L=1e-4)

    assert empty_price == loaded_price == pytest.approx(0.0)


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
