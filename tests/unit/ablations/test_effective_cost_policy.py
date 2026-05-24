"""Unit tests for the effective-cost ablation LP-only policy."""

from __future__ import annotations

import pytest

from experiments.ablations.effective_cost import policy as effective_cost_policy
from experiments.ablations.effective_cost.policy import LPOnlyAblationPolicy, RollingLatencyProfile
from experiments.simulation.common import (
    make_api_provider,
    make_concurrency_provider,
    make_quota_provider,
)
from rwsim.core.lp import BudgetLPCandidate, cost_tiebroken_objective, solve_budget_lp
from rwsim.engine.state import SimulationState
from rwsim.policies.routewise import RouteWisePolicy, quota_shadow_price
from rwsim.schemas import Request, RoutingDecision, RoutingOutcome


def _request() -> Request:
    return Request(
        id=1,
        timestamp=0.0,
        request_tokens=100,
        response_tokens=100,
        total_tokens=200,
    )


def test_exp_lu_quota_effective_cost_matches_production_formula() -> None:
    provider = make_quota_provider(
        "quota",
        quota_size=10,
        latency_family="heavy_tail",
    )
    for _ in range(5):
        provider.quota.charge(0.0)

    policy = LPOnlyAblationPolicy(
        quota_curve="exp_lu",
        concurrency_curve="util_linear_u",
        p=0.5,
        cost_envelope=(1.0, 100.0),
    )

    assert policy.effective_cost_for_provider(
        provider,
        _request(),
        0.0,
        L=1.0,
        U=100.0,
    ) == pytest.approx(quota_shadow_price(provider, 0.0, L=1.0, U=100.0))


def test_util_linear_u_concurrency_effective_cost_uses_utilization_times_u() -> None:
    provider = make_concurrency_provider("concurrency", concurrency_limit=10)
    for request_id in range(4):
        provider.concurrency.admit(request_id, 0.0, 10.0)

    policy = LPOnlyAblationPolicy(
        quota_curve="exp_lu",
        concurrency_curve="util_linear_u",
        p=0.5,
        cost_envelope=(1.0, 100.0),
    )

    assert policy.effective_cost_for_provider(
        provider,
        _request(),
        0.0,
        L=1.0,
        U=100.0,
    ) == pytest.approx(40.0)


def test_lp_only_current_curve_matches_routewise_cost_router_metadata() -> None:
    providers = [
        make_quota_provider("quota", quota_size=10, latency_family="heavy_tail"),
        make_api_provider(
            "api_cheap",
            cost_per_million_tokens=1.0,
            latency_family="heavy_tail",
        ),
        make_api_provider(
            "api_expensive",
            cost_per_million_tokens=4.0,
            latency_family="heavy_tail",
        ),
    ]
    request = _request()
    state = SimulationState.from_providers({provider.name: provider for provider in providers})
    cost_envelope = (0.0001, 0.001)

    ablation = LPOnlyAblationPolicy(
        quota_curve="exp_lu",
        concurrency_curve="util_linear_u",
        p=0.5,
        cost_envelope=cost_envelope,
        seed=7,
    )
    routewise = RouteWisePolicy(
        hedging=False,
        explorer=False,
        p=0.5,
        cost_envelope=cost_envelope,
        seed=7,
    )

    ablation_decision = ablation.route(request, state)
    routewise_decision = routewise.route(request, state)

    assert ablation_decision.metadata["c_eff"] == pytest.approx(
        routewise_decision.metadata["c_eff"]
    )
    assert ablation_decision.metadata["budget"] == pytest.approx(
        routewise_decision.metadata["budget"]
    )
    assert ablation_decision.metadata["weights"] == pytest.approx(
        routewise_decision.metadata["weights"]
    )
    assert ablation_decision.primary_provider == routewise_decision.primary_provider


def test_p_changes_budget_without_changing_cost_envelope() -> None:
    providers = [
        make_quota_provider("quota", quota_size=10, latency_family="heavy_tail"),
        make_api_provider(
            "api_cheap",
            cost_per_million_tokens=1.0,
            latency_family="heavy_tail",
        ),
        make_api_provider(
            "api_expensive",
            cost_per_million_tokens=4.0,
            latency_family="heavy_tail",
        ),
    ]
    state = SimulationState.from_providers({provider.name: provider for provider in providers})
    request = _request()
    cost_envelope = (0.0001, 0.001)

    low = LPOnlyAblationPolicy(
        quota_curve="exp_lu",
        concurrency_curve="util_linear_u",
        p=0.0,
        cost_envelope=cost_envelope,
    ).route(request, state)
    high = LPOnlyAblationPolicy(
        quota_curve="exp_lu",
        concurrency_curve="util_linear_u",
        p=1.0,
        cost_envelope=cost_envelope,
    ).route(request, state)

    assert low.metadata["L"] == pytest.approx(high.metadata["L"])
    assert low.metadata["U"] == pytest.approx(high.metadata["U"])
    assert low.metadata["budget"] < high.metadata["budget"]


def test_p_zero_fast_path_matches_lp_enumerator() -> None:
    providers = [
        make_api_provider(
            "api_cheap",
            cost_per_million_tokens=1.0,
            latency_family="heavy_tail",
        ),
        make_api_provider(
            "api_expensive",
            cost_per_million_tokens=4.0,
            latency_family="heavy_tail",
        ),
    ]
    state = SimulationState.from_providers({provider.name: provider for provider in providers})
    request = _request()
    cost_envelope = (0.0001, 0.001)
    policy = LPOnlyAblationPolicy(
        quota_curve="exp_lu",
        concurrency_curve="util_linear_u",
        p=0.0,
        cost_envelope=cost_envelope,
        seed=7,
    )

    names = [provider.name for provider in providers]
    c_eff = {
        provider.name: policy.effective_cost_for_provider(
            provider,
            request,
            state.now,
            L=cost_envelope[0],
            U=cost_envelope[1],
        )
        for provider in providers
    }
    tbar = {
        provider.name: policy._latency_objective_ms(provider, state.now) for provider in providers
    }
    budget = min(c_eff.values())
    objective = cost_tiebroken_objective(
        [tbar[name] for name in names],
        [c_eff[name] for name in names],
    )
    result = solve_budget_lp(
        [
            BudgetLPCandidate(name, objective=objective[index], effective_cost=c_eff[name])
            for index, name in enumerate(names)
        ],
        budget=budget,
    )

    decision = policy.route(request, state)

    assert result.feasible
    assert decision.metadata["weights"] == pytest.approx(result.weights)


def test_p_zero_fast_path_skips_generic_lp_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    providers = [
        make_api_provider(
            "api_cheap",
            cost_per_million_tokens=1.0,
            latency_family="heavy_tail",
        ),
        make_api_provider(
            "api_expensive",
            cost_per_million_tokens=4.0,
            latency_family="heavy_tail",
        ),
    ]
    state = SimulationState.from_providers({provider.name: provider for provider in providers})
    request = _request()
    policy = LPOnlyAblationPolicy(
        quota_curve="exp_lu",
        concurrency_curve="util_linear_u",
        p=0.0,
        cost_envelope=(0.0001, 0.001),
        seed=7,
    )

    def fail_solve_lp(*args, **kwargs):
        raise AssertionError("solve_budget_lp should not run for p=0")

    monkeypatch.setattr(effective_cost_policy, "solve_budget_lp", fail_solve_lp)

    decision = policy.route(request, state)

    assert decision.primary_provider == "api_cheap"
    assert decision.metadata["weights"] == {"api_cheap": 1.0}


def test_p_changes_weights_after_latency_profile_observations() -> None:
    providers = [
        make_api_provider(
            "api_cheap",
            cost_per_million_tokens=1.0,
            latency_family="heavy_tail",
        ),
        make_api_provider(
            "api_expensive",
            cost_per_million_tokens=4.0,
            latency_family="heavy_tail",
        ),
    ]
    state = SimulationState.from_providers({provider.name: provider for provider in providers})
    request = _request()
    cost_envelope = (0.0001, 0.001)

    def policy_with_profile(p: float) -> LPOnlyAblationPolicy:
        policy = LPOnlyAblationPolicy(
            quota_curve="exp_lu",
            concurrency_curve="util_linear_u",
            p=p,
            cost_envelope=cost_envelope,
            seed=7,
        )
        policy.observe(
            request,
            RoutingDecision(primary_provider="api_cheap"),
            RoutingOutcome(
                request_id=request.id,
                primary_provider="api_cheap",
                final_provider="api_cheap",
                ttft_ms=500.0,
                cost_usd=0.0,
                metadata={"primary_observed_at": 0.0, "primary_ttft_ms": 500.0},
            ),
        )
        policy.observe(
            request,
            RoutingDecision(primary_provider="api_expensive"),
            RoutingOutcome(
                request_id=request.id,
                primary_provider="api_expensive",
                final_provider="api_expensive",
                ttft_ms=100.0,
                cost_usd=0.0,
                metadata={"primary_observed_at": 0.0, "primary_ttft_ms": 100.0},
            ),
        )
        return policy

    low = policy_with_profile(0.0).route(request, state)
    high = policy_with_profile(1.0).route(request, state)

    assert low.metadata["weights"] == {"api_cheap": 1.0}
    assert high.metadata["weights"] == {"api_expensive": 1.0}


def test_ablation_uses_optimized_rolling_latency_profile_semantics() -> None:
    profile = RollingLatencyProfile(window_sec=5.0)

    profile.add_sample(10.0, 1000.0)
    profile.add_sample(2.0, 100.0)
    profile.add_sample(12.0, 300.0)
    profile.add_sample(4.0, 200.0)

    assert profile.mean(6.0) == pytest.approx(150.0)
    assert profile.mean(11.0) == pytest.approx(1000.0)
    assert profile.mean(13.0) == pytest.approx(650.0)


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_rejects_invalid_p(p: float) -> None:
    with pytest.raises(ValueError, match="p must be in \\[0, 1\\]"):
        LPOnlyAblationPolicy(
            quota_curve="exp_lu",
            concurrency_curve="util_linear_u",
            p=p,
            cost_envelope=(1.0, 2.0),
        )
