"""Unit tests for the effective-cost ablation LP-only policy."""

from __future__ import annotations

import pytest

from experiments.ablations.effective_cost.policy import LPOnlyAblationPolicy
from experiments.simulation.common import make_api_provider, make_quota_provider
from rwsim.engine.state import SimulationState
from rwsim.policies.routewise import RouteWisePolicy, quota_shadow_price
from rwsim.schemas import Request


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
        concurrency_curve="legacy_linear_u",
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
        concurrency_curve="legacy_linear_u",
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
        concurrency_curve="legacy_linear_u",
        p=0.0,
        cost_envelope=cost_envelope,
    ).route(request, state)
    high = LPOnlyAblationPolicy(
        quota_curve="exp_lu",
        concurrency_curve="legacy_linear_u",
        p=1.0,
        cost_envelope=cost_envelope,
    ).route(request, state)

    assert low.metadata["L"] == pytest.approx(high.metadata["L"])
    assert low.metadata["U"] == pytest.approx(high.metadata["U"])
    assert low.metadata["budget"] < high.metadata["budget"]


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_rejects_invalid_p(p: float) -> None:
    with pytest.raises(ValueError, match="p must be in \\[0, 1\\]"):
        LPOnlyAblationPolicy(
            quota_curve="exp_lu",
            concurrency_curve="legacy_linear_u",
            p=p,
            cost_envelope=(1.0, 2.0),
        )
