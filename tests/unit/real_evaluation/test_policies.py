"""Routing-policy regressions: unprofiled penalty + paper effective cost."""

from __future__ import annotations

import pytest

from experiments.real_evaluation.inventory import (
    ProviderSpec,
    ProviderState,
)
from experiments.real_evaluation.policies import (
    OR_AUTO_SENTINEL,
    OR_SORT_SENTINEL_TO_MODE,
    UNPROFILED_LATENCY_PENALTY_MS,
    BudgetRangeHedgePolicy,
    BudgetRangePolicy,
    RequestContext,
    build_policy,
)
from experiments.real_evaluation.shadow_price import (
    concurrency_shadow_price,
    effective_cost,
    quota_shadow_price,
)
from experiments.real_evaluation.transports import TransportConfig


def _api_spec(name: str, in_p: float, out_p: float) -> ProviderSpec:
    return ProviderSpec(
        name=name,
        tier="api",
        transport_cfg=TransportConfig(
            name=name,
            transport="openrouter",
            model="x",
            input_price_per_m=in_p,
            output_price_per_m=out_p,
        ),
    )


def test_unprofiled_provider_does_not_appear_fastest() -> None:
    """Without a rolling profile, a provider's body-latency proxy must
    fall back to ``UNPROFILED_LATENCY_PENALTY_MS`` (large finite),
    not ``U * 1000`` (which is in USD * 1000 ≈ 0.1ms and made
    unprofiled providers look fastest)."""
    specs = [
        _api_spec("Cheap_unprofiled", 0.1, 0.5),
        _api_spec("Mid_profiled", 0.3, 1.5),
    ]
    policy = BudgetRangeHedgePolicy(specs, slo_ms=2000.0, budget_percentile=100)
    now = 1_000.0
    for _ in range(20):
        policy.add_sample("Mid_profiled", now, 1100.0)

    decision = policy.route(now, RequestContext(50, 128))
    weights = decision.lp_weights or {}
    assert weights.get("Mid_profiled", 0.0) == 1.0
    assert weights.get("Cheap_unprofiled", 0.0) == 0.0
    assert UNPROFILED_LATENCY_PENALTY_MS >= 1e8


def test_budget_range_policy_names_match_simulator_ablation_layers() -> None:
    """Real eval should expose the same first two simulator layers:
    LP-only ``budget_range_p*`` and LP+hedge ``budget_range_p*_hedge``."""
    specs = [_api_spec("OR_x", 0.3, 1.2)]

    lp_only = build_policy("budget_range_p75", specs=specs, slo_ms=2000.0)
    hedged = build_policy("budget_range_p75_hedge", specs=specs, slo_ms=2000.0)

    assert isinstance(lp_only, BudgetRangePolicy)
    assert lp_only.name == "budget_range_p75"
    assert lp_only.use_hedge is False
    assert isinstance(hedged, BudgetRangeHedgePolicy)
    assert hedged.name == "budget_range_p75_hedge"
    assert hedged.use_hedge is True


def test_profile_bootstrap_requirement_is_policy_owned() -> None:
    specs = [_api_spec("OR_x", 0.3, 1.2)]

    needs_profile = [
        "budget_range_p75",
        "budget_range_p75_hedge",
        "fastest_fixed",
        "or_fastest_fixed",
        "quota_first",
        "concurrency_first",
    ]
    profile_free = [
        "openrouter_auto",
        "sort_price",
        "cheapest_fixed",
        "or_cheapest_fixed",
    ]

    for name in needs_profile:
        assert build_policy(
            name, specs=specs, slo_ms=2000.0
        ).requires_latency_profile_bootstrap
    for name in profile_free:
        assert not build_policy(
            name, specs=specs, slo_ms=2000.0
        ).requires_latency_profile_bootstrap


def test_quota_provider_effective_cost_is_paper_piecewise() -> None:
    """A quota provider with an extra admission constraint still uses only
    the quota shadow price for routing effective cost.

    The extra concurrency limit is enforced through availability, not by
    adding a second shadow-price term to ``c_eff``.
    """
    spec = ProviderSpec(
        name="Ollama_SQ",
        tier="quota",
        transport_cfg=TransportConfig(
            name="Ollama_SQ", transport="ollama_cloud", model="x"
        ),
        quota_window_sec=3600,
        quota_requests=2000,
        concurrency_limit=3,
    )
    state = ProviderState.from_spec(spec)
    now = 100.0
    state.quota.charge(now)
    state.quota.charge(now)
    state.concurrency.admit(1, now, 60.0)
    state.concurrency.admit(2, now, 60.0)

    q_sp = quota_shadow_price(state, now, U=1.0, L=0.001)
    c_sp = concurrency_shadow_price(state, now, U=1.0, L=0.001)
    assert q_sp > 0.0
    assert c_sp > 0.0
    eff = effective_cost(state, request_cost_usd=0.0, now=now, U=1.0, L=0.001)
    assert eff == q_sp


def test_concurrency_provider_effective_cost_is_paper_piecewise() -> None:
    """A concurrency-tier provider uses only the concurrency shadow price."""
    spec = ProviderSpec(
        name="Featherless_SC",
        tier="concurrency",
        transport_cfg=TransportConfig(
            name="Featherless_SC", transport="featherless", model="x"
        ),
        concurrency_limit=3,
    )
    state = ProviderState.from_spec(spec)
    now = 100.0
    state.concurrency.admit(1, now, 60.0)
    state.concurrency.admit(2, now, 60.0)

    c_sp = concurrency_shadow_price(state, now, U=1.0, L=0.001)
    eff = effective_cost(state, request_cost_usd=0.0, now=now, U=1.0, L=0.001)
    assert c_sp == pytest.approx(0.001)
    assert eff == c_sp


def test_or_baselines_use_distinct_sentinels() -> None:
    """All four OR baselines (auto + 3 sort modes) round-trip to a unique
    sentinel string and a sensible ``provider.sort`` value."""
    specs = [_api_spec("OR_x", 0.3, 1.2)]
    auto = build_policy("openrouter_auto", specs=specs, slo_ms=2000.0).route(
        0.0, RequestContext(10, 8)
    )
    latency = build_policy("sort_latency", specs=specs, slo_ms=2000.0).route(
        0.0, RequestContext(10, 8)
    )
    price = build_policy("sort_price", specs=specs, slo_ms=2000.0).route(
        0.0, RequestContext(10, 8)
    )
    throughput = build_policy(
        "sort_throughput", specs=specs, slo_ms=2000.0
    ).route(0.0, RequestContext(10, 8))

    sentinels = {auto.primary, latency.primary, price.primary, throughput.primary}
    assert len(sentinels) == 4
    assert auto.primary == OR_AUTO_SENTINEL
    assert OR_SORT_SENTINEL_TO_MODE[latency.primary] == "latency"
    assert OR_SORT_SENTINEL_TO_MODE[price.primary] == "price"
    assert OR_SORT_SENTINEL_TO_MODE[throughput.primary] == "throughput"


def test_or_only_fixed_baselines_filter_to_openrouter() -> None:
    """``or_cheapest_fixed`` and ``or_fastest_fixed`` must ignore subscription
    providers, keeping the apples-to-apples baseline against OpenRouter
    sort modes."""
    cheap_or = _api_spec("OR_cheap", 0.05, 0.2)
    expensive_or = _api_spec("OR_expensive", 0.5, 2.0)
    sub = ProviderSpec(
        name="Chutes_SQ",
        tier="quota",
        transport_cfg=TransportConfig(
            name="Chutes_SQ", transport="chutes", model="x"
        ),
        quota_window_sec=3600,
        quota_requests=100,
    )
    specs = [cheap_or, expensive_or, sub]

    or_cheapest = build_policy(
        "or_cheapest_fixed", specs=specs, slo_ms=2000.0
    ).route(0.0, RequestContext(10, 8))
    joint_cheapest = build_policy(
        "cheapest_fixed", specs=specs, slo_ms=2000.0
    ).route(0.0, RequestContext(10, 8))

    # OR-only must pick the cheap OR provider; joint pool can pick the
    # zero-priced subscription provider.
    assert or_cheapest.primary == "OR_cheap"
    assert joint_cheapest.primary == "Chutes_SQ"
