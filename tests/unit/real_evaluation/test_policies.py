"""Routing-policy regressions: unprofiled penalty + dual shadow price."""

from __future__ import annotations

from experiments.real_evaluation.inventory import (
    ProviderSpec,
    ProviderState,
)
from experiments.real_evaluation.policies import (
    UNPROFILED_LATENCY_PENALTY_MS,
    BudgetRangeHedgePolicy,
    OR_AUTO_SENTINEL,
    OR_SORT_SENTINEL_TO_MODE,
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


def test_dual_constraint_provider_gets_both_shadow_prices() -> None:
    """A provider declared ``tier='quota'`` with an extra ``concurrency_limit``
    must accumulate both shadow-price terms in ``effective_cost``.

    Earlier code gated each shadow price on the ``tier`` string, so
    Ollama_SQ (tier=quota, concurrency_limit=3) never received a
    concurrency penalty even when its slots were saturated.
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
    c_sp = concurrency_shadow_price(state, now, U=1.0)
    assert q_sp > 0.0
    assert c_sp > 0.0
    eff = effective_cost(state, request_cost_usd=0.0, now=now, U=1.0, L=0.001)
    assert eff == q_sp + c_sp


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
