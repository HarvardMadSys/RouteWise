"""Routing-policy regressions: unprofiled penalty + paper effective cost."""

from __future__ import annotations

import threading

import pytest

from experiments.offline_stage.value_estimators import ConstantOutputPredictor
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
    CapacityUnavailableError,
    RequestContext,
    build_policy,
    select_checkpoint_backup,
)
from experiments.real_evaluation.shadow_price import (
    concurrency_shadow_price,
    effective_cost,
    quota_shadow_price,
    workload_cost_envelope,
)
from experiments.real_evaluation.transports import TransportConfig
from llm_routewise.core.lp import BudgetLPCandidate, solve_budget_lp


def _api_spec(
    name: str,
    in_p: float,
    out_p: float,
    *,
    cached_in_p: float | None = None,
) -> ProviderSpec:
    return ProviderSpec(
        name=name,
        tier="api",
        transport_cfg=TransportConfig(
            name=name,
            transport="openrouter",
            model="x",
            input_price_per_m=in_p,
            cached_input_price_per_m=cached_in_p,
            output_price_per_m=out_p,
            stream_cancel_billing="stops",
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
    policy.set_cost_envelope((1.0e-4, 2.0e-4))
    now = 1_000.0
    for _ in range(20):
        policy.add_sample("Mid_profiled", now, 1100.0)

    decision = policy.route(now, RequestContext(50, 128))
    weights = decision.lp_weights or {}
    assert weights.get("Mid_profiled", 0.0) == 1.0
    assert weights.get("Cheap_unprofiled", 0.0) == 0.0
    assert UNPROFILED_LATENCY_PENALTY_MS >= 1e8


def test_budget_range_lp_tiebreak_prefers_lower_effective_cost() -> None:
    """When latency objectives tie, real-eval should match simulator cost tiebreak."""
    mid = _api_spec("OR_mid", 1.0, 0.0)
    cheap = _api_spec("OR_cheap", 0.5, 0.0)
    expensive = _api_spec("OR_expensive", 1.5, 0.0)
    policy = BudgetRangePolicy(
        [mid, cheap, expensive],
        slo_ms=5000.0,
        budget_percentile=100,
    )
    policy.set_cost_envelope((5.0e-5, 1.5e-4))
    now = 100.0
    for _ in range(10):
        policy.add_sample("OR_mid", now, 500.0)
        policy.add_sample("OR_cheap", now, 500.0)
        policy.add_sample("OR_expensive", now, 500.0)

    ctx = RequestContext(prompt_tokens=100, completion_tokens_budget=0)
    decision = policy.route(now, ctx)

    assert decision.primary == "OR_cheap"
    assert decision.lp_weights == {"OR_cheap": 1.0}
    assert policy.rate_limit_fallback_candidates(now, ctx, excluded=set())[0] == "OR_cheap"


def test_real_eval_budget_lp_keeps_small_core_epsilon_weight() -> None:
    result = solve_budget_lp(
        [
            BudgetLPCandidate("main", objective=100.0, effective_cost=1.0),
            BudgetLPCandidate("tiny", objective=0.0, effective_cost=2.0),
        ],
        budget=1.0 + 1e-7,
    )

    assert result.feasible
    assert result.weights == pytest.approx({"main": 1.0 - 1e-7, "tiny": 1e-7})


def test_budget_range_latency_objective_penalizes_error_attempts() -> None:
    fast_flaky = _api_spec("OR_fast_flaky", 1.0, 0.0)
    steady = _api_spec("OR_steady", 1.0, 0.0)
    policy = BudgetRangePolicy(
        [fast_flaky, steady],
        slo_ms=5000.0,
        budget_percentile=100,
    )
    policy.set_cost_envelope((1.0e-4, 2.0e-4))
    now = 100.0
    for _ in range(4):
        policy.add_sample("OR_fast_flaky", now, 100.0)
    for _ in range(6):
        policy.add_sample("OR_fast_flaky", now, -1.0, "HTTP 429")
    for _ in range(10):
        policy.add_sample("OR_steady", now, 1000.0)

    decision = policy.route(now, RequestContext(prompt_tokens=100, completion_tokens_budget=0))

    assert decision.primary == "OR_steady"


def test_greedy_latency_penalizes_error_attempts() -> None:
    fast_flaky = _api_spec("OR_fast_flaky", 1.0, 0.0)
    steady = _api_spec("OR_steady", 1.0, 0.0)
    policy = build_policy("greedy_latency", [fast_flaky, steady], slo_ms=5000.0)
    now = 100.0
    for _ in range(4):
        policy.add_sample("OR_fast_flaky", now, 100.0)
    for _ in range(6):
        policy.add_sample("OR_fast_flaky", now, -1.0, "HTTP 429")
    for _ in range(10):
        policy.add_sample("OR_steady", now, 1000.0)

    decision = policy.route(now, RequestContext(prompt_tokens=100, completion_tokens_budget=0))

    assert decision.primary == "OR_steady"


def test_budget_range_policy_names_match_simulator_ablation_layers() -> None:
    """Real eval should expose the same first two simulator layers:
    LP-only ``budget_range_alpha*`` and LP+hedge ``budget_range_alpha*_hedge``."""
    specs = [_api_spec("OR_x", 0.3, 1.2)]

    lp_only = build_policy("budget_range_alpha75", specs=specs, slo_ms=2000.0)
    hedged = build_policy("budget_range_alpha75_hedge", specs=specs, slo_ms=2000.0)

    assert isinstance(lp_only, BudgetRangePolicy)
    assert lp_only.name == "budget_range_alpha75"
    assert lp_only.use_hedge is False
    assert isinstance(hedged, BudgetRangeHedgePolicy)
    assert hedged.name == "budget_range_alpha75_hedge"
    assert hedged.use_hedge is True


def test_checkpoint_backup_does_not_fall_back_to_low_median_when_infeasible() -> None:
    primary = _api_spec("OR_Inceptron", 0.24, 0.9)
    low_median_heavy_tail = _api_spec("OR_AtlasCloud", 0.295, 1.2)
    slow_backup = _api_spec("OR_Chutes", 0.118, 0.99)
    states = {
        spec.name: ProviderState.from_spec(spec)
        for spec in [primary, low_median_heavy_tail, slow_backup]
    }
    now = 100.0
    for _ in range(20):
        states["OR_Inceptron"].profile.add_sample(now, 6000.0)
        states["OR_Chutes"].profile.add_sample(now, 5500.0)
    for _ in range(10):
        states["OR_AtlasCloud"].profile.add_sample(now, 1000.0)
        states["OR_AtlasCloud"].profile.add_sample(now, 6000.0)

    decision = select_checkpoint_backup(
        primary="OR_Inceptron",
        states=states,
        ctx=RequestContext(10, 8),
        slo_sec=5.0,
        now=now,
        elapsed_sec=1.0,
    )

    assert decision.backup is None


def test_profile_bootstrap_requirement_is_policy_owned() -> None:
    specs = [_api_spec("OR_x", 0.3, 1.2)]

    needs_profile = [
        "budget_range_alpha75",
        "budget_range_alpha75_hedge",
        "greedy_latency",
        "or_greedy_latency",
        "quota_first",
        "concurrency_first",
    ]
    profile_free = [
        "or_auto",
        "or_sort_cost",
        "greedy_cost",
        "or_greedy_cost",
        "random",
    ]

    for name in needs_profile:
        assert build_policy(name, specs=specs, slo_ms=2000.0).requires_latency_profile_bootstrap
    for name in profile_free:
        assert not build_policy(name, specs=specs, slo_ms=2000.0).requires_latency_profile_bootstrap


def test_quota_provider_effective_cost_is_paper_piecewise() -> None:
    """A quota provider with an extra admission constraint still uses only
    the quota shadow price for routing effective cost.

    The extra concurrency limit is enforced through availability, not by
    adding a second shadow-price term to ``c_eff``.
    """
    spec = ProviderSpec(
        name="Ollama_SQ",
        tier="quota",
        transport_cfg=TransportConfig(name="Ollama_SQ", transport="ollama_cloud", model="x"),
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
    assert c_sp == 0.0
    eff = effective_cost(state, request_cost_usd=0.0, now=now, U=1.0, L=0.001)
    assert eff == q_sp


def test_concurrency_provider_effective_cost_is_paper_piecewise() -> None:
    """A concurrency-tier provider uses zero real-eval concurrency shadow price."""
    spec = ProviderSpec(
        name="Featherless_SC",
        tier="concurrency",
        transport_cfg=TransportConfig(name="Featherless_SC", transport="featherless", model="x"),
        concurrency_limit=3,
    )
    state = ProviderState.from_spec(spec)
    now = 100.0
    state.concurrency.admit(1, now, 60.0)
    state.concurrency.admit(2, now, 60.0)

    c_sp = concurrency_shadow_price(state, now, U=1.0, L=0.001)
    eff = effective_cost(state, request_cost_usd=0.0, now=now, U=1.0, L=0.001)
    assert c_sp == 0.0
    assert eff == c_sp


def test_concurrency_capacity_holds_until_explicit_release() -> None:
    """Live concurrency accounting must track real completion, not estimates."""
    spec = ProviderSpec(
        name="Featherless_SC",
        tier="concurrency",
        transport_cfg=TransportConfig(name="Featherless_SC", transport="featherless", model="x"),
        concurrency_limit=1,
    )
    policy = build_policy("greedy_cost", specs=[spec], slo_ms=2000.0)
    now = 100.0

    request_id = policy.charge_capacity("Featherless_SC", now, expected_service_sec=0.1)

    assert request_id is not None
    assert not policy.states["Featherless_SC"].is_available(now + 60.0)
    with pytest.raises(CapacityUnavailableError):
        policy.charge_capacity("Featherless_SC", now + 60.0, expected_service_sec=0.1)
    policy.release_capacity("Featherless_SC", request_id, now + 60.0)
    assert policy.states["Featherless_SC"].is_available(now + 60.0)


def test_or_baselines_use_distinct_sentinels() -> None:
    """All four OR baselines (auto + 3 sort modes) round-trip to a unique
    sentinel string and a sensible ``provider.sort`` value."""
    specs = [_api_spec("OR_x", 0.3, 1.2)]
    auto = build_policy("or_auto", specs=specs, slo_ms=2000.0).route(0.0, RequestContext(10, 8))
    latency = build_policy("or_sort_latency", specs=specs, slo_ms=2000.0).route(
        0.0, RequestContext(10, 8)
    )
    price = build_policy("or_sort_cost", specs=specs, slo_ms=2000.0).route(
        0.0, RequestContext(10, 8)
    )
    throughput = build_policy("or_sort_throughput", specs=specs, slo_ms=2000.0).route(
        0.0, RequestContext(10, 8)
    )

    sentinels = {auto.primary, latency.primary, price.primary, throughput.primary}
    assert len(sentinels) == 4
    assert auto.primary == OR_AUTO_SENTINEL
    assert OR_SORT_SENTINEL_TO_MODE[latency.primary] == "latency"
    assert OR_SORT_SENTINEL_TO_MODE[price.primary] == "price"
    assert OR_SORT_SENTINEL_TO_MODE[throughput.primary] == "throughput"


def test_or_only_greedy_baselines_filter_to_openrouter() -> None:
    """OR-only greedy aliases must ignore subscription
    providers, keeping the apples-to-apples baseline against OpenRouter
    sort modes."""
    cheap_or = _api_spec("OR_cheap", 0.05, 0.2)
    expensive_or = _api_spec("OR_expensive", 0.5, 2.0)
    sub = ProviderSpec(
        name="Chutes_SQ",
        tier="quota",
        transport_cfg=TransportConfig(name="Chutes_SQ", transport="chutes", model="x"),
        quota_window_sec=3600,
        quota_requests=100,
    )
    specs = [cheap_or, expensive_or, sub]

    or_cheapest = build_policy("or_greedy_cost", specs=specs, slo_ms=2000.0).route(
        0.0, RequestContext(10, 8)
    )
    joint_cheapest = build_policy("greedy_cost", specs=specs, slo_ms=2000.0).route(
        0.0, RequestContext(10, 8)
    )

    # OR-only must pick the cheap OR provider; joint pool can pick the
    # zero-priced subscription provider.
    assert or_cheapest.primary == "OR_cheap"
    assert joint_cheapest.primary == "Chutes_SQ"


def test_greedy_cost_prefers_concurrency_before_quota_on_zero_cost_tie() -> None:
    """Paper greedy-cost tie-break should spend perishable S_C capacity first."""
    quota = ProviderSpec(
        name="Chutes_SQ",
        tier="quota",
        transport_cfg=TransportConfig(name="Chutes_SQ", transport="chutes", model="x"),
        quota_window_sec=3600,
        quota_requests=100,
    )
    concurrency = ProviderSpec(
        name="Featherless_SC",
        tier="concurrency",
        transport_cfg=TransportConfig(name="Featherless_SC", transport="featherless", model="x"),
        concurrency_limit=1,
    )
    api = _api_spec("OR_cheap", 0.05, 0.2)
    policy = build_policy("greedy_cost", specs=[quota, concurrency, api], slo_ms=2000.0)

    assert policy.route(0.0, RequestContext(10, 8)).primary == "Featherless_SC"
    assert (
        policy.rate_limit_fallback_candidates(
            0.0,
            RequestContext(10, 8),
            excluded={"Featherless_SC"},
        )[0]
        == "Chutes_SQ"
    )


def test_greedy_cost_tiebreaks_alphabetically_and_ignores_latency() -> None:
    """When two providers have identical price + tier, ``greedy_cost`` must
    pick by ``spec.name`` (alphabetical), NOT by rolling latency. The cost
    baseline should not benefit from latency information — that's the
    territory of ``greedy_latency`` and ``budget_range_*``.

    Setup: identical 0.15/1.15 prices, same api tier. We feed the
    alphabetically-later provider a clearly faster latency profile and the
    alphabetically-earlier provider a clearly slower one. ``greedy_cost``
    must still pick the alphabetically-earlier one (i.e. ignore latency).
    """
    fast_later = _api_spec("OR_zeta", 0.15, 1.15)
    slow_earlier = _api_spec("OR_alpha", 0.15, 1.15)
    policy = build_policy(
        "greedy_cost",
        specs=[fast_later, slow_earlier],
        slo_ms=2000.0,
    )
    now = 0.0
    # Alphabetically-later provider gets the better profile (10ms TTFT).
    policy.add_sample("OR_zeta", now, 10.0, None)
    # Alphabetically-earlier provider gets the worse profile (2000ms TTFT).
    policy.add_sample("OR_alpha", now, 2000.0, None)
    decision = policy.route(now, RequestContext(10, 8))
    assert decision.primary == "OR_alpha", (
        f"greedy_cost must tie-break by spec.name alphabetical, not by "
        f"rolling latency. Got primary={decision.primary}"
    )


def test_greedy_cost_stops_routing_to_quota_provider_after_exhaustion() -> None:
    """Once a quota provider's per-window budget is fully consumed by this
    policy's ``charge_capacity`` calls, ``route()`` must skip that provider
    on every subsequent request until the window rolls. Regression coverage
    for the monitor's ``quota left`` column: the displayed remaining count
    must actually correspond to "no more dispatches allowed", not just a
    diagnostic counter.
    """
    quota_size = 3
    quota = ProviderSpec(
        name="Chutes_SQ",
        tier="quota",
        transport_cfg=TransportConfig(name="Chutes_SQ", transport="chutes", model="x"),
        quota_window_sec=3600,
        quota_requests=quota_size,
    )
    api = _api_spec("OR_cheap", 0.05, 0.2)
    policy = build_policy("greedy_cost", specs=[quota, api], slo_ms=2000.0)
    ctx = RequestContext(10, 8)
    now = 0.0

    # The first ``quota_size`` requests must go to the zero-cost quota
    # provider; in the same loop we drive the policy's quota counter via
    # ``charge_capacity`` exactly the same way the live runner does.
    for _ in range(quota_size):
        assert policy.route(now, ctx).primary == "Chutes_SQ"
        policy.charge_capacity("Chutes_SQ", now, expected_service_sec=0.1)

    # Quota is now exhausted in this window: even though the policy still
    # *prefers* zero-cost providers, ``is_available`` must reject Chutes
    # and routing falls back to the API tier.
    assert not policy.states["Chutes_SQ"].is_available(now)
    for _ in range(5):
        assert policy.route(now, ctx).primary == "OR_cheap"

    # And it must not appear as a rate-limit fallback candidate either:
    # exhausted-quota providers are no more routable as fallbacks than as
    # primaries. (If this regresses, the runner would burn 429-retry budget
    # on a provider it knows is locked out.)
    fallbacks = policy.rate_limit_fallback_candidates(now, ctx, excluded=set())
    assert "Chutes_SQ" not in fallbacks


def test_random_policy_uses_only_available_providers() -> None:
    quota = ProviderSpec(
        name="Chutes_SQ",
        tier="quota",
        transport_cfg=TransportConfig(name="Chutes_SQ", transport="chutes", model="x"),
        quota_window_sec=3600,
        quota_requests=1,
    )
    concurrency = ProviderSpec(
        name="Featherless_SC",
        tier="concurrency",
        transport_cfg=TransportConfig(name="Featherless_SC", transport="featherless", model="x"),
        concurrency_limit=1,
    )
    api = _api_spec("OR_cheap", 0.05, 0.2)
    policy = build_policy("random", specs=[quota, concurrency, api], slo_ms=2000.0)
    policy.charge_capacity("Chutes_SQ", 0.0, 60.0)
    policy.charge_capacity("Featherless_SC", 0.0, 60.0)

    assert policy.route(0.0, RequestContext(10, 8)).primary == "OR_cheap"


def test_budget_range_policy_uses_trace_cached_input_tokens_for_api_cost() -> None:
    slow_cheap = _api_spec("OR_slow_cheap", 1.0, 0.1)
    fast_expensive = _api_spec("OR_fast_expensive", 10.0, 0.1, cached_in_p=0.0)
    now = 100.0

    no_cache = BudgetRangePolicy(
        [slow_cheap, fast_expensive],
        slo_ms=2000.0,
        budget_percentile=0,
        prefix_cache_routing=False,
        output_predictor=ConstantOutputPredictor(value=1.0),
    )
    cache_aware = BudgetRangePolicy(
        [slow_cheap, fast_expensive],
        slo_ms=2000.0,
        budget_percentile=0,
        prefix_cache_routing=True,
        output_predictor=ConstantOutputPredictor(value=1.0),
    )
    for policy in (no_cache, cache_aware):
        policy.set_cost_envelope((1.0e-4, 2.0e-3))
        for _ in range(10):
            policy.add_sample("OR_slow_cheap", now, 1000.0)
            policy.add_sample("OR_fast_expensive", now, 100.0)

    ctx = RequestContext(
        100, 1, prefix_id="conv-1", trace_cached_input_tokens=100
    )

    assert no_cache.route(now, ctx).primary == "OR_slow_cheap"
    assert cache_aware.route(now, ctx).primary == "OR_fast_expensive"
    cached_tokens, estimated_cost = cache_aware.routing_cache_diagnostics("OR_fast_expensive", ctx)
    assert cached_tokens == 100
    assert estimated_cost == pytest.approx(0.1 / 1_000_000.0)


def test_request_cost_uses_output_predictor_not_completion_budget() -> None:
    """Route-time API cost must come from the policy's output-token predictor,
    not from ``ctx.completion_tokens_budget``.

    The latter is the trace's realized output length — an oracle. Real-eval
    must mirror the simulator and ask the policy's online predictor (default:
    bucket mean) for the predicted output length so the live LP path uses
    arrival-time information only.
    """
    api = _api_spec("OR_x", in_p=0.0, out_p=1.0)
    predictor = ConstantOutputPredictor(value=42.0)
    policy = build_policy(
        "greedy_cost",
        specs=[api],
        slo_ms=2000.0,
        output_predictor=predictor,
    )

    # ``completion_tokens_budget=999`` is the oracle value we MUST NOT use.
    ctx = RequestContext(prompt_tokens=10, completion_tokens_budget=999)
    cost = policy.request_cost_for_spec(api, ctx)

    # ``out_p = 1.0`` USD/M tokens, so cost == predicted_output * 1.0 / 1e6.
    # If the implementation still used ctx.completion_tokens_budget, the
    # answer would be 999 / 1e6 (~9.99e-04), not 42 / 1e6 (~4.2e-05).
    assert cost == pytest.approx(42.0 / 1_000_000.0)


def test_observe_response_updates_predictor() -> None:
    """``observe_response`` should feed realized completion tokens back into
    the predictor so cold-start defaults are replaced by warmed-up means."""
    api = _api_spec("OR_x", in_p=0.0, out_p=1.0)
    from experiments.offline_stage.value_estimators import BucketMeanOutputPredictor

    predictor = BucketMeanOutputPredictor(
        min_samples_bucket=2,
        min_samples_global=2,
        default_output_tokens=500.0,
    )
    policy = build_policy(
        "greedy_cost",
        specs=[api],
        slo_ms=2000.0,
        output_predictor=predictor,
    )
    ctx = RequestContext(prompt_tokens=1024, completion_tokens_budget=1)

    # Cold start: predictor returns the default; cost reflects 500.
    cold_cost = policy.request_cost_for_spec(api, ctx)
    assert cold_cost == pytest.approx(500.0 / 1_000_000.0)

    # Two observed completions in the same input-length bucket warm the
    # bucket up, so the next prediction is their mean (= 80), not the default.
    policy.observe_response(ctx, completion_tokens=60)
    policy.observe_response(ctx, completion_tokens=100)
    warm_cost = policy.request_cost_for_spec(api, ctx)
    assert warm_cost == pytest.approx(80.0 / 1_000_000.0)


def test_predict_and_observe_response_are_mutually_excluded() -> None:
    """Concurrent ``route``/``observe_response`` calls must serialize their
    accesses to the predictor so a routing read cannot land in the middle
    of an ``update``.

    The custom predictor asserts that ``predict`` and ``update`` never
    overlap, which would catch the lock asymmetry where ``observe_response``
    was guarded but ``_predicted_output_tokens`` was not.
    """
    from experiments.offline_stage.value_estimators import OutputTokenPredictor
    from experiments.offline_stage.value_estimators.base import QuantilePrediction

    class _ConcurrencyAssertingPredictor(OutputTokenPredictor):
        """Records overlap between predict and update; never crashes."""

        def __init__(self) -> None:
            self._in_predict = 0
            self._in_update = 0
            self._cross_violation = False
            self._race_lock = threading.Lock()

        def predict(self, request):
            with self._race_lock:
                if self._in_update > 0:
                    self._cross_violation = True
                self._in_predict += 1
            try:
                # Burn a few microseconds inside the critical section so
                # an unprotected caller would actually overlap with update.
                for _ in range(50):
                    pass
            finally:
                with self._race_lock:
                    self._in_predict -= 1
            return QuantilePrediction(q10=10.0, q50=10.0, q90=10.0)

        def update(self, request):
            with self._race_lock:
                if self._in_predict > 0:
                    self._cross_violation = True
                self._in_update += 1
            try:
                for _ in range(50):
                    pass
            finally:
                with self._race_lock:
                    self._in_update -= 1

        def reset(self) -> None:
            pass

        @property
        def is_warmed_up(self) -> bool:
            return True

    predictor = _ConcurrencyAssertingPredictor()
    api = _api_spec("OR_x", in_p=0.0, out_p=1.0)
    policy = build_policy(
        "greedy_cost",
        specs=[api],
        slo_ms=2000.0,
        output_predictor=predictor,
    )
    ctx = RequestContext(prompt_tokens=64, completion_tokens_budget=1)

    stop = threading.Event()

    def hammer_predict() -> None:
        while not stop.is_set():
            policy.request_cost_for_spec(api, ctx)

    def hammer_update() -> None:
        while not stop.is_set():
            policy.observe_response(ctx, completion_tokens=42)

    predict_threads = [threading.Thread(target=hammer_predict) for _ in range(4)]
    update_threads = [threading.Thread(target=hammer_update) for _ in range(4)]
    for t in predict_threads + update_threads:
        t.start()
    # Let the contention build up briefly; under an unlocked predict path
    # this is enough to trip the cross-violation flag.
    threading.Event().wait(0.05)
    stop.set()
    for t in predict_threads + update_threads:
        t.join(timeout=2.0)

    assert not predictor._cross_violation, (
        "predict and update overlapped — policy lock did not serialize predictor access"
    )


def test_observe_response_ignores_missing_completion_tokens() -> None:
    """Zero or ``None`` completion-token counts (e.g. failed responses) must
    not corrupt the predictor with a zero sample."""
    api = _api_spec("OR_x", in_p=0.0, out_p=1.0)
    predictor = ConstantOutputPredictor(value=42.0)
    policy = build_policy(
        "greedy_cost",
        specs=[api],
        slo_ms=2000.0,
        output_predictor=predictor,
    )
    ctx = RequestContext(prompt_tokens=10, completion_tokens_budget=1)

    policy.observe_response(ctx, completion_tokens=None)
    policy.observe_response(ctx, completion_tokens=0)

    # Constant predictor is unaffected, but the call must not raise.
    cost = policy.request_cost_for_spec(api, ctx)
    assert cost == pytest.approx(42.0 / 1_000_000.0)


def test_latency_profile_memory_bounded_under_legacy_only_queries() -> None:
    """Baseline-policy pattern: legacy reads never advance the core clocks.

    Before retention pruning, a 24h feed left every sample of the run in the
    shared profile's ``_mean_pending`` heap and ``_samples_by_ts`` index for
    profiles only ever read through the legacy methods (median_ms etc.).
    """
    from experiments.real_evaluation.inventory import LatencyProfile

    profile = LatencyProfile(window_sec=100.0)
    for i in range(5000):
        profile.add_sample(float(i), 50.0)
        if i % 10 == 0:
            profile.median_ms(float(i))
    bound = 2 * 100 + 1024 + 50  # two windows at 1 sample/sec + prune-batch slack
    assert len(profile._mean_pending) <= bound
    assert len(profile._samples_by_ts) <= bound
    assert len(profile.samples) <= bound
    # Legacy semantics are untouched by the pruning: window [4899, 4999]
    # holds 101 one-per-second samples.
    assert profile.median_ms(4999.0) == pytest.approx(50.0)
    assert profile.sample_count(4999.0) == 101


def test_budget_range_rate_limit_fallback_uses_routewise_objective() -> None:
    slow_alphabetical_first = _api_spec("OR_AtlasCloud", 0.1, 0.1)
    fast_alphabetical_later = _api_spec("OR_Friendli", 10.0, 10.0)
    policy = BudgetRangePolicy(
        [slow_alphabetical_first, fast_alphabetical_later],
        slo_ms=5000.0,
        budget_percentile=100,
    )
    policy.set_cost_envelope((1.0e-4, 2.0e-3))
    now = 100.0
    for _ in range(10):
        policy.add_sample("OR_AtlasCloud", now, 3000.0)
        policy.add_sample("OR_Friendli", now, 300.0)

    candidates = policy.rate_limit_fallback_candidates(
        now,
        RequestContext(prompt_tokens=10, completion_tokens_budget=8),
        excluded={"Featherless_SC"},
    )

    assert candidates[0] == "OR_Friendli"


def test_workload_cost_envelope_uses_p10_p90_of_cheapest_api_costs() -> None:
    """Workload envelope is P10/P90 of cheapest-API per-request cost, not U/1000."""
    cheap = _api_spec("OR_Chutes", in_p=0.118, out_p=0.99)
    mid = _api_spec("OR_DeepInfra", in_p=0.27, out_p=0.95)
    expensive = _api_spec("OR_Friendli", in_p=0.30, out_p=1.20)
    specs = [cheap, mid, expensive]

    # 500 requests with prompt=650, output=15: cheapest is always OR_Chutes
    # at 650 * 0.118/M + 15 * 0.99/M = $7.67e-5 + $1.485e-5 = $9.155e-5.
    prompts = [650] * 500
    completions = [15] * 500

    L, U = workload_cost_envelope(specs, prompts, completions)

    # Constant workload -> P10 == P90 -> falls back to L = U * floor_ratio.
    assert pytest.approx(9.155e-5, rel=1e-3) == U
    assert pytest.approx(U * 1e-3, rel=1e-6) == L

    # Varied workload -> envelope should span real per-request cost range.
    prompts2 = [100, 200, 400, 800, 1600]
    completions2 = [5, 10, 20, 40, 80]
    L2, U2 = workload_cost_envelope(specs, prompts2, completions2)
    assert 0 < L2 < U2
    assert U2 / L2 < 100  # Should NOT use the old single-request U/1000 shortcut.


def test_budget_range_requires_workload_cost_envelope_before_route() -> None:
    """BudgetRange routing must fail fast without workload-level ``(L, U)``."""
    spec = _api_spec("OR_x", in_p=0.30, out_p=1.20)
    policy = BudgetRangePolicy([spec], 2000.0)

    assert policy.cost_envelope is None
    with pytest.raises(RuntimeError, match="requires a workload cost envelope"):
        policy.route(0.0, RequestContext(650, 15))
    with pytest.raises(RuntimeError, match="requires a workload cost envelope"):
        policy.rate_limit_fallback_candidates(
            0.0,
            RequestContext(650, 15),
            excluded={"Some_other_provider"},
        )

    policy.set_cost_envelope((1.0e-4, 2.0e-4))
    assert policy.cost_envelope == (1.0e-4, 2.0e-4)

    # Bad input rejected.
    with pytest.raises(ValueError):
        policy.set_cost_envelope((1.0, 0.5))


def test_routing_cache_diagnostics_returns_zero_when_trace_field_missing() -> None:
    spec = _api_spec("OR_cached", 1.0, 0.1, cached_in_p=0.0)
    policy = BudgetRangePolicy([spec], 2000.0, prefix_cache_routing=True)
    ctx_missing = RequestContext(50, 5, prefix_id="conv-1")
    ctx_with_field = RequestContext(
        50, 5, prefix_id="conv-1", trace_cached_input_tokens=40
    )

    assert policy.routing_cache_diagnostics("OR_cached", ctx_missing)[0] == 0
    assert policy.routing_cache_diagnostics("OR_cached", ctx_with_field)[0] == 40
