"""Behavioral tests for the stateful API-provider facade."""

from __future__ import annotations

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest

from routewise._capacity_controller import _CapacitySnapshot, _NoopReservation
from routewise.errors import NoProviderError, OutcomeError, ValidationError
from routewise.facade import Provider, Router, Tuning


class FakeClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now
        self.reads = 0

    def __call__(self) -> float:
        self.reads += 1
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class RejectingCapacityController:
    def __init__(self, rejected: set[str]) -> None:
        self.rejected = rejected
        self.reserve_attempts: list[str] = []

    def snapshot(self, *, resource_key: str, now: float) -> _CapacitySnapshot:
        return _CapacitySnapshot(resource_key=resource_key, observed_at=now)

    def try_reserve(
        self,
        *,
        resource_key: str,
        attempt_id: str,
        snapshot: _CapacitySnapshot,
    ) -> _NoopReservation | None:
        assert snapshot.resource_key == resource_key
        self.reserve_attempts.append(resource_key)
        if resource_key in self.rejected:
            return None
        return _NoopReservation(resource_key=resource_key, attempt_id=attempt_id)


def _providers() -> list[Provider]:
    return [
        Provider("cheap", price_in=1.0, price_out=2.0, price_cached=0.5),
        Provider("fast", price_in=3.0, price_out=4.0),
    ]


def _warm(router: Router, provider: str, value: float = 100.0, count: int = 1) -> None:
    for _ in range(count):
        router.observe(provider, ttft_ms=value)


def _hedgeable_router(*, clock: FakeClock | None = None) -> Router:
    router = Router(
        _providers(),
        cold_start="require_observations",
        slo_ms=3000.0,
        seed=1,
        clock=clock,
    )
    _warm(router, "cheap", 100.0, count=5)
    _warm(router, "fast", 200.0, count=5)
    return router


def test_provider_and_constructor_validation() -> None:
    with pytest.raises(ValidationError):
        Provider("", 1.0, 2.0)
    with pytest.raises(ValidationError):
        Provider("p", float("nan"), 2.0)
    with pytest.raises(ValidationError):
        Router([])
    with pytest.raises(ValidationError):
        Router([Provider("p", 1.0, 2.0), Provider("p", 1.0, 2.0)])
    with pytest.raises(ValidationError):
        Router([Provider("p", 1.0, 2.0)], alpha=1.1)
    with pytest.raises(ValidationError):
        Router(None)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        Router([Provider("p", 1.0, 2.0)], clock=lambda: True)
    with pytest.raises(ValidationError):
        Router([Provider("p", 1.0, 2.0)], clock=lambda: "1.0")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Tuning(0.9)  # type: ignore[misc]


def test_strict_cold_start_requires_observations_and_routes_after_warmup() -> None:
    router = Router(_providers(), cold_start="require_observations", seed=3)

    with pytest.raises(NoProviderError, match="observe"):
        router.route(input_tokens=10)

    _warm(router, "cheap", 300.0)
    _warm(router, "fast", 100.0)
    decision = router.route(input_tokens=100, alpha=1.0)

    assert decision.provider == "fast"
    assert decision.weights == {"fast": 1.0}
    assert decision.expected_latency_ms == pytest.approx(100.0)
    assert decision.trace["reason"] == "latency_optimized_lp"
    with pytest.raises(TypeError):
        decision.weights["cheap"] = 1.0  # type: ignore[index]


def test_exploration_leases_each_selected_unprofiled_provider() -> None:
    clock = FakeClock()
    router = Router(_providers(), seed=1, clock=clock)

    first = router.route(input_tokens=100)
    second = router.route(input_tokens=100)

    assert first.provider == "cheap"
    assert second.provider == "fast"
    assert first.expected_latency_ms is None
    assert first.trace["reason"] == "cold_start_exploration"
    with pytest.raises(NoProviderError, match="leases"):
        router.route(input_tokens=100)

    first.failed(kind="health", code="timeout")
    available = router.route(input_tokens=100)
    assert available.provider == "cheap"
    snapshot = router.stats()
    assert snapshot.exploration == {"decisions": 2, "target_selected": 2}


def test_exploration_lease_expires_without_a_report() -> None:
    clock = FakeClock()
    router = Router(
        [Provider("only", 1.0, 1.0)],
        clock=clock,
        tuning=Tuning(exploration_lease_sec=2.0),
    )
    router.route(input_tokens=1)
    with pytest.raises(NoProviderError):
        router.route(input_tokens=1)

    clock.advance(2.0)
    assert router.route(input_tokens=1).provider == "only"


def test_exploration_q_zero_is_plain_lp_without_a_lease_or_counter() -> None:
    router = Router(
        [Provider("cheap", 1.0, 1.0), Provider("expensive", 3.0, 3.0)],
        seed=1,
    )
    _warm(router, "cheap", 100.0)

    decision = router.route(input_tokens=10, alpha=0.0)

    assert decision.provider == "cheap"
    assert decision.trace["reason"] == "latency_optimized_lp"
    assert "exploration_target" not in decision.trace
    assert router.stats().exploration == {"decisions": 0, "target_selected": 0}


def test_partial_exploration_mixture_respects_the_request_budget() -> None:
    router = Router(
        [Provider("cheap", 1.0, 1.0), Provider("expensive", 3.0, 3.0)],
        seed=1,
    )
    _warm(router, "cheap", 100.0)

    decision = router.route(input_tokens=10, alpha=0.25)

    assert decision.weights["cheap"] == pytest.approx(0.75)
    assert decision.weights["expensive"] == pytest.approx(0.25)
    assert decision.expected_cost_usd <= float(decision.trace["budget_usd"]) + 1e-12
    assert decision.expected_latency_ms is None


def test_exclude_recomputes_cost_bounds_over_the_remaining_set() -> None:
    providers = [
        Provider("p1", 1.0, 1.0),
        Provider("p2", 2.0, 2.0),
        Provider("p3", 3.0, 3.0),
    ]
    router = Router(providers, cold_start="require_observations")
    for provider in providers:
        _warm(router, provider.name, 100.0)

    decision = router.route(input_tokens=10, alpha=0.0, exclude={"p1"})

    assert decision.provider == "p2"
    assert decision.trace["c_min_provider"] == "p2"
    assert decision.trace["c_max_provider"] == "p3"
    assert decision.trace["budget_usd"] == pytest.approx(decision.trace["c_min_usd"])


def test_stale_exploration_attempt_does_not_release_a_newer_lease() -> None:
    clock = FakeClock()
    router = Router(
        [Provider("only", 1.0, 1.0)],
        clock=clock,
        tuning=Tuning(exploration_lease_sec=1.0),
    )
    stale = router.route(input_tokens=1)
    clock.advance(1.0)
    current = router.route(input_tokens=1)
    current_attempt_id = current.primary._attempt_id

    stale.first_token(ttft_ms=10.0)

    assert router._leases["only"].attempt_id == current_attempt_id


def test_primary_capacity_failure_replans_before_committing_state() -> None:
    providers = [Provider("cheap", 1.0, 1.0), Provider("other", 2.0, 2.0)]
    router = Router(providers, seed=7)
    capacity = RejectingCapacityController({"cheap"})
    router._capacity = capacity

    decision = router.route(input_tokens=10)

    assert capacity.reserve_attempts == ["cheap", "other"]
    assert decision.provider == "other"
    assert decision.weights == {"other": 1.0}
    assert decision.trace["capacity_exclusions"] == ("cheap",)
    assert decision.trace["capacity_replans"] == 1
    stats = router.stats()
    assert stats.providers["cheap"]["primary_selections"] == 0
    assert stats.providers["other"]["primary_selections"] == 1
    assert stats.exploration == {"decisions": 1, "target_selected": 1}


def test_failed_capacity_transaction_rolls_back_rng_and_counters() -> None:
    providers = [Provider("cheap", 1.0, 1.0), Provider("other", 2.0, 2.0)]
    router = Router(providers, seed=11)
    capacity = RejectingCapacityController({"cheap", "other"})
    router._capacity = capacity

    with pytest.raises(NoProviderError, match="capacity"):
        router.route(input_tokens=10)
    stats = router.stats()
    assert sum(int(provider["primary_selections"]) for provider in stats.providers.values()) == 0
    assert stats.exploration == {"decisions": 0, "target_selected": 0}

    capacity.rejected.clear()
    retried = router.route(input_tokens=10)
    fresh = Router(providers, seed=11).route(input_tokens=10)
    assert retried.provider == fresh.provider


def test_completed_billing_migrates_calculated_to_actual_atomically() -> None:
    router = Router([_providers()[0]], cold_start="require_observations")
    _warm(router, "cheap")
    decision = router.route(input_tokens=100, estimated_cached_tokens=20)

    decision.completed(output_tokens=50)
    calculated = router.stats().providers["cheap"]
    assert decision.state == "completed"
    assert calculated["calculated_spend_usd"] == pytest.approx(0.00019)
    assert calculated["actual_spend_usd"] == 0.0
    assert calculated["unsettled_attempts"] == 0

    decision.settle(cost_usd=0.0003)
    actual = router.stats().providers["cheap"]
    assert actual["calculated_spend_usd"] == pytest.approx(0.0)
    assert actual["actual_spend_usd"] == pytest.approx(0.0003)

    decision.settle(cost_usd=0.0003)
    with pytest.raises(OutcomeError):
        decision.settle(cost_usd=0.0004)


def test_unknown_billing_counts_once_then_becomes_calculated() -> None:
    router = Router([_providers()[0]], cold_start="require_observations")
    _warm(router, "cheap")
    decision = router.route(input_tokens=10)

    decision.completed()
    assert router.stats().providers["cheap"]["unsettled_attempts"] == 1
    decision.settle(cached_tokens=20)
    assert router.stats().providers["cheap"]["unsettled_attempts"] == 1
    decision.settle(output_tokens=5)
    stats = router.stats().providers["cheap"]
    assert stats["unsettled_attempts"] == 0
    assert stats["calculated_spend_usd"] == pytest.approx(0.000015)


def test_actual_cost_can_arrive_before_usage_without_being_recomputed() -> None:
    router = Router([_providers()[0]], cold_start="require_observations")
    _warm(router, "cheap")
    decision = router.route(input_tokens=100)

    decision.completed(cost_usd=0.25)
    decision.settle(output_tokens=10, cached_tokens=200)

    stats = router.stats().providers["cheap"]
    assert stats["actual_spend_usd"] == pytest.approx(0.25)
    assert stats["calculated_spend_usd"] == 0.0
    assert router._estimator.global_count == 1


def test_concurrent_conflicting_settlement_commits_one_atomic_delta() -> None:
    router = Router([_providers()[0]], cold_start="require_observations")
    _warm(router, "cheap")
    decision = router.route(input_tokens=100)
    decision.completed()

    def settle(cost: float) -> str:
        try:
            decision.settle(cost_usd=cost)
        except OutcomeError:
            return "conflict"
        return "committed"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(settle, (0.1, 0.2)))

    assert sorted(results) == ["committed", "conflict"]
    actual = float(router.stats().providers["cheap"]["actual_spend_usd"])
    assert actual in {0.1, 0.2}
    decision.settle(cost_usd=actual)
    assert router.stats().providers["cheap"]["actual_spend_usd"] == actual


def test_cancelled_attempt_can_settle_partial_usage_without_training() -> None:
    router = Router([_providers()[0]], cold_start="require_observations")
    _warm(router, "cheap")
    decision = router.route(input_tokens=100)

    decision.cancelled()
    decision.settle(output_tokens=10, cached_tokens=5)

    stats = router.stats().providers["cheap"]
    assert stats["calculated_spend_usd"] > 0.0
    assert stats["unsettled_attempts"] == 0
    assert router._estimator.global_count == 0


def test_typed_lifecycle_is_idempotent_and_rejects_contradictions() -> None:
    router = Router([_providers()[0]], cold_start="require_observations")
    _warm(router, "cheap")
    decision = router.route(input_tokens=1)

    decision.first_token(ttft_ms=12.0)
    decision.first_token(ttft_ms=12.0)
    decision.completed(output_tokens=1)
    decision.completed(output_tokens=1)

    with pytest.raises(OutcomeError):
        decision.failed(kind="health")
    with pytest.raises(OutcomeError):
        decision.settle(output_tokens=2)

    other = router.route(input_tokens=1)
    other.declined()
    with pytest.raises(OutcomeError):
        other.settle(cost_usd=0.0)


def test_health_failure_cooldown_and_request_failure_effects_are_separate() -> None:
    clock = FakeClock()
    router = Router(
        [Provider("only", 1.0, 1.0)],
        cold_start="require_observations",
        clock=clock,
        tuning=Tuning(cooldown_after=2, cooldown_sec=5.0),
    )
    _warm(router, "only")

    router.route(input_tokens=1).failed(kind="request", code="auth")
    router.route(input_tokens=1).failed(kind="health", code="mystery")
    router.route(input_tokens=1).failed(kind="health", code="timeout")
    with pytest.raises(NoProviderError):
        router.route(input_tokens=1)

    snapshot = router.stats().providers["only"]
    assert snapshot["errors"]["request"]["auth"] == 1
    assert snapshot["errors"]["health"]["other"] == 1
    assert snapshot["errors"]["health"]["timeout"] == 1
    assert snapshot["cooldown_remaining_sec"] == pytest.approx(5.0)

    clock.advance(5.0)
    assert router.route(input_tokens=1).provider == "only"


def test_observe_validation_is_atomic() -> None:
    router = Router([Provider("only", 1.0, 1.0)])
    with pytest.raises(ValidationError):
        router.observe("only", ttft_ms=1.0, kind="health")
    with pytest.raises(ValidationError):
        router.observe("missing", ttft_ms=1.0)

    assert router.stats().providers["only"]["ttft_p50_ms"] is None


def test_hedge_attempt_has_independent_lifecycle_and_winner_accounting() -> None:
    router = Router(
        _providers(),
        cold_start="require_observations",
        slo_ms=3000.0,
        seed=1,
    )
    _warm(router, "cheap", 100.0, count=5)
    _warm(router, "fast", 200.0, count=5)
    decision = router.route(input_tokens=100, alpha=1.0)
    assert decision.provider == "cheap"

    assert decision.hedge_now(elapsed_ms=2000.0) is None
    backup = decision.hedge_now(elapsed_ms=2700.0)
    assert backup is not None
    assert backup.provider == "fast"
    assert decision.hedge_now(elapsed_ms=2750.0) is None

    backup.first_token(ttft_ms=210.0, adopted=True)
    backup.completed(output_tokens=0)
    decision.cancelled()
    stats = router.stats()
    assert decision.state == "completed"
    assert stats.hedges["offered"] == 1
    assert stats.hedges["won"] == 1
    assert router._estimator.global_count == 0


def test_hedge_capacity_failure_tries_the_next_feasible_backup() -> None:
    providers = [
        Provider("primary", 1.0, 1.0),
        Provider("backup_a", 2.0, 2.0),
        Provider("backup_b", 3.0, 3.0),
    ]
    router = Router(
        providers,
        cold_start="require_observations",
        slo_ms=3000.0,
        seed=1,
    )
    _warm(router, "primary", 100.0, count=5)
    _warm(router, "backup_a", 200.0, count=5)
    _warm(router, "backup_b", 250.0, count=5)
    capacity = RejectingCapacityController({"backup_a"})
    router._capacity = capacity
    decision = router.route(input_tokens=1, alpha=1.0)

    backup = decision.hedge_now(elapsed_ms=2700.0)

    assert backup is not None
    assert backup.provider == "backup_b"
    assert capacity.reserve_attempts[-2:] == ["backup_a", "backup_b"]
    assert decision.trace["hedge_capacity_exclusions"] == ("backup_a",)


def test_provider_identity_and_decision_outputs_are_read_only() -> None:
    router = Router([Provider("only", 1.0, 1.0)])
    decision = router.route(input_tokens=1)

    with pytest.raises(AttributeError):
        decision.provider = "changed"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        decision.expected_cost_usd = 99.0  # type: ignore[misc]
    with pytest.raises(AttributeError):
        decision.primary.provider = "changed"  # type: ignore[misc]


def test_adopted_backup_failure_controls_the_logical_outcome() -> None:
    router = _hedgeable_router()
    decision = router.route(input_tokens=1, alpha=1.0)
    backup = decision.hedge_now(elapsed_ms=2700.0)
    assert backup is not None

    backup.first_token(ttft_ms=200.0, adopted=True)
    backup.failed(kind="health", code="timeout")
    decision.cancelled()

    assert decision.state == "failed"
    assert router.stats().hedges["won"] == 0


def test_resolution_without_adoption_prefers_failed_then_cancelled() -> None:
    failed_router = _hedgeable_router()
    failed = failed_router.route(input_tokens=1, alpha=1.0)
    failed_backup = failed.hedge_now(elapsed_ms=2700.0)
    assert failed_backup is not None
    failed_backup.declined()
    failed.failed(kind="request", code="bad_request")
    assert failed.state == "failed"

    cancelled_router = _hedgeable_router()
    cancelled = cancelled_router.route(input_tokens=1, alpha=1.0)
    cancelled_backup = cancelled.hedge_now(elapsed_ms=2700.0)
    assert cancelled_backup is not None
    cancelled_backup.declined()
    cancelled.cancelled()
    assert cancelled.state == "cancelled"


def test_hedge_spend_is_a_cross_slice_of_provider_spend() -> None:
    router = _hedgeable_router()
    decision = router.route(input_tokens=1, alpha=1.0)
    backup = decision.hedge_now(elapsed_ms=2700.0)
    assert backup is not None

    backup.completed(cost_usd=0.2, adopted=True)
    decision.cancelled()
    decision.settle(cost_usd=0.1)

    stats = router.stats()
    provider_total = sum(
        float(provider["actual_spend_usd"]) for provider in stats.providers.values()
    )
    assert provider_total == pytest.approx(0.3)
    assert stats.hedges["actual_spend_usd"] == pytest.approx(0.2)


def test_concurrent_hedge_calls_offer_at_most_one_backup() -> None:
    router = _hedgeable_router()
    decision = router.route(input_tokens=1, alpha=1.0)

    with ThreadPoolExecutor(max_workers=8) as pool:
        attempts = list(pool.map(lambda _index: decision.hedge_now(elapsed_ms=2700.0), range(20)))

    assert sum(attempt is not None for attempt in attempts) == 1
    assert router.stats().hedges["offered"] == 1


def test_clock_is_read_at_most_once_per_public_operation() -> None:
    clock = FakeClock()
    router = Router([Provider("only", 1.0, 1.0)], clock=clock)

    before = clock.reads
    decision = router.route(input_tokens=1)
    assert clock.reads - before == 1
    before = clock.reads
    decision.first_token(ttft_ms=10.0)
    assert clock.reads - before == 1
    before = clock.reads
    decision.completed(output_tokens=1)
    assert clock.reads - before <= 1
    before = clock.reads
    router.stats()
    assert clock.reads - before == 1


def test_window_expiry_makes_a_strict_provider_unprofiled_again() -> None:
    clock = FakeClock()
    router = Router(
        [Provider("only", 1.0, 1.0)],
        cold_start="require_observations",
        clock=clock,
        tuning=Tuning(window_min=0.01),
    )
    _warm(router, "only")
    assert router.route(input_tokens=1).provider == "only"

    clock.advance(0.61)
    with pytest.raises(NoProviderError, match="observe"):
        router.route(input_tokens=1)


def test_router_does_not_keep_a_strong_reference_to_decisions() -> None:
    router = Router([Provider("only", 1.0, 1.0)])
    decision = router.route(input_tokens=1)
    reference = weakref.ref(decision)

    del decision
    gc.collect()

    assert reference() is None


def test_no_cache_discount_and_missing_cache_mapping_price_correctly() -> None:
    router = Router(
        [
            Provider("full_price", price_in=1.0, price_out=0.0),
            Provider("discount", price_in=1.0, price_out=0.0, price_cached=0.0),
        ],
        seed=1,
    )

    decision = router.route(
        input_tokens=100,
        estimated_cached_tokens={"full_price": 100},
        alpha=1.0,
    )

    assert decision.trace["c_min_provider"] == "discount"
    assert decision.trace["c_min_usd"] == pytest.approx(0.0001)
    assert decision.trace["c_max_usd"] == pytest.approx(0.0001)


def test_multiple_attempt_completion_without_adoption_is_unresolved() -> None:
    router = Router(
        _providers(),
        cold_start="require_observations",
        slo_ms=3000.0,
    )
    _warm(router, "cheap", 100.0, count=5)
    _warm(router, "fast", 200.0, count=5)
    decision = router.route(input_tokens=1, alpha=1.0)
    assert decision.hedge_now(elapsed_ms=2000.0) is None
    backup = decision.hedge_now(elapsed_ms=2700.0)
    assert backup is not None

    decision.completed(output_tokens=1)
    backup.declined()
    assert decision.state == "unresolved"
    assert router.stats().decisions_without_adoption == 1


def test_primary_first_token_closes_unused_hedge_slot() -> None:
    router = Router(
        _providers(),
        cold_start="require_observations",
        slo_ms=3000.0,
    )
    _warm(router, "cheap", 100.0, count=5)
    _warm(router, "fast", 200.0, count=5)
    decision = router.route(input_tokens=1, alpha=1.0)

    decision.first_token(ttft_ms=120.0)
    assert decision.hedge_now(elapsed_ms=2000.0) is None


def test_backward_clock_is_clamped_and_stats_are_deeply_immutable() -> None:
    clock = FakeClock(10.0)
    router = Router([Provider("only", 1.0, 1.0)], clock=clock)
    decision = router.route(input_tokens=1)
    clock.now = 5.0
    decision.completed(ttft_ms=10.0, output_tokens=1)

    snapshot = router.stats()
    assert snapshot.providers["only"]["ttft_p50_ms"] == 10.0
    with pytest.raises(TypeError):
        snapshot.providers["only"]["errors"]["health"]["x"] = 1  # type: ignore[index]


def test_streaming_adoption_cannot_be_declared_at_completion() -> None:
    router = _hedgeable_router()
    decision = router.route(input_tokens=1, alpha=1.0)
    backup = decision.hedge_now(elapsed_ms=2700.0)
    assert backup is not None

    backup.first_token(ttft_ms=200.0)
    with pytest.raises(OutcomeError, match="declared on first_token"):
        backup.completed(output_tokens=1, adopted=True)

    assert backup.state == "streaming"
    assert decision.state == "pending"
    assert router.stats().hedges["won"] == 0


def test_derived_cost_overflow_is_rejected_before_state_changes() -> None:
    with pytest.raises(ValidationError, match="derived request cost"):
        Router([Provider("huge", 1e308, 1e308)]).route(input_tokens=2)

    router = Router([Provider("only", 1.0, 1.0)])
    decision = router.route(input_tokens=1)
    with pytest.raises(ValidationError, match="derived request cost"):
        decision.completed(output_tokens=10**400)

    assert decision.state == "pending"
    assert decision.primary.state == "pending"
    assert decision.primary.billing_state == "unknown"
    assert router.stats().providers["only"]["unsettled_attempts"] == 0

    decision.completed(output_tokens=1)
    assert decision.state == "completed"
