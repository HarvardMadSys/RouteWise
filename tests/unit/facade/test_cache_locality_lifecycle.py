#!/usr/bin/env python3
"""Lifecycle tests for cache-locality observations and hedges."""

from __future__ import annotations

import pytest

from llm_routewise._capacity_controller import (
    _CapacitySnapshot,
    _NoopReservation,
)
from llm_routewise.facade import OutcomeError, Provider, Router


class RejectingCapacityController:
    def __init__(self, rejected: set[str]) -> None:
        self.rejected = rejected
        self.reserve_attempts: list[str] = []

    def snapshot(self, *, resource_key: str, now: float) -> _CapacitySnapshot:
        return _CapacitySnapshot(resource_key=resource_key, observed_at=now)

    def try_reserve(
        self, *, resource_key: str, attempt_id: str, snapshot: _CapacitySnapshot
    ) -> _NoopReservation | None:
        self.reserve_attempts.append(resource_key)
        if resource_key in self.rejected:
            return None
        return _NoopReservation(resource_key=resource_key, attempt_id=attempt_id)


class DeterministicClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _warm(router: Router, provider: str, value: float = 100.0, count: int = 5) -> None:
    for _ in range(count):
        router.observe(provider, ttft_ms=value)


# ---------------------------------------------------------------------------
# Tests: observation lifecycle
# ---------------------------------------------------------------------------


class TestAdversarialEstimateVsActual:
    """Proof that the system learns from ACTUAL completion data, not from
    routing-time estimates."""

    def test_predicted_warm_actual_miss_degrades_evidence(self) -> None:
        """Predicted 90 warm, actual 0 miss — degrades evidence."""
        clock = DeterministicClock()
        router = Router(
            [
                Provider("A", price_in=2.0, price_out=1.0, price_cached=0.2),
                Provider("B", price_in=1.0, price_out=1.0),
            ],
            cold_start="require_observations",
            seed=1,
            clock=clock,
        )
        _warm(router, "A", 100.0, 5)
        _warm(router, "B", 100.0, 5)

        # Inject positive evidence for A
        router._locality_estimator.record(
            "A", "prefix_X", cached_tokens=90, input_tokens=100, now=clock.now
        )
        old_evidence = router._locality_estimator.estimate("A", "prefix_X", 100, clock.now)
        assert old_evidence == 90

        # Route with affinity_key (A wins because of evidence)
        d1 = router.route(input_tokens=100, affinity_key="prefix_X", estimated_output_tokens=10)
        assert d1.provider == "A"

        # Complete with ACTUAL MISS (cached_tokens=0)
        d1.completed(output_tokens=10, cached_tokens=0)

        # Evidence should be DEGRADED (miss reduces confidence) but NOT destroyed
        new_evidence = router._locality_estimator.estimate("A", "prefix_X", 100, clock.now)
        assert new_evidence < 90, (
            f"Predicted warm + actual miss should degrade evidence. Expected < 90, got {new_evidence}"
        )
        assert new_evidence > 0, (
            f"Predicted warm + actual miss should not destroy evidence. Expected > 0, got {new_evidence}"
        )

    def test_predicted_cold_actual_hit_records_evidence(self) -> None:
        """Predicted 0 cold, actual 90 hit — MUST record positive evidence."""
        clock = DeterministicClock()
        router = Router(
            [
                Provider("A", price_in=2.0, price_out=1.0, price_cached=0.2),
                Provider("B", price_in=1.0, price_out=1.0),
            ],
            cold_start="require_observations",
            seed=1,
            clock=clock,
        )
        _warm(router, "A", 100.0, 5)
        _warm(router, "B", 100.0, 5)

        # Route with affinity key; no prior evidence exists
        d1 = router.route(input_tokens=100, affinity_key="prefix_X", estimated_output_tokens=10)
        # A is more expensive; B wins on cost
        assert d1.provider == "B"

        # Complete with ACTUAL HIT: 90 cached tokens
        d1.completed(output_tokens=10, cached_tokens=90)

        # Evidence must be the actual 90, not 0 (route-time estimate)
        evidence = router._locality_estimator.estimate("B", "prefix_X", 100, clock.now)
        assert evidence == 90, f"Actual hit of 90 should be recorded, got {evidence}"

    def test_prediction_differs_from_observation_records_actual(self) -> None:
        """Route estimate = 90, actual cached_tokens = 30 — evidence based on 30."""
        clock = DeterministicClock()
        router = Router(
            [
                Provider("A", price_in=2.0, price_out=1.0, price_cached=0.2),
                Provider("B", price_in=1.0, price_out=1.0),
            ],
            cold_start="require_observations",
            seed=1,
            clock=clock,
        )
        _warm(router, "A", 100.0, 5)
        _warm(router, "B", 100.0, 5)

        # Make A preferred via evidence
        router._locality_estimator.record(
            "A", "prefix_X", cached_tokens=90, input_tokens=100, now=clock.now
        )
        d1 = router.route(input_tokens=100, affinity_key="prefix_X", estimated_output_tokens=10)
        assert d1.provider == "A"

        # Actual completion: only 30 cached tokens (not the 90 that was expected)
        d1.completed(output_tokens=10, cached_tokens=30)

        # Evidence should be 30 (actual), not 90 (prediction)
        evidence = router._locality_estimator.estimate("A", "prefix_X", 100, clock.now)
        assert evidence == 30, (
            f"Evidence should reflect actual 30, not prediction 90. Got {evidence}"
        )

    def test_caller_estimate_affects_pricing_not_observations(self) -> None:
        """Call-time estimated_cached_tokens=100 affects pricing;
        actual cached_tokens=20 affects future evidence."""
        clock = DeterministicClock()
        router = Router(
            [
                Provider("A", price_in=2.0, price_out=1.0, price_cached=0.2),
                Provider("B", price_in=1.0, price_out=1.0),
            ],
            cold_start="require_observations",
            seed=1,
            clock=clock,
        )
        _warm(router, "A", 100.0, 5)
        _warm(router, "B", 100.0, 5)

        # Caller says A has 100 cached (affects pricing, not evidence)
        d1 = router.route(
            input_tokens=100,
            affinity_key="prefix_X",
            estimated_output_tokens=10,
            estimated_cached_tokens={"A": 100},
        )
        assert d1.provider == "A"

        # Actual completion: only 20 cached tokens
        d1.completed(output_tokens=10, cached_tokens=20)

        # Future evidence must be 20 (actual), not 100 (caller estimate)
        evidence = router._locality_estimator.estimate("A", "prefix_X", 100, clock.now)
        assert evidence == 20, (
            f"Future evidence must be actual 20, not caller estimate 100. Got {evidence}"
        )

    def test_completed_none_cached_tokens_no_evidence(self) -> None:
        """completed(cached_tokens=None) produces no positive evidence."""
        clock = DeterministicClock()
        router = Router(
            [
                Provider("A", price_in=2.0, price_out=1.0, price_cached=0.2),
                Provider("B", price_in=1.0, price_out=1.0),
            ],
            cold_start="require_observations",
            seed=1,
            clock=clock,
        )
        _warm(router, "A", 100.0, 5)
        _warm(router, "B", 100.0, 5)

        d1 = router.route(input_tokens=100, affinity_key="prefix_X", estimated_output_tokens=10)
        d1.completed(output_tokens=10, cached_tokens=None)

        # No evidence should be created
        assert router._locality_estimator.evidence_count == 0

    def test_learned_cache_hit_does_not_understate_unknown_calculated_spend(self) -> None:
        """A learned hit must not become confirmed billing when usage is unknown."""
        clock = DeterministicClock()
        router = Router(
            [Provider("A", price_in=2.0, price_out=1.0, price_cached=0.2)],
            cold_start="require_observations",
            seed=1,
            clock=clock,
        )
        _warm(router, "A", 100.0, 5)
        router._locality_estimator.record(
            "A", "prefix_X", cached_tokens=90, input_tokens=100, now=clock.now
        )

        decision = router.route(
            input_tokens=100, affinity_key="prefix_X", estimated_output_tokens=10
        )
        assert decision._estimated_cached_tokens["A"] == 90
        decision.completed(output_tokens=10, cached_tokens=None)

        # Routing may use the learned hit, but accounting falls back to the
        # uncached price until the provider reports actual usage.
        expected_uncached = (2.0 * 100 + 1.0 * 10) / 1_000_000.0
        assert router.stats().providers["A"]["calculated_spend_usd"] == pytest.approx(
            expected_uncached
        )

    def test_duplicate_completion_idempotent(self) -> None:
        """Calling completed() twice with same values is idempotent."""
        clock = DeterministicClock()
        router = Router(
            [
                Provider("A", price_in=2.0, price_out=1.0, price_cached=0.2),
                Provider("B", price_in=1.0, price_out=1.0),
            ],
            cold_start="require_observations",
            seed=1,
            clock=clock,
        )
        _warm(router, "A", 100.0, 5)
        _warm(router, "B", 100.0, 5)

        router._locality_estimator.record(
            "A", "prefix_X", cached_tokens=90, input_tokens=100, now=clock.now
        )
        d1 = router.route(input_tokens=100, affinity_key="prefix_X", estimated_output_tokens=10)
        assert d1.provider == "A"

        # First completion with actual 90
        d1.completed(output_tokens=10, cached_tokens=90)
        evidence_after_first = router._locality_estimator.estimate("A", "prefix_X", 100, clock.now)

        # Second completion with same values (idempotent)
        d1.completed(output_tokens=10, cached_tokens=90)
        evidence_after_second = router._locality_estimator.estimate("A", "prefix_X", 100, clock.now)

        assert evidence_after_second == evidence_after_first == 90

    def test_duplicate_completion_different_values_raises(self) -> None:
        """Calling completed() twice with DIFFERENT values raises OutcomeError."""
        clock = DeterministicClock()
        router = Router(
            [Provider("A", price_in=2.0, price_out=1.0, price_cached=0.2)],
            cold_start="require_observations",
            seed=1,
            clock=clock,
        )
        _warm(router, "A", 100.0, 5)

        d1 = router.route(input_tokens=100, estimated_output_tokens=10)
        d1.completed(output_tokens=10, cached_tokens=90)

        with pytest.raises(OutcomeError, match="cached_tokens was already settled"):
            d1.completed(output_tokens=10, cached_tokens=0)


# ---------------------------------------------------------------------------
# Baseline vs Candidate experiment
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests: hedge lifecycle
# ---------------------------------------------------------------------------


class TestHedgeLocality:
    """Hedge attempts should interact correctly with locality learning."""

    def test_completed_backup_contributes_evidence(self) -> None:
        """Completed backup with actual cached_tokens records evidence."""
        clock = DeterministicClock()
        router = Router(
            [
                Provider("primary", price_in=1.0, price_out=1.0, price_cached=0.1),
                Provider("backup", price_in=1.0, price_out=1.0, price_cached=0.1),
            ],
            cold_start="require_observations",
            slo_ms=3000.0,
            seed=1,
            clock=clock,
        )
        _warm(router, "primary", 100.0, 5)
        _warm(router, "backup", 200.0, 5)
        decision = router.route(
            input_tokens=100, affinity_key="prefix_X", estimated_output_tokens=10
        )
        assert decision.provider == "primary"
        backup = decision.hedge_now(elapsed_ms=2700.0)
        assert backup is not None
        backup.completed(output_tokens=10, cached_tokens=80)
        decision.cancelled()
        # Evidence for backup provider should be recorded
        assert router._locality_estimator.estimate("backup", "prefix_X", 100, clock.now) > 0

    def test_cancelled_hedge_no_evidence(self) -> None:
        """Cancelled hedge doesn't produce locality evidence."""
        clock = DeterministicClock()
        router = Router(
            [
                Provider("primary", price_in=1.0, price_out=1.0, price_cached=0.1),
                Provider("backup", price_in=1.0, price_out=1.0, price_cached=0.1),
            ],
            cold_start="require_observations",
            slo_ms=3000.0,
            seed=1,
            clock=clock,
        )
        _warm(router, "primary", 100.0, 5)
        _warm(router, "backup", 200.0, 5)
        decision = router.route(
            input_tokens=100, affinity_key="prefix_X", estimated_output_tokens=10
        )
        backup = decision.hedge_now(elapsed_ms=2700.0)
        assert backup is not None
        backup.cancelled()
        decision.completed(output_tokens=10, cached_tokens=0)
        assert router._locality_estimator.estimate("backup", "prefix_X", 100, clock.now) == 0

    def test_failed_hedge_no_evidence(self) -> None:
        """Failed hedge doesn't produce locality evidence."""
        clock = DeterministicClock()
        router = Router(
            [
                Provider("primary", price_in=1.0, price_out=1.0, price_cached=0.1),
                Provider("backup", price_in=1.0, price_out=1.0, price_cached=0.1),
            ],
            cold_start="require_observations",
            slo_ms=3000.0,
            seed=1,
            clock=clock,
        )
        _warm(router, "primary", 100.0, 5)
        _warm(router, "backup", 200.0, 5)
        decision = router.route(
            input_tokens=100, affinity_key="prefix_X", estimated_output_tokens=10
        )
        backup = decision.hedge_now(elapsed_ms=2700.0)
        assert backup is not None
        backup.failed(kind="request", code="timeout")
        decision.completed(output_tokens=10, cached_tokens=0)
        assert router._locality_estimator.estimate("backup", "prefix_X", 100, clock.now) == 0

    def test_settle_on_completed_backup(self) -> None:
        """Late settle(cached_tokens=...) on completed backup trains locality."""
        clock = DeterministicClock()
        router = Router(
            [
                Provider("primary", price_in=1.0, price_out=1.0, price_cached=0.1),
                Provider("backup", price_in=1.0, price_out=1.0, price_cached=0.1),
            ],
            cold_start="require_observations",
            slo_ms=3000.0,
            seed=1,
            clock=clock,
        )
        _warm(router, "primary", 100.0, 5)
        _warm(router, "backup", 200.0, 5)
        decision = router.route(
            input_tokens=100, affinity_key="prefix_X", estimated_output_tokens=10
        )
        backup = decision.hedge_now(elapsed_ms=2700.0)
        assert backup is not None
        backup.completed(output_tokens=10, cached_tokens=None)
        decision.cancelled()

        # No evidence yet (cached_tokens=None)
        assert router._locality_estimator.estimate("backup", "prefix_X", 100, clock.now) == 0

        # Settle with actual cached_tokens
        backup.settle(cached_tokens=80)

        # Now evidence should be recorded
        assert router._locality_estimator.estimate("backup", "prefix_X", 100, clock.now) > 0
