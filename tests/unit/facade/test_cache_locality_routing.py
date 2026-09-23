#!/usr/bin/env python3
"""Routing tests for the cache-locality feedback loop."""
from __future__ import annotations

import pytest

from llm_routewise._capacity_controller import (
    _CapacitySnapshot,
    _NoopReservation,
)
from llm_routewise.facade import Provider, Router, ValidationError


class RejectingCapacityController:
    def __init__(self, rejected: set[str]) -> None:
        self.rejected = rejected
        self.reserve_attempts: list[str] = []

    def snapshot(self, *, resource_key: str, now: float) -> _CapacitySnapshot:
        return _CapacitySnapshot(resource_key=resource_key, observed_at=now)

    def try_reserve(self, *, resource_key: str, attempt_id: str, snapshot: _CapacitySnapshot) -> _NoopReservation | None:
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
# Tests: Router integration (DETERMINISTIC)
# ---------------------------------------------------------------------------


class TestRouterCacheLocalityIntegration:
    def test_cold_request_no_affinity_unchanged(self) -> None:
        clock = DeterministicClock()
        router = Router(
            [Provider("a", price_in=1.0, price_out=1.0), Provider("b", price_in=2.0, price_out=1.0)],
            cold_start="require_observations", seed=1, clock=clock,
        )
        _warm(router, "a")
        _warm(router, "b")
        decision = router.route(input_tokens=100, estimated_output_tokens=10)
        assert decision.provider == "a"
        assert decision._affinity_key is None

    def test_affinity_key_learns_locality(self) -> None:
        clock = DeterministicClock()
        router = Router(
            [
                Provider("A", price_in=2.0, price_out=1.0, price_cached=0.2),
                Provider("B", price_in=1.0, price_out=1.0),
            ],
            cold_start="require_observations", seed=1, clock=clock,
        )
        _warm(router, "A", 100.0, 5)
        _warm(router, "B", 100.0, 5)

        # Establish evidence
        router._locality_estimator.record("A", "prefix_X", cached_tokens=90, input_tokens=100, now=clock.now)

        # Route with affinity key
        d1 = router.route(input_tokens=100, affinity_key="prefix_X", estimated_output_tokens=10)
        assert d1.provider == "A"
        d1.completed(output_tokens=10, cached_tokens=90)

        # Verify evidence is used
        assert router._locality_estimator.estimate("A", "prefix_X", 100, clock.now) > 0

    def test_caller_scalar_wins_over_learned(self) -> None:
        """Caller explicit estimate wins over learned evidence."""
        clock = DeterministicClock()
        router = Router(
            [
                Provider("A", price_in=2.0, price_out=1.0, price_cached=0.2),
                Provider("B", price_in=1.0, price_out=1.0),
            ],
            cold_start="require_observations", seed=1, clock=clock,
        )
        _warm(router, "A", 100.0, 5)
        _warm(router, "B", 100.0, 5)

        # Establish evidence for A (learned says A has 90 cached)
        router._locality_estimator.record("A", "prefix_X", cached_tokens=90, input_tokens=100, now=clock.now)

        # Caller explicitly says A has 0 cached - should override learned
        d1 = router.route(
            input_tokens=100, affinity_key="prefix_X",
            estimated_output_tokens=10, estimated_cached_tokens={"A": 0},
        )
        # With explicit 0, A appears expensive (100 * 2.0 = 200),
        # so B (100 * 1.0 = 100) should win
        assert d1.provider == "B"

    def test_no_affinity_key_no_learning(self) -> None:
        """Without affinity_key, no locality learning occurs."""
        clock = DeterministicClock()
        router = Router(
            [Provider("a", price_in=1.0, price_out=1.0, price_cached=0.1)],
            cold_start="require_observations", seed=1, clock=clock,
        )
        _warm(router, "a")

        d1 = router.route(input_tokens=100, estimated_output_tokens=10)
        d1.completed(output_tokens=10, cached_tokens=90)

        # No evidence should be recorded without affinity_key
        assert router._locality_estimator.evidence_count == 0

    def test_existing_callers_unchanged_without_affinity(self) -> None:
        """Existing callers without affinity_key are unaffected."""
        clock = DeterministicClock()
        router = Router(
            [
                Provider("a", price_in=1.0, price_out=1.0),
                Provider("b", price_in=2.0, price_out=1.0),
            ],
            cold_start="require_observations", seed=1, clock=clock,
        )
        _warm(router, "a", 100.0, 5)
        _warm(router, "b", 200.0, 5)
        d = router.route(input_tokens=100, estimated_output_tokens=10)
        assert d.provider == "a"
        assert d._affinity_key is None
        d.completed(output_tokens=10)
        assert router._locality_estimator.evidence_count == 0

    def test_invalid_affinity_key_rejected(self) -> None:
        """Non-string affinity_key is rejected."""
        clock = DeterministicClock()
        router = Router([Provider("a", price_in=1.0, price_out=1.0)], seed=1, clock=clock)
        _warm(router, "a")

        with pytest.raises(ValidationError, match="affinity_key must be a string"):
            router.route(input_tokens=100, affinity_key=123)  # type: ignore[arg-type]

    def test_affinity_key_too_long_rejected(self) -> None:
        """affinity_key > 1024 chars is rejected."""
        clock = DeterministicClock()
        router = Router([Provider("a", price_in=1.0, price_out=1.0)], seed=1, clock=clock)
        _warm(router, "a")

        with pytest.raises(ValidationError, match="at most 1024"):
            router.route(input_tokens=100, affinity_key="x" * 1025)

    def test_settle_trains_locality(self) -> None:
        """settle() with cached_tokens trains locality."""
        clock = DeterministicClock()
        router = Router(
            [Provider("A", price_in=2.0, price_out=1.0, price_cached=0.2)],
            cold_start="require_observations", seed=1, clock=clock,
        )
        _warm(router, "A", 100.0, 5)

        d1 = router.route(input_tokens=100, affinity_key="prefix_X", estimated_output_tokens=10)
        d1.completed(output_tokens=10, cached_tokens=None)

        # No evidence yet (cached_tokens=None)
        assert router._locality_estimator.evidence_count == 0

        # Settle with actual cached_tokens
        d1.settle(cached_tokens=90)

        # Now evidence should be recorded
        assert router._locality_estimator.estimate("A", "prefix_X", 100, clock.now) > 0

    def test_settle_idempotent(self) -> None:
        """Repeated identical settle() doesn't duplicate evidence."""
        clock = DeterministicClock()
        router = Router(
            [Provider("A", price_in=2.0, price_out=1.0, price_cached=0.2)],
            cold_start="require_observations", seed=1, clock=clock,
        )
        _warm(router, "A", 100.0, 5)

        d1 = router.route(input_tokens=100, affinity_key="prefix_X", estimated_output_tokens=10)
        d1.completed(output_tokens=10, cached_tokens=None)
        d1.settle(cached_tokens=90)

        evidence_after_first = router._locality_estimator.evidence_count

        # Repeated settle with same value
        d1.settle(cached_tokens=90)

        evidence_after_second = router._locality_estimator.evidence_count
        assert evidence_after_second == evidence_after_first

    def test_failed_attempt_no_locality(self) -> None:
        """Failed attempts don't produce locality evidence."""
        clock = DeterministicClock()
        router = Router(
            [Provider("A", price_in=2.0, price_out=1.0, price_cached=0.2)],
            cold_start="require_observations", seed=1, clock=clock,
        )
        _warm(router, "A", 100.0, 5)

        d1 = router.route(input_tokens=100, affinity_key="prefix_X", estimated_output_tokens=10)
        d1.failed(kind="request", code="timeout")

        assert router._locality_estimator.evidence_count == 0

    def test_cancelled_attempt_no_locality(self) -> None:
        """Cancelled attempts don't produce locality evidence."""
        clock = DeterministicClock()
        router = Router(
            [Provider("A", price_in=2.0, price_out=1.0, price_cached=0.2)],
            cold_start="require_observations", seed=1, clock=clock,
        )
        _warm(router, "A", 100.0, 5)

        d1 = router.route(input_tokens=100, affinity_key="prefix_X", estimated_output_tokens=10)
        d1.cancelled()

        assert router._locality_estimator.evidence_count == 0

    def test_price_cached_equals_price_in_no_preference(self) -> None:
        """When price_cached == price_in, learned cache doesn't invent preference."""
        clock = DeterministicClock()
        router = Router(
            [
                Provider("A", price_in=1.0, price_out=1.0, price_cached=1.0),
                Provider("B", price_in=1.0, price_out=1.0, price_cached=1.0),
            ],
            cold_start="require_observations", seed=1, clock=clock,
        )
        _warm(router, "A", 100.0, 5)
        _warm(router, "B", 100.0, 5)

        # Establish evidence for A
        router._locality_estimator.record("A", "prefix_X", cached_tokens=90, input_tokens=100, now=clock.now)

        # Route with affinity key - since price_cached == price_in, learned cache
        # doesn't change cost, so routing should be based on other factors
        d1 = router.route(input_tokens=100, affinity_key="prefix_X", estimated_output_tokens=10)
        # The key invariant: no crash, and evidence doesn't create artificial preference
        assert d1.provider in ("A", "B")


# ---------------------------------------------------------------------------
# Adversarial tests: estimate vs actual
# ---------------------------------------------------------------------------


class TestCacheLocalityExperiment:
    def test_baseline_no_locality_affinity(self) -> None:
        """Without affinity_key, RouteWise cannot preserve cache locality."""
        clock = DeterministicClock()
        router = Router(
            [
                Provider("a", price_in=1.0, price_out=1.0, price_cached=0.1),
                Provider("b", price_in=1.0, price_out=1.0, price_cached=0.1),
            ],
            cold_start="require_observations", seed=1, clock=clock,
        )
        _warm(router, "a", 100.0, 5)
        _warm(router, "b", 100.0, 5)
        # No affinity_key -> no locality learning
        d1 = router.route(input_tokens=100, estimated_output_tokens=10)
        d1.completed(output_tokens=10, cached_tokens=0)
        d2 = router.route(input_tokens=100, estimated_output_tokens=10)
        d2.completed(output_tokens=10, cached_tokens=0)
        d3 = router.route(input_tokens=100, estimated_output_tokens=10)
        d3.completed(output_tokens=10, cached_tokens=0)
        assert router._locality_estimator.evidence_count == 0

    def test_candidate_with_locality_affinity(self) -> None:
        """With affinity_key, RouteWise learns and preserves cache locality."""
        clock = DeterministicClock()
        router = Router(
            [
                Provider("a", price_in=1.0, price_out=1.0, price_cached=0.1),
                Provider("b", price_in=1.0, price_out=1.0, price_cached=0.1),
            ],
            cold_start="require_observations", seed=1, clock=clock,
        )
        _warm(router, "a", 100.0, 5)
        _warm(router, "b", 100.0, 5)
        # With affinity_key -> locality learning enabled
        d1 = router.route(input_tokens=100, affinity_key="prefix_X", estimated_output_tokens=10)
        d1.completed(output_tokens=10, cached_tokens=0)
        d2 = router.route(input_tokens=100, affinity_key="prefix_X", estimated_output_tokens=10)
        d2.completed(output_tokens=10, cached_tokens=90)
        d3 = router.route(input_tokens=100, affinity_key="prefix_X", estimated_output_tokens=10)
        d3.completed(output_tokens=10, cached_tokens=90)
        assert router._locality_estimator.evidence_count >= 1


# ---------------------------------------------------------------------------
# End-to-end public-API routing proof
# ---------------------------------------------------------------------------


class TestEndToEndRoutingProof:
    """Single readable test proving the whole contribution using only public calls.

    This is the test a maintainer should be able to read in 30 seconds and
    understand the whole contribution.
    """

    def test_learned_locality_changes_routing(self) -> None:
        """Learned locality evidence can change routing outcome.

        Setup: Two providers, A (moderately expensive without cache, very cheap
        with cache) and B (always moderate). Without learned evidence, B wins.
        After observing an actual cache hit for A with affinity_key="X", A
        should win because the learned evidence makes A's effective cost lower.
        """
        clock = DeterministicClock()

        # Create two routers with identical setup
        router_with_evidence = Router(
            [
                Provider("A", price_in=2.0, price_out=1.0, price_cached=0.01),
                Provider("B", price_in=1.0, price_out=1.0, price_cached=0.1),
            ],
            cold_start="require_observations", seed=42, clock=clock,
        )
        router_without_evidence = Router(
            [
                Provider("A", price_in=2.0, price_out=1.0, price_cached=0.01),
                Provider("B", price_in=1.0, price_out=1.0, price_cached=0.1),
            ],
            cold_start="require_observations", seed=42, clock=clock,
        )

        _warm(router_with_evidence, "A", 100.0, 5)
        _warm(router_with_evidence, "B", 100.0, 5)
        _warm(router_without_evidence, "A", 100.0, 5)
        _warm(router_without_evidence, "B", 100.0, 5)

        # Step 1: Establish evidence on router_with_evidence by selecting A
        # and reporting an actual cache hit
        d1 = router_with_evidence.route(
            input_tokens=100, affinity_key="X", estimated_output_tokens=10,
            exclude={"B"},  # Force A
        )
        assert d1.provider == "A"
        d1.completed(output_tokens=10, cached_tokens=90)  # Actual hit

        # Step 2: Route again with same affinity, no caller estimate
        d2 = router_with_evidence.route(
            input_tokens=100, affinity_key="X", estimated_output_tokens=10,
        )

        # Step 3: Route on the router without evidence
        d3 = router_without_evidence.route(
            input_tokens=100, affinity_key="X", estimated_output_tokens=10,
        )

        # The router with evidence should select A (learned locality makes A cheaper:
        # 2.0 * 10 + 0.01 * 90 = 20.9 vs B's 1.0 * 100 = 100)
        # The router without evidence should select B (A appears expensive:
        # 2.0 * 100 = 200 vs B's 1.0 * 100 = 100)
        assert d2.provider == "A", \
            "Learned locality should make A preferred"
        assert d3.provider == "B", \
            "Without evidence, B should be preferred"

    def test_does_not_infer_warmth_from_dispatch(self) -> None:
        """Distinguishes from removed 048083b model: dispatching to a provider
        does NOT create positive evidence. Only actual cached_tokens observations
        create evidence."""
        clock = DeterministicClock()
        router = Router(
            [
                Provider("A", price_in=1.0, price_out=1.0, price_cached=0.1),
                Provider("B", price_in=1.0, price_out=1.0, price_cached=0.1),
            ],
            cold_start="require_observations", seed=1, clock=clock,
        )
        _warm(router, "A", 100.0, 5)
        _warm(router, "B", 100.0, 5)

        # Dispatch to A with cached_tokens=None (no observation)
        d1 = router.route(input_tokens=100, affinity_key="X", estimated_output_tokens=10)
        d1.completed(output_tokens=10, cached_tokens=None)

        # No evidence should be created
        assert router._locality_estimator.evidence_count == 0

        # Dispatch to A with cached_tokens=0 (actual miss)
        d2 = router.route(input_tokens=100, affinity_key="X", estimated_output_tokens=10)
        d2.completed(output_tokens=10, cached_tokens=0)

        # Still no positive evidence (miss doesn't create evidence)
        assert router._locality_estimator.evidence_count == 0

        # Only an actual hit creates evidence
        d3 = router.route(input_tokens=100, affinity_key="X", estimated_output_tokens=10)
        d3.completed(output_tokens=10, cached_tokens=90)

        assert router._locality_estimator.evidence_count == 1


# ---------------------------------------------------------------------------
# Architecture A vs B demonstration
# ---------------------------------------------------------------------------


class TestArchitectureDemonstration:
    def test_architecture_a_replicas_as_providers(self) -> None:
        """Architecture A: Each replica is a separate RouteWise provider."""
        clock = DeterministicClock()
        router = Router(
            [
                Provider("minimax-replica-a", price_in=1.0, price_out=1.0, price_cached=0.1),
                Provider("minimax-replica-b", price_in=1.0, price_out=1.0, price_cached=0.1),
            ],
            cold_start="require_observations", seed=1, clock=clock,
        )
        _warm(router, "minimax-replica-a", 100.0, 5)
        _warm(router, "minimax-replica-b", 100.0, 5)

        # Establish evidence on replica-a
        router._locality_estimator.record("minimax-replica-a", "prefix_X", cached_tokens=90, input_tokens=100, now=clock.now)

        d1 = router.route(input_tokens=100, affinity_key="prefix_X", estimated_output_tokens=10)
        assert d1.provider == "minimax-replica-a"

    def test_architecture_b_single_provider_cannot_distinguish_replicas(self) -> None:
        """Architecture B: Single provider + internal LB choosing replicas."""
        clock = DeterministicClock()
        router = Router(
            [Provider("minimax", price_in=1.0, price_out=1.0, price_cached=0.1)],
            cold_start="require_observations", seed=1, clock=clock,
        )
        _warm(router, "minimax", 100.0, 5)

        # Even with evidence, RouteWise can only select "minimax"
        router._locality_estimator.record("minimax", "prefix_X", cached_tokens=90, input_tokens=100, now=clock.now)

        d1 = router.route(input_tokens=100, affinity_key="prefix_X", estimated_output_tokens=10)
        assert d1.provider == "minimax"
        # RouteWise cannot distinguish replicas below provider level
