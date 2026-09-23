"""Focused tests for the internal cache-locality evidence primitive."""

from __future__ import annotations

import threading

from llm_routewise._cache_locality import _CacheLocalityEstimator


class DeterministicClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ---------------------------------------------------------------------------
# Tests: _CacheLocalityEstimator
# ---------------------------------------------------------------------------


class TestCacheLocalityEstimator:
    def test_record_and_estimate(self) -> None:
        clock = DeterministicClock()
        est = _CacheLocalityEstimator(ttl_sec=300.0)
        est.record("a", "prefix_X", cached_tokens=90, input_tokens=100, now=clock.now)
        assert est.estimate("a", "prefix_X", 100, clock.now) == 90

    def test_estimate_unknown_returns_zero(self) -> None:
        clock = DeterministicClock()
        est = _CacheLocalityEstimator(ttl_sec=300.0)
        assert est.estimate("a", "prefix_X", 100, clock.now) == 0

    def test_estimate_expires_after_ttl(self) -> None:
        clock = DeterministicClock()
        est = _CacheLocalityEstimator(ttl_sec=300.0)
        est.record("a", "prefix_X", cached_tokens=90, input_tokens=100, now=clock.now)
        clock.advance(301.0)
        assert est.estimate("a", "prefix_X", 100, clock.now) == 0

    def test_estimate_never_exceeds_observed(self) -> None:
        clock = DeterministicClock()
        est = _CacheLocalityEstimator(ttl_sec=300.0)
        est.record("a", "prefix_X", cached_tokens=50, input_tokens=100, now=clock.now)
        assert est.estimate("a", "prefix_X", 200, clock.now) == 50

    def test_estimate_never_exceeds_current_input(self) -> None:
        clock = DeterministicClock()
        est = _CacheLocalityEstimator(ttl_sec=300.0)
        est.record("a", "prefix_X", cached_tokens=100, input_tokens=100, now=clock.now)
        assert est.estimate("a", "prefix_X", 50, clock.now) == 50

    def test_invalidate_specific(self) -> None:
        clock = DeterministicClock()
        est = _CacheLocalityEstimator(ttl_sec=300.0)
        est.record("a", "prefix_X", cached_tokens=90, input_tokens=100, now=clock.now)
        est.record("a", "prefix_Y", cached_tokens=80, input_tokens=100, now=clock.now)
        est.invalidate("a", "prefix_X")
        assert est.estimate("a", "prefix_X", 100, clock.now) == 0
        assert est.estimate("a", "prefix_Y", 100, clock.now) == 80

    def test_invalidate_provider(self) -> None:
        clock = DeterministicClock()
        est = _CacheLocalityEstimator(ttl_sec=300.0)
        est.record("a", "prefix_X", cached_tokens=90, input_tokens=100, now=clock.now)
        est.record("a", "prefix_Y", cached_tokens=80, input_tokens=100, now=clock.now)
        est.record("b", "prefix_X", cached_tokens=70, input_tokens=100, now=clock.now)
        est.invalidate_provider("a")
        assert est.estimate("a", "prefix_X", 100, clock.now) == 0
        assert est.estimate("a", "prefix_Y", 100, clock.now) == 0
        assert est.estimate("b", "prefix_X", 100, clock.now) == 70

    def test_bounded_capacity(self) -> None:
        clock = DeterministicClock()
        est = _CacheLocalityEstimator(ttl_sec=300.0)
        # Use reflection to set max_entries for testing
        est._max_entries = 10
        for i in range(100):
            est.record("a", f"prefix_{i}", cached_tokens=50, input_tokens=100, now=clock.now)
        assert est.evidence_count <= 10

    def test_zero_cached_tokens_not_recorded(self) -> None:
        clock = DeterministicClock()
        est = _CacheLocalityEstimator(ttl_sec=300.0)
        est.record("a", "prefix_X", cached_tokens=0, input_tokens=100, now=clock.now)
        assert est.estimate("a", "prefix_X", 100, clock.now) == 0
        assert est.evidence_count == 0

    def test_thread_safety(self) -> None:
        clock = DeterministicClock()
        est = _CacheLocalityEstimator(ttl_sec=300.0)
        est._max_entries = 1000
        errors: list[Exception] = []

        def record_many(provider: str) -> None:
            try:
                for i in range(100):
                    est.record(
                        provider, f"prefix_{i}", cached_tokens=i, input_tokens=100, now=clock.now
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many, args=(f"p{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


# ---------------------------------------------------------------------------
# Tests: Miss confidence decay
# ---------------------------------------------------------------------------


class TestMissConfidenceDecay:
    """Observed misses degrade confidence in existing evidence."""

    def test_hit_then_miss_reduces_confidence(self) -> None:
        """A hit followed by a miss should reduce confidence."""
        clock = DeterministicClock()
        est = _CacheLocalityEstimator(ttl_sec=300.0)

        # Record a hit
        est.record("A", "prefix_X", cached_tokens=90, input_tokens=100, now=clock.now)
        est_after_hit = est.estimate("A", "prefix_X", 100, clock.now)
        assert est_after_hit == 90

        # Record a miss
        est.record("A", "prefix_X", cached_tokens=0, input_tokens=100, now=clock.now)
        est_after_miss = est.estimate("A", "prefix_X", 100, clock.now)
        # Confidence should have decayed: 90 * 0.3 = 27
        assert est_after_miss < est_after_hit, (
            f"Miss should reduce estimate: {est_after_miss} !< {est_after_hit}"
        )

    def test_repeated_misses_reduce_preference(self) -> None:
        """Repeated misses should eventually eliminate the preference."""
        clock = DeterministicClock()
        est = _CacheLocalityEstimator(ttl_sec=300.0)

        # Record a hit
        est.record("A", "prefix_X", cached_tokens=100, input_tokens=100, now=clock.now)

        # Repeated misses
        for _ in range(10):
            est.record("A", "prefix_X", cached_tokens=0, input_tokens=100, now=clock.now)

        est_after = est.estimate("A", "prefix_X", 100, clock.now)
        # After many misses, estimate should be very small or zero
        assert est_after < 10, f"Repeated misses should nearly eliminate estimate, got {est_after}"

    def test_miss_then_hit_recovers_evidence(self) -> None:
        """A miss followed by a hit should restore confidence."""
        clock = DeterministicClock()
        est = _CacheLocalityEstimator(ttl_sec=300.0)

        # Hit
        est.record("A", "prefix_X", cached_tokens=100, input_tokens=100, now=clock.now)
        # Miss
        est.record("A", "prefix_X", cached_tokens=0, input_tokens=100, now=clock.now)
        est_after_miss = est.estimate("A", "prefix_X", 100, clock.now)
        assert est_after_miss < 100

        # Another hit
        est.record("A", "prefix_X", cached_tokens=100, input_tokens=100, now=clock.now)
        est_after_hit = est.estimate("A", "prefix_X", 100, clock.now)
        assert est_after_hit == 100, f"Hit should restore full estimate, got {est_after_hit}"

    def test_miss_without_prior_evidence_creates_nothing(self) -> None:
        """A miss without prior evidence should not create evidence."""
        clock = DeterministicClock()
        est = _CacheLocalityEstimator(ttl_sec=300.0)

        est.record("A", "prefix_X", cached_tokens=0, input_tokens=100, now=clock.now)
        assert est.evidence_count == 0
