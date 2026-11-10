"""Focused tests for the internal cache-locality evidence primitive."""

from __future__ import annotations

import threading

from llm_routewise._cache_locality import _CacheLocalityEstimator


class DeterministicClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

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

    def test_miss_after_ttl_does_not_resurrect_expired_evidence(self) -> None:
        clock = DeterministicClock()
        est = _CacheLocalityEstimator(ttl_sec=300.0)
        est.record("a", "prefix_X", cached_tokens=90, input_tokens=100, now=clock.now)

        clock.advance(300.001)
        est.record("a", "prefix_X", cached_tokens=0, input_tokens=100, now=clock.now)

        assert est.estimate("a", "prefix_X", 100, clock.now) == 0
        assert est.evidence_count == 0

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

    def test_estimate_is_not_negative_for_negative_current_input(self) -> None:
        clock = DeterministicClock()
        est = _CacheLocalityEstimator(ttl_sec=300.0)
        est.record("a", "prefix_X", cached_tokens=100, input_tokens=100, now=clock.now)
        assert est.estimate("a", "prefix_X", -5, clock.now) == 0

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

    def test_capacity_churn_evicts_oldest_in_constant_time_order(self) -> None:
        """High-cardinality churn keeps recent evidence and drops the oldest."""
        clock = DeterministicClock()
        est = _CacheLocalityEstimator(ttl_sec=300.0)
        est._max_entries = 3

        for prefix in ("oldest", "middle", "newest"):
            est.record("a", prefix, cached_tokens=50, input_tokens=100, now=clock.now)

        # An updated observation refreshes this key's eviction position.
        est.record("a", "oldest", cached_tokens=60, input_tokens=100, now=clock.now)
        est.record("a", "incoming", cached_tokens=50, input_tokens=100, now=clock.now)

        assert est.evidence_count == 3
        assert est.estimate("a", "oldest", 100, clock.now) == 60
        assert est.estimate("a", "middle", 100, clock.now) == 0
        assert est.estimate("a", "newest", 100, clock.now) == 50
        assert est.estimate("a", "incoming", 100, clock.now) == 50

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
        assert est.evidence_count == 990
        assert est.estimate("p0", "prefix_1", 100, clock.now) == 1
        assert est.estimate("p9", "prefix_99", 100, clock.now) == 99
        assert est.estimate("p0", "prefix_0", 100, clock.now) == 0


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

    def test_hit_decays_by_half_at_half_life(self) -> None:
        """A single hit retains half its estimate after one half-life."""
        est = _CacheLocalityEstimator(ttl_sec=300.0)
        est.record("A", "prefix_X", cached_tokens=100, input_tokens=100, now=0.0)

        assert est.estimate("A", "prefix_X", 100, 150.0) == 50

    def test_miss_near_ttl_does_not_extend_positive_evidence(self) -> None:
        """A miss cannot keep a hit alive past its original evidence TTL."""
        est = _CacheLocalityEstimator(ttl_sec=300.0)
        est.record("A", "prefix_X", cached_tokens=100, input_tokens=100, now=0.0)
        est.record("A", "prefix_X", cached_tokens=0, input_tokens=100, now=299.0)

        assert est.estimate("A", "prefix_X", 100, 301.0) == 0
        assert est.evidence_count == 0

    def test_miss_never_increases_estimate_at_a_later_time(self) -> None:
        """A miss penalty cannot make a later estimate exceed the no-miss case."""
        without_miss = _CacheLocalityEstimator(ttl_sec=300.0)
        with_miss = _CacheLocalityEstimator(ttl_sec=300.0)
        for est in (without_miss, with_miss):
            est.record("A", "prefix_X", cached_tokens=100, input_tokens=100, now=0.0)
        with_miss.record("A", "prefix_X", cached_tokens=0, input_tokens=100, now=100.0)

        assert with_miss.estimate("A", "prefix_X", 100, 150.0) <= without_miss.estimate(
            "A", "prefix_X", 100, 150.0
        )

    def test_repeated_misses_delete_dead_evidence_without_estimate(self) -> None:
        """Recording misses prunes sub-threshold evidence without a lookup."""
        est = _CacheLocalityEstimator(ttl_sec=300.0)
        est.record("A", "prefix_X", cached_tokens=100, input_tokens=100, now=0.0)

        for timestamp in range(1, 5):
            est.record("A", "prefix_X", cached_tokens=0, input_tokens=100, now=float(timestamp))

        assert est.evidence_count == 0

    def test_dead_miss_evidence_cannot_evict_a_more_recent_hit(self) -> None:
        """Misses do not refresh LRU priority and dead entries are removed."""
        est = _CacheLocalityEstimator(ttl_sec=300.0)
        est._max_entries = 2
        est.record("A", "old", cached_tokens=50, input_tokens=100, now=0.0)
        est.record("A", "recent", cached_tokens=60, input_tokens=100, now=1.0)

        for timestamp in range(2, 6):
            est.record("A", "old", cached_tokens=0, input_tokens=100, now=float(timestamp))
        est.record("A", "incoming", cached_tokens=70, input_tokens=100, now=6.0)

        assert est.evidence_count == 2
        assert est.estimate("A", "recent", 100, 6.0) == 58
        assert est.estimate("A", "incoming", 100, 6.0) == 70

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
