"""Tests for the rolling latency profile's error-aware extensions."""

from __future__ import annotations

import pytest

from rwsim.core.latency_profile import RollingLatencyProfile


def _profile_with_samples(
    samples: list[tuple[float, float]],
    window_sec: float = 100.0,
) -> RollingLatencyProfile:
    profile = RollingLatencyProfile(window_sec=window_sec)
    for ts, value in samples:
        profile.add_sample(ts, value)
    return profile


def test_count_matches_window_membership() -> None:
    profile = _profile_with_samples([(0.0, 10.0), (50.0, 20.0), (120.0, 30.0)])
    assert profile.count(120.0) == 2  # ts=0 expired (window 100)
    assert profile.count(200.0) == 1


def test_count_backwards_query() -> None:
    profile = _profile_with_samples([(0.0, 10.0), (50.0, 20.0), (120.0, 30.0)])
    assert profile.count(130.0) == 2
    # Behind the advanced clock: window [ -40, 60 ] holds two samples.
    assert profile.count(60.0) == 2
    assert profile.mean(60.0) == pytest.approx(15.0)


def test_cdf_backwards_query_values() -> None:
    """Pin the stripe-corrected values of plain cdf() behind the clock.

    The queries are chosen so both stripes matter: the drop stripe removes
    the newest in-window sample and the restore stripe brings back one the
    persistent window already expired. Simulator hedging feeds on these
    values via LatencyBeliefs.cdf, so they must be exact, not just crash-free.
    """
    profile = _profile_with_samples([(0.0, 10.0), (50.0, 20.0), (120.0, 30.0)])
    # Forward query advances the cdf clock to 130; window [30, 130] = {20, 30}.
    assert profile.cdf(25.0, 130.0) == pytest.approx(0.5)
    # Behind the clock at now=60: window [-40, 60] = {10, 20}.
    assert profile.cdf(15.0, 60.0) == pytest.approx(0.5)
    assert profile.cdf(25.0, 60.0) == pytest.approx(1.0)
    # Further behind at now=40: window [-60, 40] = {10}.
    assert profile.cdf(15.0, 40.0) == pytest.approx(1.0)
    assert profile.cdf(5.0, 40.0) == pytest.approx(0.0)


def test_error_count_is_windowed() -> None:
    profile = RollingLatencyProfile(window_sec=100.0)
    profile.add_error(10.0, "rate_limit")
    profile.add_error(90.0, "timeout")
    assert profile.error_count(95.0) == 2
    assert profile.error_count(150.0) == 1
    assert profile.error_count(300.0) == 0


def test_mean_with_errors_equals_mean_when_no_errors() -> None:
    profile = _profile_with_samples([(0.0, 10.0), (10.0, 30.0)])
    assert profile.mean_with_errors(20.0, error_penalty_ms=60_000.0) == profile.mean(20.0)


def test_mean_with_errors_folds_in_penalty() -> None:
    profile = _profile_with_samples([(0.0, 100.0), (10.0, 200.0)])
    profile.add_error(15.0, "rate_limit")
    expected = (100.0 + 200.0 + 60_000.0) / 3
    assert profile.mean_with_errors(20.0, error_penalty_ms=60_000.0) == pytest.approx(expected)


def test_mean_with_errors_only_errors_returns_penalty() -> None:
    profile = RollingLatencyProfile(window_sec=100.0)
    profile.add_error(5.0, "timeout")
    assert profile.mean_with_errors(10.0, error_penalty_ms=60_000.0) == pytest.approx(60_000.0)


def test_mean_with_errors_empty_returns_none() -> None:
    profile = RollingLatencyProfile(window_sec=100.0)
    assert profile.mean_with_errors(10.0, error_penalty_ms=60_000.0) is None


def test_cdf_counting_errors_equals_cdf_when_no_errors() -> None:
    profile = _profile_with_samples([(0.0, 10.0), (10.0, 30.0), (20.0, 50.0)])
    assert profile.cdf_counting_errors(30.0, 25.0) == profile.cdf(30.0, 25.0)


def test_cdf_counting_errors_treats_errors_as_misses() -> None:
    profile = _profile_with_samples([(0.0, 10.0), (10.0, 30.0)])
    profile.add_error(15.0, "rate_limit")
    # Two successes under threshold out of three total attempts.
    assert profile.cdf_counting_errors(40.0, 20.0) == pytest.approx(2.0 / 3.0)
    assert profile.cdf_counting_errors(5.0, 20.0) == pytest.approx(0.0)


def test_cdf_counting_errors_backwards_query() -> None:
    profile = _profile_with_samples([(0.0, 10.0), (50.0, 20.0), (120.0, 30.0)])
    profile.add_error(55.0, "timeout")
    # Advance the cdf clock, then query behind it.
    assert profile.cdf(25.0, 130.0) is not None
    # Window at now=60: samples at ts=0,50 plus the error at ts=55.
    assert profile.cdf_counting_errors(15.0, 60.0) == pytest.approx(1.0 / 3.0)


def test_mean_with_errors_backwards_query() -> None:
    profile = _profile_with_samples([(0.0, 100.0), (50.0, 200.0), (120.0, 300.0)])
    profile.add_error(55.0, "rate_limit")
    assert profile.mean(130.0) is not None  # advance mean clock
    expected = (100.0 + 200.0 + 60_000.0) / 3
    assert profile.mean_with_errors(60.0, error_penalty_ms=60_000.0) == pytest.approx(expected)


def test_error_free_paths_unchanged_by_error_machinery() -> None:
    plain = _profile_with_samples([(0.0, 10.0), (10.0, 30.0), (20.0, 50.0)])
    assert plain.mean(25.0) == pytest.approx(30.0)
    assert plain.cdf(30.0, 25.0) == pytest.approx(2.0 / 3.0)
    assert plain.error_count(25.0) == 0


# Retention bounds for the long-feed tests: two windows of samples at one
# sample per second, plus the index's prune-batch slack.
_WINDOW_SAMPLES = 100
_UNBATCHED_BOUND = 2 * _WINDOW_SAMPLES + 50
_BATCHED_BOUND = 2 * _WINDOW_SAMPLES + 1024 + 50


def test_long_feed_memory_bounded_with_live_query_clock() -> None:
    """Router-pattern feed: interleaved queries keep every structure bounded."""
    profile = RollingLatencyProfile(window_sec=100.0)
    for i in range(5000):
        profile.add_sample(float(i), 50.0)
        if i % 10 == 0:
            profile.mean(float(i))
            profile.cdf(60.0, float(i))
    assert len(profile.samples) <= _UNBATCHED_BOUND
    assert len(profile._mean_pending) <= _UNBATCHED_BOUND
    assert len(profile._samples_by_ts) <= _BATCHED_BOUND


def test_long_feed_memory_bounded_before_any_query() -> None:
    """Feed-only pattern: retention anchors on the newest sample instead."""
    profile = RollingLatencyProfile(window_sec=100.0)
    for i in range(5000):
        profile.add_sample(float(i), 50.0)
    assert len(profile.samples) <= _UNBATCHED_BOUND
    assert len(profile._mean_pending) <= _UNBATCHED_BOUND
    assert len(profile._samples_by_ts) <= _BATCHED_BOUND
    # A first query near the freshest observation is unaffected by pruning.
    assert profile.mean(5000.0) == pytest.approx(50.0)
    assert profile.count(5000.0) == 100


def test_stale_cdf_clock_does_not_pin_retention() -> None:
    """One early cdf query, then a mean-only feed (hedge-policy pattern).

    A cdf clock initialized once and never advanced again must not pin the
    retention cutoff at its own stale position, and the cdf pending heap —
    which only drains on cdf queries — must be pruned like the rest.
    """
    profile = RollingLatencyProfile(window_sec=100.0)
    profile.add_sample(0.0, 50.0)
    profile.cdf(60.0, 0.0)
    for i in range(1, 5000):
        profile.add_sample(float(i), 50.0)
        if i % 10 == 0:
            profile.mean(float(i))
    # Stale-clamped horizon: four windows behind the freshest activity.
    stale_unbatched = 2 * _UNBATCHED_BOUND
    assert len(profile.samples) <= stale_unbatched
    assert len(profile._cdf_pending) <= stale_unbatched
    assert len(profile._mean_pending) <= stale_unbatched
    assert len(profile._samples_by_ts) <= stale_unbatched + 1024
    # A cdf query finally arriving near the present is exact: the pruned
    # pending entries would all have expired from its window anyway.
    assert profile.cdf(60.0, 5000.0) == pytest.approx(1.0)
    assert profile.cdf(40.0, 5000.0) == pytest.approx(0.0)


def test_backwards_query_around_stalled_clock_degrades_to_none() -> None:
    """Stripes reaching below the retention cutoff must not silently answer."""
    profile = RollingLatencyProfile(window_sec=100.0)
    for i in range(300):
        profile.add_sample(float(i), 50.0)
    profile.cdf(60.0, 200.0)
    for i in range(300, 5000):
        profile.add_sample(float(i), 50.0)
        profile.mean(float(i))
    # The stalled cdf clock's backwards envelope reaches pruned data: fall
    # back to the defensive scan (empty here), never a stripe answer missing
    # its restore samples.
    assert profile.cdf(60.0, 150.0) is None
    # The live mean clock keeps its full backwards envelope.
    assert profile.mean(4920.0) == pytest.approx(50.0)


def test_backwards_queries_stay_exact_after_long_feed_pruning() -> None:
    fed = [(float(i), float(i % 7)) for i in range(3000)]
    profile = RollingLatencyProfile(window_sec=100.0)
    for ts, value in fed:
        profile.add_sample(ts, value)
        if int(ts) % 5 == 0:
            profile.mean(ts)
            profile.cdf(3.0, ts)
    # One window behind the clock (2995), well past thousands of pruned samples.
    now = 2920.0
    window = [value for ts, value in fed if now - 100.0 <= ts <= now]
    assert profile.mean(now) == pytest.approx(sum(window) / len(window))
    assert profile.cdf(3.0, now) == pytest.approx(
        sum(1 for value in window if value <= 3.0) / len(window)
    )
