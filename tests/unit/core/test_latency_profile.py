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
