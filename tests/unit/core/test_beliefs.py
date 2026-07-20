"""Tests for the unified latency-beliefs layer."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from llm_routewise.core.beliefs import LatencyBeliefs


@dataclass
class FakeView:
    """Minimal ProviderView stub for belief queries."""

    name: str
    tier: str = "api"
    prior_mean: float | None = None
    prior_cdf: float | None = None

    def is_available(self, now: float) -> bool:
        return True

    def quota_fraction_used(self, now: float) -> float | None:
        return None

    def route_cost_usd(self, now: float) -> float:
        return 0.0

    def hedge_cost_usd(self, now: float) -> float:
        return 0.0

    def prior_ttft_mean_ms(self, now: float) -> float | None:
        return self.prior_mean

    def prior_ttft_cdf(self, value_ms: float, now: float) -> float | None:
        return self.prior_cdf


def test_observed_mean_uses_window_samples() -> None:
    beliefs = LatencyBeliefs(window_sec=100.0)
    beliefs.observe("p", 0.0, 100.0)
    beliefs.observe("p", 10.0, 300.0)
    view = FakeView("p", prior_mean=999.0)
    assert beliefs.mean_ms(view, 20.0) == pytest.approx(200.0)
    assert beliefs.mean_fallback_rate() == 0.0
    assert beliefs.fallback_rate() == 0.0


def test_observed_mean_falls_back_to_prior_then_penalty() -> None:
    beliefs = LatencyBeliefs(window_sec=100.0, unprofiled_penalty_ms=1e9)
    with_prior = FakeView("oracle", prior_mean=42.0)
    without_prior = FakeView("cold")
    assert beliefs.mean_ms(with_prior, 0.0) == pytest.approx(42.0)
    assert beliefs.mean_ms(without_prior, 0.0) == pytest.approx(1e9)
    # Both queries fell back past the rolling window.
    assert beliefs.mean_fallback_rate() == 1.0
    assert beliefs.fallback_rate() == 1.0


def test_observed_mean_applies_error_penalty() -> None:
    beliefs = LatencyBeliefs(window_sec=100.0, error_penalty_ms=60_000.0)
    beliefs.observe("p", 0.0, 100.0)
    beliefs.observe_error("p", 5.0, "rate_limit")
    view = FakeView("p")
    assert beliefs.mean_ms(view, 10.0) == pytest.approx((100.0 + 60_000.0) / 2)


def test_observed_cdf_counts_errors_as_misses_and_prior_fallback() -> None:
    beliefs = LatencyBeliefs(window_sec=100.0)
    beliefs.observe("p", 0.0, 100.0)
    beliefs.observe_error("p", 5.0, "timeout")
    view = FakeView("p", prior_cdf=0.7)
    assert beliefs.cdf(view, 200.0, 10.0) == pytest.approx(0.5)
    cold = FakeView("cold", prior_cdf=0.7)
    assert beliefs.cdf(cold, 200.0, 10.0) == pytest.approx(0.7)
    no_prior = FakeView("none")
    assert beliefs.cdf(no_prior, 200.0, 10.0) == 0.0
    assert beliefs.cdf_fallback_rate() == pytest.approx(2.0 / 3.0)
    assert beliefs.mean_fallback_rate() == 0.0


def test_prior_only_mode_ignores_observations() -> None:
    beliefs = LatencyBeliefs(window_sec=100.0, mode="prior_only")
    beliefs.observe("p", 0.0, 100.0)
    beliefs.ensure("p")
    assert beliefs.profiles == {}
    view = FakeView("p", prior_mean=42.0, prior_cdf=0.9)
    assert beliefs.mean_ms(view, 10.0) == pytest.approx(42.0)
    assert beliefs.cdf(view, 100.0, 10.0) == pytest.approx(0.9)
    # prior_only queries are oracle reads, not profile queries.
    assert beliefs.mean_fallback_rate() == 0.0
    assert beliefs.cdf_fallback_rate() == 0.0
    assert beliefs.fallback_rate() == 0.0


def test_fallback_rate_mixed_queries() -> None:
    beliefs = LatencyBeliefs(window_sec=100.0)
    beliefs.observe("warm", 0.0, 100.0)
    warm = FakeView("warm")
    cold = FakeView("cold", prior_mean=10.0)
    beliefs.mean_ms(warm, 10.0)
    beliefs.mean_ms(cold, 10.0)
    assert beliefs.mean_fallback_rate() == pytest.approx(0.5)
    assert beliefs.fallback_rate() == pytest.approx(0.5)


def test_shared_profile_store_is_used() -> None:
    from llm_routewise.core.latency_profile import RollingLatencyProfile

    shared = RollingLatencyProfile(window_sec=100.0)
    beliefs = LatencyBeliefs(window_sec=100.0, profiles={"p": shared})
    beliefs.observe("p", 0.0, 123.0)
    assert shared.mean(1.0) == pytest.approx(123.0)
