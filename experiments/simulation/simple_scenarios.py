"""Minimal cost-latency scenarios for LP-budget sanity checks.

These scenarios intentionally avoid quota, concurrency, and shadow pricing so
the expected behavior of the range-budget LP can be reasoned about by hand.
They are meant to be run before the full simulation grid.
"""

from __future__ import annotations

import math

from rwsim.world import (
    LogNormal,
    ProviderTier,
    ScenarioConfig,
    TieredProvider,
)


SIMPLE_SCENARIOS = (
    "simple_same_cost_different_latency",
    "simple_slow_cheap_fast_expensive",
    "simple_heavy_tail_with_hedge",
)

_TPS = LogNormal(mu=5.5, sigma=0.1)
_N_REQUESTS = 2400
_DURATION_SECONDS = 600.0
_COST_UNIT = 1e-6


def _ln_p50_p99(p50_ms: float, p99_ms: float) -> LogNormal:
    mu = math.log(p50_ms)
    sigma = (math.log(p99_ms) - mu) / 2.326
    return LogNormal(mu=mu, sigma=max(sigma, 0.01))


def _api_provider(
    name: str,
    *,
    cost_units: float,
    p50_ms: float,
    p99_ms: float,
) -> TieredProvider:
    return TieredProvider(
        name=name,
        cost_per_token=cost_units * _COST_UNIT,
        ttft_dist=_ln_p50_p99(p50_ms, p99_ms),
        tps_dist=_TPS,
        tier=ProviderTier.S_A,
    )


def _scenario(
    name: str,
    description: str,
    providers: list[TieredProvider],
    *,
    primary_slo_ms: float = 1000.0,
) -> ScenarioConfig:
    return ScenarioConfig(
        name=name,
        description=description,
        providers=providers,
        n_requests=_N_REQUESTS,
        duration_seconds=_DURATION_SECONDS,
        arrival_process="poisson",
        primary_slo_ms=primary_slo_ms,
    )


def make_simple_scenarios() -> dict[str, ScenarioConfig]:
    """Return toy scenarios with hand-checkable LP-budget behavior."""
    same_cost = _scenario(
        "simple_same_cost_different_latency",
        (
            "All providers have identical API cost but different latency. "
            "Range-budget variants should all choose the fastest provider."
        ),
        [
            _api_provider("samecost_slow", cost_units=1.0, p50_ms=900.0, p99_ms=1200.0),
            _api_provider("samecost_medium", cost_units=1.0, p50_ms=500.0, p99_ms=700.0),
            _api_provider("samecost_fast", cost_units=1.0, p50_ms=200.0, p99_ms=300.0),
        ],
    )

    slow_cheap = _scenario(
        "simple_slow_cheap_fast_expensive",
        (
            "Latency and cost are intentionally opposed: slow is cheap, fast is "
            "expensive. Increasing p should move traffic toward faster providers."
        ),
        [
            _api_provider("cheap_slow", cost_units=0.25, p50_ms=900.0, p99_ms=1200.0),
            _api_provider("mid_balanced", cost_units=1.00, p50_ms=500.0, p99_ms=700.0),
            _api_provider("expensive_fast", cost_units=4.00, p50_ms=200.0, p99_ms=300.0),
        ],
    )

    heavy_tail = _scenario(
        "simple_heavy_tail_with_hedge",
        (
            "The cheap provider has good median latency but a bad tail. Hedging "
            "should reduce P99/SLO violations when that provider is selected."
        ),
        [
            _api_provider("cheap_heavytail", cost_units=0.25, p50_ms=250.0, p99_ms=3500.0),
            _api_provider("stable_mid", cost_units=1.00, p50_ms=450.0, p99_ms=700.0),
            _api_provider("expensive_stable_fast", cost_units=4.00, p50_ms=220.0, p99_ms=320.0),
        ],
    )

    return {
        scenario.name: scenario
        for scenario in (same_cost, slow_cheap, heavy_tail)
    }


__all__ = ["SIMPLE_SCENARIOS", "make_simple_scenarios"]
