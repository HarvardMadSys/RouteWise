"""Stress-test scenarios for the tiered joint router.

Goes beyond the S6/S7/S8 mechanism tests to exercise regimes that matter
for real deployment:

ST1  Multi-S_A choice
     Three S_A providers with different cost / latency profiles. Tests
     whether joint_ucb picks correctly among multiple eligible S_A
     providers and whether the exploration bonus unnecessarily diversifies.

ST2  Mid-run S_Q degradation
     S_Q starts fast (well within SLO) and abruptly slows at t=30 min.
     Tests how quickly the Bernoulli + CP UCB filter rejects S_Q once its
     miss rate climbs. Also compares against two_layer which has no
     mechanism for reacting.

ST3  Multi-day quota rollover
     Three-day simulation with the S_Q quota resetting every 24 h. Tests
     that the shadow-price schedule re-initializes cleanly at each window
     boundary and that joint_ucb uses fresh capacity promptly.
"""

from __future__ import annotations

import math

from ..providers import LogNormal
from .providers import (
    ConcurrencyState,
    ProviderTier,
    QuotaState,
    TieredProvider,
)
from .scenarios import TieredScenarioConfig


_TPS = LogNormal(mu=5.5, sigma=0.3)


def _ln_p50_sigma(p50_ms: float, sigma: float = 0.5) -> LogNormal:
    return LogNormal(mu=math.log(p50_ms), sigma=sigma)


def _ln_p50_p99(p50_ms: float, p99_ms: float) -> LogNormal:
    mu = math.log(p50_ms)
    sigma = (math.log(p99_ms) - mu) / 2.326
    return LogNormal(mu=mu, sigma=max(sigma, 0.01))


def make_stress_scenarios() -> dict[str, TieredScenarioConfig]:
    """Build ST1, ST2, ST3."""
    return {
        # ------------------------------------------------------------- ST1
        "st1_multi_s_a": TieredScenarioConfig(
            name="st1_multi_s_a",
            description=(
                "3 S_A providers + 1 S_Q. Cost vs latency tradeoff within S_A "
                "tier. Tests whether joint_ucb picks the right S_A under "
                "cross-tier constraints."
            ),
            providers=[
                TieredProvider(
                    name="Chutes_S_Q",
                    cost_per_token=0.0,
                    ttft_dist=_ln_p50_sigma(400),
                    tps_dist=_TPS,
                    tier=ProviderTier.S_Q,
                    quota=QuotaState(size=200, window_sec=86400.0),
                ),
                TieredProvider(
                    name="S_A_fast",
                    cost_per_token=5.0e-6,    # expensive
                    ttft_dist=_ln_p50_sigma(100),
                    tps_dist=_TPS,
                    tier=ProviderTier.S_A,
                ),
                TieredProvider(
                    name="S_A_medium",
                    cost_per_token=2.0e-6,
                    ttft_dist=_ln_p50_sigma(300),
                    tps_dist=_TPS,
                    tier=ProviderTier.S_A,
                ),
                TieredProvider(
                    name="S_A_cheap",
                    cost_per_token=0.5e-6,    # cheapest but slowest
                    ttft_dist=_ln_p50_sigma(800),
                    tps_dist=_TPS,
                    tier=ProviderTier.S_A,
                ),
            ],
            n_requests=500,
            duration_seconds=3600.0,
            primary_slo_ms=2000.0,
            slo_thresholds_ms=[1000.0, 2000.0, 3000.0, 5000.0],
        ),

        # ------------------------------------------------------------- ST2
        "st2_s_q_degradation": TieredScenarioConfig(
            name="st2_s_q_degradation",
            description=(
                "S_Q starts fast (P50=200ms) then suddenly degrades to "
                "P50=2000ms at t=1800s. Tests how quickly joint_ucb's filter "
                "rejects S_Q once its Bernoulli miss rate climbs."
            ),
            providers=[
                TieredProvider(
                    name="Degrading_S_Q",
                    cost_per_token=0.0,
                    ttft_dist=_ln_p50_sigma(200),
                    tps_dist=_TPS,
                    tier=ProviderTier.S_Q,
                    quota=QuotaState(size=500, window_sec=86400.0),
                    shift_time=1800.0,
                    ttft_dist_after=_ln_p50_p99(2000, 6000),  # P95 ≈ 3800
                ),
                TieredProvider(
                    name="Stable_S_A",
                    cost_per_token=3.0e-6,
                    ttft_dist=_ln_p50_sigma(150),
                    tps_dist=_TPS,
                    tier=ProviderTier.S_A,
                ),
            ],
            n_requests=1000,
            duration_seconds=3600.0,
            primary_slo_ms=1500.0,
            slo_thresholds_ms=[500.0, 1000.0, 1500.0, 3000.0],
        ),

        # ------------------------------------------------------------- ST3
        "st3_multi_day_rollover": TieredScenarioConfig(
            name="st3_multi_day_rollover",
            description=(
                "3-day simulation with S_Q quota=100 resetting every 24 h. "
                "Tests that psi(z) re-initializes at each window boundary "
                "and joint_ucb uses fresh capacity without hysteresis."
            ),
            providers=[
                TieredProvider(
                    name="Daily_S_Q",
                    cost_per_token=0.0,
                    ttft_dist=_ln_p50_sigma(300),
                    tps_dist=_TPS,
                    tier=ProviderTier.S_Q,
                    quota=QuotaState(size=100, window_sec=86400.0),  # 1 day
                ),
                TieredProvider(
                    name="Always_S_A",
                    cost_per_token=3.0e-6,
                    ttft_dist=_ln_p50_sigma(200),
                    tps_dist=_TPS,
                    tier=ProviderTier.S_A,
                ),
            ],
            # 3 days total; 500 req/day -> quota exhausts early each day,
            # then spillover to S_A until next reset.
            n_requests=1500,
            duration_seconds=3 * 86400.0,  # 3 days
            primary_slo_ms=2000.0,
            slo_thresholds_ms=[1000.0, 2000.0, 3000.0],
        ),
    }
