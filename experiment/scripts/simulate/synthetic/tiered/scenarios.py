"""Tiered scenario definitions (S6-S8) for joint vs two-layer comparison.

S6 - Slow-but-free trap
    Tests whether two-layer routing greedily fills a slow S_Q quota while
    joint routing correctly rejects it based on the P50 band.

S7 - Quota depletion transition
    Tests whether the exponential shadow price produces a smooth handoff
    from S_Q to S_A as quota approaches exhaustion, vs the cliff behavior
    of two-layer.

S8 - Concurrency saturation spillover
    Tests whether the congestion price triggers spill to S_A before an
    S_C bottleneck inflates the tail, vs two-layer's blocking queue.
"""

from __future__ import annotations

import math

from .._core.scenarios import ScenarioConfig as TieredScenarioConfig
from ..providers import LogNormal
from .providers import (
    ConcurrencyState,
    ProviderTier,
    QuotaState,
    TieredProvider,
)

# ---------------------------------------------------------------------------
# LogNormal helpers (mirrors of the base scenarios module)
# ---------------------------------------------------------------------------


_TPS = LogNormal(mu=5.5, sigma=0.3)


def _ln_p50_sigma(p50_ms: float, sigma: float = 0.5) -> LogNormal:
    return LogNormal(mu=math.log(p50_ms), sigma=sigma)


def _ln_p50_p99(p50_ms: float, p99_ms: float) -> LogNormal:
    mu = math.log(p50_ms)
    sigma = (math.log(p99_ms) - mu) / 2.326
    return LogNormal(mu=mu, sigma=max(sigma, 0.01))


# ---------------------------------------------------------------------------
# Scenario factory
# ---------------------------------------------------------------------------


def make_tiered_scenarios() -> dict[str, TieredScenarioConfig]:
    """Create S6, S7, S8 scenarios for joint vs two-layer comparison."""
    return {
        # -------------------------------------------------------------------
        # S6: Slow-but-free trap.
        # S_Q is free but much slower than S_A. Two-layer greedy-by-tier
        # routes everything to S_Q and violates the SLO; joint_v2 sees S_Q
        # fail the P50 band check and routes to S_A.
        # -------------------------------------------------------------------
        "s6_slow_q_trap": TieredScenarioConfig(
            name="s6_slow_q_trap",
            description=(
                "S_Q is free but slow (P50=2000 ms). "
                "S_A is fast (P50=100 ms) at $3/M. "
                "Two-layer greedy-by-tier routes all traffic to S_Q and "
                "blows the SLO; joint_v2 rejects S_Q via the P50 band."
            ),
            providers=[
                TieredProvider(
                    name="Chutes_S_Q",
                    cost_per_token=0.0,
                    ttft_dist=_ln_p50_p99(2000, 5000),
                    tps_dist=_TPS,
                    tier=ProviderTier.S_Q,
                    quota=QuotaState(size=1000, window_sec=86400.0),
                ),
                TieredProvider(
                    name="Together_S_A",
                    cost_per_token=3.0e-6,
                    ttft_dist=_ln_p50_p99(100, 400),
                    tps_dist=_TPS,
                    tier=ProviderTier.S_A,
                ),
            ],
            n_requests=500,
            duration_seconds=86400.0,
            primary_slo_ms=1000.0,
            slo_thresholds_ms=[500.0, 1000.0, 2000.0, 5000.0],
        ),

        # -------------------------------------------------------------------
        # S7: Quota depletion transition.
        # Workload sized so S_Q runs out at ~50 % of the simulation. Both
        # providers are SLO-safe, so the question is purely about cost /
        # smooth handoff. Two-layer flips cliff-wise at z=1; joint_v2
        # ramps smoothly as psi(z) -> U.
        # -------------------------------------------------------------------
        "s7_quota_depletion": TieredScenarioConfig(
            name="s7_quota_depletion",
            description=(
                "S_Q (quota=100, P50=300 ms, $0) and S_A (P50=200 ms, $3/M). "
                "200-request workload causes quota to deplete at ~50 % of "
                "the run. Two-layer flips cliff-wise at z=1; joint_v2 ramps "
                "smoothly as psi(z) climbs."
            ),
            providers=[
                TieredProvider(
                    name="Chutes_S_Q",
                    cost_per_token=0.0,
                    ttft_dist=_ln_p50_sigma(300),
                    tps_dist=_TPS,
                    tier=ProviderTier.S_Q,
                    quota=QuotaState(size=100, window_sec=86400.0),
                ),
                TieredProvider(
                    name="Together_S_A",
                    cost_per_token=3.0e-6,
                    ttft_dist=_ln_p50_sigma(200),
                    tps_dist=_TPS,
                    tier=ProviderTier.S_A,
                ),
            ],
            n_requests=200,
            duration_seconds=3600.0,
            primary_slo_ms=2000.0,
        ),

        # -------------------------------------------------------------------
        # S8: Concurrency saturation spillover.
        # S_C has 4 slots and 2 s service time; arrival rate is 3 req/s
        # (3x capacity). Two-layer admits until saturated and queues or
        # retries; joint_v2 sees lambda(u) -> U and spills to S_A.
        # -------------------------------------------------------------------
        "s8_concurrency_saturation": TieredScenarioConfig(
            name="s8_concurrency_saturation",
            description=(
                "S_C (C=4 slots, 2s service, $0) and S_A (P50=100 ms, $3/M). "
                "Arrival rate 3 req/s is 3x the S_C capacity. "
                "Two-layer saturates S_C and queues; joint_v2 spills via lambda(u)."
            ),
            providers=[
                TieredProvider(
                    name="Featherless_S_C",
                    cost_per_token=0.0,
                    ttft_dist=_ln_p50_sigma(500),
                    tps_dist=_TPS,
                    tier=ProviderTier.S_C,
                    concurrency=ConcurrencyState(limit=4),
                    service_time_dist=_ln_p50_sigma(2000),  # 2 s service
                ),
                TieredProvider(
                    name="Together_S_A",
                    cost_per_token=3.0e-6,
                    ttft_dist=_ln_p50_sigma(100),
                    tps_dist=_TPS,
                    tier=ProviderTier.S_A,
                ),
            ],
            n_requests=3000,
            duration_seconds=1000.0,  # arrival rate = 3 req/s
            primary_slo_ms=2000.0,
        ),
    }
