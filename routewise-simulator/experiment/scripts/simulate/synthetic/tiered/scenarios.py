"""Tiered scenario definitions for the synthetic simulator.

The headline scenario is `unified_pool`: a single provider pool shared
across all three tiers (S_Q, S_C, S_A) with five latency slots per tier.
Every tier reuses the same five (P50, P99) profiles so the method
comparison isolates tier-constraint handling from per-provider latency
differences.

S6/S7/S8/S7m/S8m remain as mechanism-isolation sanity checks exercising
single failure modes (slow-S_Q trap, quota depletion, concurrency
saturation, multi-S_Q quota hierarchy) and are retained for regression
testing and appendix material.
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


def _s_q_provider(
    *,
    name: str,
    p50_ms: float,
    p99_ms: float,
    quota_size: int,
) -> TieredProvider:
    return TieredProvider(
        name=name,
        cost_per_token=0.0,
        ttft_dist=_ln_p50_p99(p50_ms, p99_ms),
        tps_dist=_TPS,
        tier=ProviderTier.S_Q,
        quota=QuotaState(size=quota_size, window_sec=86400.0),
    )


def _s_c_provider(
    *,
    name: str,
    p50_ms: float,
    p99_ms: float,
    limit: int,
    service_ms: float,
) -> TieredProvider:
    return TieredProvider(
        name=name,
        cost_per_token=0.0,
        ttft_dist=_ln_p50_p99(p50_ms, p99_ms),
        tps_dist=_TPS,
        tier=ProviderTier.S_C,
        concurrency=ConcurrencyState(limit=limit),
        service_time_dist=_ln_p50_sigma(service_ms),
    )


def _s_a_provider(
    *,
    name: str,
    p50_ms: float,
    p99_ms: float,
    cost_per_token: float,
) -> TieredProvider:
    return TieredProvider(
        name=name,
        cost_per_token=cost_per_token,
        ttft_dist=_ln_p50_p99(p50_ms, p99_ms),
        tps_dist=_TPS,
        tier=ProviderTier.S_A,
    )


def make_tiered_scenarios() -> dict[str, TieredScenarioConfig]:
    """Create S6-S9 and J1-J3 scenarios for the LP study."""
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

        # -------------------------------------------------------------------
        # S9: Quota-vs-concurrency priority conflict.
        # This is a control-baseline disambiguation scenario. Both free tiers
        # exist simultaneously: S_Q is slower but abundant, S_C is faster but
        # capacity-limited, and S_A is the paid fallback. Quota-first should
        # greedily stay on S_Q, while concurrency-first should use S_C first.
        # -------------------------------------------------------------------
        "s9_quota_concurrency_priority": TieredScenarioConfig(
            name="s9_quota_concurrency_priority",
            description=(
                "S_Q (quota=800, P50=450 ms, $0), S_C (C=4 slots, "
                "P50=140 ms, 2s service, $0), and S_A (P50=100 ms, $3/M). "
                "Quota-first stays on S_Q while concurrency-first prefers S_C. "
                "This scenario is intended to separate fixed tier-priority "
                "baselines in a mixed-tier world."
            ),
            providers=[
                TieredProvider(
                    name="Ollama_S_Q",
                    cost_per_token=0.0,
                    ttft_dist=_ln_p50_p99(450, 1200),
                    tps_dist=_TPS,
                    tier=ProviderTier.S_Q,
                    quota=QuotaState(size=800, window_sec=86400.0),
                ),
                TieredProvider(
                    name="Featherless_S_C",
                    cost_per_token=0.0,
                    ttft_dist=_ln_p50_p99(140, 600),
                    tps_dist=_TPS,
                    tier=ProviderTier.S_C,
                    concurrency=ConcurrencyState(limit=4),
                    service_time_dist=_ln_p50_sigma(2000),
                ),
                TieredProvider(
                    name="Together_S_A",
                    cost_per_token=3.0e-6,
                    ttft_dist=_ln_p50_p99(100, 400),
                    tps_dist=_TPS,
                    tier=ProviderTier.S_A,
                ),
            ],
            n_requests=1800,
            duration_seconds=600.0,  # arrival rate = 3 req/s
            primary_slo_ms=1000.0,
        ),

        # -------------------------------------------------------------------
        # unified_pool: headline scenario for the NSDI submission.
        #
        # Five shared latency slots instantiated across all three tiers, so a
        # S_Q slot-k, S_C slot-k, and S_A slot-k have identical TTFT
        # distributions. Method-to-method differences therefore isolate
        # tier-constraint handling (quota shadow price, concurrency shadow
        # price, pay-per-token cost) from per-provider latency variability.
        #
        # Slots:
        #   0: fast-normal    (P50=200,  P99=500)
        #   1: tail-heavy     (P50=250,  P99=2500)   -- wide sigma; hedging bait
        #   2: normal         (P50=700,  P99=1800)
        #   3: trap           (P50=1000, P99=2600)   -- mid-run degradation on S_A
        #   4: slow-deep      (P50=1500, P99=4000)   -- mid-run improvement on S_A
        #
        # Cost (S_A only, breaks the "fast=expensive" monotone):
        #   slot 0: $2/M (normal)
        #   slot 1: $2/M (normal)
        #   slot 2: $1/M (dominant -- reproduces real WandB/Alibaba obs.)
        #   slot 3: $3/M (trap -- slow AND expensive)
        #   slot 4: $0.5/M (cheapest, slowest)
        #
        # Quota / Concurrency: monotone with latency (fast = scarce):
        #   slot 0: S_Q quota=200,   S_C C=1
        #   slot 1: S_Q quota=400,   S_C C=2
        #   slot 2: S_Q quota=800,   S_C C=3
        #   slot 3: S_Q quota=1200,  S_C C=4
        #   slot 4: S_Q quota=2000,  S_C C=5
        #
        # Drift (S_A only, mirrors paper fig:latency_drift 5x drift claim):
        #   slot 3 degrades at 50% duration  (P50 1000 -> 1800, P99 2600 -> 5000)
        #   slot 4 improves at 30% duration  (P50 1500 -> 900,  P99 4000 -> 2400)
        # S_Q / S_C are kept static (subscriptions are usually more stable).
        # -------------------------------------------------------------------
        "unified_pool": _make_unified_pool_scenario(
            duration_seconds=1500.0,
            n_requests=5000,
            primary_slo_ms=2000.0,
            slo_thresholds_ms=[500.0, 1000.0, 2000.0, 3000.0, 5000.0],
        ),
    }


# ---------------------------------------------------------------------------
# Unified pool builder
# ---------------------------------------------------------------------------


# Five shared latency profiles used across all three tiers.
_UNIFIED_SLOTS: tuple[tuple[str, float, float], ...] = (
    ("s0_fastnorm", 200.0, 500.0),
    ("s1_tailhvy", 250.0, 2500.0),
    ("s2_normal", 700.0, 1800.0),
    ("s3_trap", 1000.0, 2600.0),
    ("s4_slowdeep", 1500.0, 4000.0),
)

_UNIFIED_SQ_QUOTA: tuple[int, ...] = (200, 400, 800, 1200, 2000)
_UNIFIED_SC_LIMIT: tuple[int, ...] = (1, 2, 3, 4, 5)
_UNIFIED_SC_SERVICE_MS: float = 2000.0

# S_A cost (USD per token) per slot. Intentionally breaks the monotone
# "faster is more expensive" rule to reproduce real observations on
# OpenRouter (e.g. WandB / Alibaba are both fast and cheap on Qwen3-235B).
_UNIFIED_SA_COST: tuple[float, ...] = (
    2.0e-6,  # slot 0: normal
    2.0e-6,  # slot 1: normal
    1.0e-6,  # slot 2: dominant (fast AND cheap)
    3.0e-6,  # slot 3: trap (slow AND expensive)
    0.5e-6,  # slot 4: cheapest but slowest
)


def _make_unified_pool_scenario(
    *,
    duration_seconds: float,
    n_requests: int,
    primary_slo_ms: float,
    slo_thresholds_ms: list[float],
) -> TieredScenarioConfig:
    """Build the unified pool scenario shared by all three tiers."""
    providers: list[TieredProvider] = []

    # S_Q: free but quota-capped; fast slots get scarce quota, slow slots get
    # deep quota. No drift on subscriptions.
    for idx, (slot_tag, p50, p99) in enumerate(_UNIFIED_SLOTS):
        providers.append(
            _s_q_provider(
                name=f"SQ_{slot_tag}",
                p50_ms=p50,
                p99_ms=p99,
                quota_size=_UNIFIED_SQ_QUOTA[idx],
            )
        )

    # S_C: free but concurrency-capped; fast slots have tight slot counts.
    for idx, (slot_tag, p50, p99) in enumerate(_UNIFIED_SLOTS):
        providers.append(
            _s_c_provider(
                name=f"SC_{slot_tag}",
                p50_ms=p50,
                p99_ms=p99,
                limit=_UNIFIED_SC_LIMIT[idx],
                service_ms=_UNIFIED_SC_SERVICE_MS,
            )
        )

    # S_A: pay-per-token with non-monotone cost and drift on slots 3 and 4.
    shift_degrade_sec = 0.5 * duration_seconds
    shift_improve_sec = 0.3 * duration_seconds
    for idx, (slot_tag, p50, p99) in enumerate(_UNIFIED_SLOTS):
        provider = _s_a_provider(
            name=f"SA_{slot_tag}",
            p50_ms=p50,
            p99_ms=p99,
            cost_per_token=_UNIFIED_SA_COST[idx],
        )
        if idx == 3:
            provider.shift_time = shift_degrade_sec
            provider.ttft_dist_after = _ln_p50_p99(1800.0, 5000.0)
        elif idx == 4:
            provider.shift_time = shift_improve_sec
            provider.ttft_dist_after = _ln_p50_p99(900.0, 2400.0)
        providers.append(provider)

    return TieredScenarioConfig(
        name="unified_pool",
        description=(
            "Unified 5-slot pool shared across S_Q, S_C, and S_A. Each tier "
            "holds five providers with the same five TTFT profiles (fast-"
            "normal, tail-heavy, normal, trap, slow-deep). S_A cost breaks "
            "the fast-expensive monotone to encode dominant providers and a "
            "trap; two S_A providers drift mid-run to exercise the latency "
            "router under non-stationary profiles."
        ),
        providers=providers,
        n_requests=n_requests,
        duration_seconds=duration_seconds,
        primary_slo_ms=primary_slo_ms,
        slo_thresholds_ms=slo_thresholds_ms,
    )
