"""Tiered scenario definitions (S6-S9, J1-J3) for joint comparison.

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

S9 - Quota-vs-concurrency priority conflict
    Tests whether a quota-first heuristic and a concurrency-first heuristic
    diverge when both free tiers coexist. The joint LP should route across
    S_Q, S_C, and S_A without committing to either fixed priority.

J1 - Fully-joint quota-biased market
    Multiple providers in every tier. Fast-but-shallow S_Q competes with
    deeper but slower S_Q and with capacity-limited S_C. A good LP should
    use all three tiers without collapsing to a fixed heuristic priority.

J2 - Fully-joint concurrency-stressed market
    Multiple S_C providers look attractive on body latency but saturate
    under load, forcing spillover to S_Q and S_A. This stresses whether
    the LP reacts smoothly before tail inflation appears.

J3 - Fully-joint rich convex-hull market
    A richer provider set where every tier contains non-dominated options.
    This is the closest synthetic approximation to the paper's general
    tiered marketplace claim.
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
        # J1: Fully-joint quota-biased market.
        # Every tier has multiple providers. A fast but shallow S_Q competes
        # with a slower deep-quota S_Q and two S_C providers with different
        # latency/capacity tradeoffs. API providers offer a priced fallback.
        # -------------------------------------------------------------------
        "j1_joint_quota_bias": TieredScenarioConfig(
            name="j1_joint_quota_bias",
            description=(
                "Fully-joint market with 2x S_Q, 2x S_C, 3x S_A. "
                "S_Q_fast is low-latency but quota-limited; S_Q_deep is slower "
                "but abundant. S_C has one fast-shallow and one slower-deeper "
                "option. S_A spans cheap / balanced / fast priced providers. "
                "This scenario tests whether the LP forms a meaningful joint "
                "frontier instead of committing to a single tier priority."
            ),
            providers=[
                _s_q_provider(
                    name="SQ_FastShallow",
                    p50_ms=240,
                    p99_ms=620,
                    quota_size=260,
                ),
                _s_q_provider(
                    name="SQ_DeepSlow",
                    p50_ms=520,
                    p99_ms=1350,
                    quota_size=900,
                ),
                _s_c_provider(
                    name="SC_FastTight",
                    p50_ms=160,
                    p99_ms=560,
                    limit=2,
                    service_ms=1700,
                ),
                _s_c_provider(
                    name="SC_Balanced",
                    p50_ms=250,
                    p99_ms=820,
                    limit=5,
                    service_ms=2100,
                ),
                _s_a_provider(
                    name="SA_Cheap",
                    p50_ms=340,
                    p99_ms=1200,
                    cost_per_token=1.6e-6,
                ),
                _s_a_provider(
                    name="SA_Balanced",
                    p50_ms=190,
                    p99_ms=760,
                    cost_per_token=2.6e-6,
                ),
                _s_a_provider(
                    name="SA_Fast",
                    p50_ms=115,
                    p99_ms=420,
                    cost_per_token=4.2e-6,
                ),
            ],
            n_requests=2200,
            duration_seconds=900.0,
            primary_slo_ms=1200.0,
            slo_thresholds_ms=[400.0, 800.0, 1200.0, 2000.0],
        ),

        # -------------------------------------------------------------------
        # J2: Fully-joint concurrency-stressed market.
        # The S_C tier looks attractive on P50 but is not deep enough for the
        # offered load. The LP should spill before the tail explodes and should
        # not get stuck on the slower deep-quota option or overuse pricey S_A.
        # -------------------------------------------------------------------
        "j2_joint_concurrency_stress": TieredScenarioConfig(
            name="j2_joint_concurrency_stress",
            description=(
                "Fully-joint market with 2x S_Q, 2x S_C, 3x S_A under stronger "
                "concurrency pressure. S_C providers are attractive on body "
                "latency but saturate under load, forcing careful spillover. "
                "This scenario stresses the body/tail split in a general "
                "many-provider-per-tier world."
            ),
            providers=[
                _s_q_provider(
                    name="SQ_Budget",
                    p50_ms=500,
                    p99_ms=1350,
                    quota_size=850,
                ),
                _s_q_provider(
                    name="SQ_SlowDeep",
                    p50_ms=860,
                    p99_ms=2280,
                    quota_size=1800,
                ),
                _s_c_provider(
                    name="SC_UltraFast",
                    p50_ms=120,
                    p99_ms=440,
                    limit=2,
                    service_ms=2600,
                ),
                _s_c_provider(
                    name="SC_Elastic",
                    p50_ms=250,
                    p99_ms=840,
                    limit=4,
                    service_ms=2900,
                ),
                _s_a_provider(
                    name="SA_Cheap",
                    p50_ms=380,
                    p99_ms=1420,
                    cost_per_token=9.0e-7,
                ),
                _s_a_provider(
                    name="SA_Balanced",
                    p50_ms=145,
                    p99_ms=500,
                    cost_per_token=1.7e-6,
                ),
                _s_a_provider(
                    name="SA_Fast",
                    p50_ms=105,
                    p99_ms=360,
                    cost_per_token=4.0e-6,
                ),
            ],
            n_requests=4800,
            duration_seconds=900.0,
            primary_slo_ms=1400.0,
            slo_thresholds_ms=[400.0, 800.0, 1400.0, 2200.0],
        ),

        # -------------------------------------------------------------------
        # J3: Fully-joint rich convex-hull market.
        # This is the richest synthetic world in the suite. All tiers contain
        # multiple non-dominated providers, so any strong result here is much
        # harder to dismiss as a two-point or single-tier artifact.
        # -------------------------------------------------------------------
        "j3_joint_rich_market": TieredScenarioConfig(
            name="j3_joint_rich_market",
            description=(
                "Rich fully-joint market with 3x S_Q, 2x S_C, 3x S_A. Every tier "
                "contains non-dominated options with different cost/latency/"
                "capacity tradeoffs. This scenario is intended as the paper's "
                "strongest synthetic generality check."
            ),
            providers=[
                _s_q_provider(
                    name="SQ_Fast",
                    p50_ms=260,
                    p99_ms=620,
                    quota_size=220,
                ),
                _s_q_provider(
                    name="SQ_Mid",
                    p50_ms=420,
                    p99_ms=980,
                    quota_size=700,
                ),
                _s_q_provider(
                    name="SQ_Deep",
                    p50_ms=700,
                    p99_ms=1750,
                    quota_size=1400,
                ),
                _s_c_provider(
                    name="SC_Fast",
                    p50_ms=135,
                    p99_ms=500,
                    limit=2,
                    service_ms=2100,
                ),
                _s_c_provider(
                    name="SC_Mid",
                    p50_ms=270,
                    p99_ms=860,
                    limit=5,
                    service_ms=2400,
                ),
                _s_a_provider(
                    name="SA_Cheap",
                    p50_ms=310,
                    p99_ms=980,
                    cost_per_token=1.0e-6,
                ),
                _s_a_provider(
                    name="SA_Mid",
                    p50_ms=160,
                    p99_ms=620,
                    cost_per_token=1.9e-6,
                ),
                _s_a_provider(
                    name="SA_Fast",
                    p50_ms=95,
                    p99_ms=320,
                    cost_per_token=4.2e-6,
                ),
            ],
            n_requests=3200,
            duration_seconds=1000.0,
            primary_slo_ms=1600.0,
            slo_thresholds_ms=[400.0, 900.0, 1600.0, 2400.0],
        ),
    }
