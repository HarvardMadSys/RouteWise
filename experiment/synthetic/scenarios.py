"""Synthetic experiment scenarios.

Each scenario is a carefully designed (providers, workload, slo) triple
that tests a specific failure mode or property of Two-layer vs Joint
architectures. The scenarios are listed in `SCENARIOS` with a short
description of what they test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from experiment.synthetic.provider import (
    SyntheticProvider,
    TIER_S_A,
    TIER_S_C,
    TIER_S_Q,
)
from experiment.synthetic.workload import (
    SyntheticRequest,
    generate_bimodal_workload,
    generate_workload,
)


@dataclass
class Scenario:
    """A synthetic test scenario."""

    name: str
    description: str
    providers: list[SyntheticProvider]
    requests: list[SyntheticRequest]
    slo_ms: float


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _ln_from_target_p50(target_ms: float, sigma: float) -> float:
    """Return log-normal mu such that median = target_ms."""
    return math.log(target_ms)


def _ln_from_p50_p99(p50_ms: float, p99_ms: float) -> tuple[float, float]:
    """Return (mu, sigma) log-normal params from desired P50 and P99."""
    mu = math.log(p50_ms)
    # P99: mu + 2.326 * sigma = log(p99) => sigma = (log(p99) - mu) / 2.326
    sigma = max(0.05, (math.log(p99_ms) - mu) / 2.326)
    return mu, sigma


# -----------------------------------------------------------------------------
# Scenario 1: Benign - all tiers have similar latency, quota abundant
# Tests: Joint should not regress vs Two-layer when Two-layer is already optimal.
# -----------------------------------------------------------------------------


def make_scenario_1_benign(seed: int = 0) -> Scenario:
    mu_q, sig_q = _ln_from_p50_p99(200, 500)
    mu_a, sig_a = _ln_from_p50_p99(200, 500)

    providers = [
        SyntheticProvider(
            name="S_Q_fast",
            tier=TIER_S_Q,
            daily_quota=10_000,
            ttft_mu=mu_q,
            ttft_sigma=sig_q,
        ),
        SyntheticProvider(
            name="S_A_together",
            tier=TIER_S_A,
            price_per_m_output=3.0,
            ttft_mu=mu_a,
            ttft_sigma=sig_a,
        ),
        SyntheticProvider(
            name="S_A_fireworks",
            tier=TIER_S_A,
            price_per_m_output=4.0,
            ttft_mu=mu_a,
            ttft_sigma=sig_a,
        ),
    ]

    requests = generate_workload(
        n_requests=2000,
        duration_sec=86400,  # 1 day
        seed=seed,
    )

    return Scenario(
        name="S1_benign",
        description=(
            "All tiers similar latency; S_Q quota abundant. Two-layer should "
            "happily use S_Q; Joint should do the same. Tests no regression."
        ),
        providers=providers,
        requests=requests,
        slo_ms=2000,
    )


# -----------------------------------------------------------------------------
# Scenario 2: Slow-but-free trap - S_Q much slower than S_A
# Tests: Two-layer gets stuck on slow S_Q, Joint correctly escapes.
# -----------------------------------------------------------------------------


def make_scenario_2_slow_subscription(seed: int = 0) -> Scenario:
    mu_q, sig_q = _ln_from_p50_p99(2000, 5000)  # slow!
    mu_a, sig_a = _ln_from_p50_p99(150, 500)     # fast

    providers = [
        SyntheticProvider(
            name="S_Q_slow",
            tier=TIER_S_Q,
            daily_quota=10_000,
            ttft_mu=mu_q,
            ttft_sigma=sig_q,
        ),
        SyntheticProvider(
            name="S_A_fast",
            tier=TIER_S_A,
            price_per_m_output=3.0,
            ttft_mu=mu_a,
            ttft_sigma=sig_a,
        ),
    ]

    requests = generate_workload(n_requests=2000, duration_sec=86400, seed=seed)
    return Scenario(
        name="S2_slow_subscription",
        description=(
            "Slow S_Q (P50=2s) + fast S_A (P50=150ms). SLO=1s. Two-layer "
            "routes to S_Q and violates SLO massively; Joint should refuse "
            "S_Q because its P50 is way outside the band."
        ),
        providers=providers,
        requests=requests,
        slo_ms=1000,
    )


# -----------------------------------------------------------------------------
# Scenario 3: Quota depletion - S_Q quota << request count, both fast
# Tests: Joint smooth transition vs Two-layer cliff.
# -----------------------------------------------------------------------------


def make_scenario_3_quota_depletion(seed: int = 0) -> Scenario:
    """Bimodal workload with scarce S_Q quota.

    Tests value-density saving: the best policy is to use scarce S_Q quota
    for long requests (high API cost) and let short requests pay the API.
    Two-layer (greedy fill-first) is oblivious to value and wastes quota
    on whichever request arrives first.
    """
    mu_q, sig_q = _ln_from_p50_p99(200, 500)
    mu_a, sig_a = _ln_from_p50_p99(200, 500)

    providers = [
        SyntheticProvider(
            name="S_Q_limited",
            tier=TIER_S_Q,
            daily_quota=300,  # very scarce
            ttft_mu=mu_q,
            ttft_sigma=sig_q,
        ),
        SyntheticProvider(
            name="S_A_backup",
            tier=TIER_S_A,
            price_per_m_output=3.0,
            ttft_mu=mu_a,
            ttft_sigma=sig_a,
        ),
    ]

    # 1000 req/day, bimodal: 70% short (50 tok) + 30% long (2000 tok).
    # Quota 300 covers 30% of requests; right-sized to let quota hold all longs.
    requests = generate_bimodal_workload(
        n_requests=1000,
        duration_sec=86400,
        short_tokens=50,
        long_tokens=2000,
        long_fraction=0.3,
        seed=seed,
    )

    return Scenario(
        name="S3_quota_depletion",
        description=(
            "Scarce S_Q quota (300) vs 1000 bimodal requests (70% short, "
            "30% long). Two providers latency-equivalent. Tests value-"
            "density saving: Joint should reserve quota for long requests."
        ),
        providers=providers,
        requests=requests,
        slo_ms=2000,
    )


# -----------------------------------------------------------------------------
# Scenario 4: Concurrency saturation
# Tests: Two-layer queues on S_C, Joint spills to S_A.
# -----------------------------------------------------------------------------


def make_scenario_4_concurrency_saturation(seed: int = 0) -> Scenario:
    """Heavy S_C saturation: S_C fast but capacity-limited, S_A slower unlimited.

    Note: the simulator's availability check already gates S_C once u=1, so
    even two-layer naturally spills to S_A when S_C is full. The distinction
    in this scenario is that Joint (with congestion shadow price) starts
    spilling at u=0.8 (soft ramp), which reduces queue-induced latency.
    """
    mu_c, sig_c = _ln_from_p50_p99(200, 400)
    mu_a, sig_a = _ln_from_p50_p99(500, 1000)

    providers = [
        SyntheticProvider(
            name="S_C_limited",
            tier=TIER_S_C,
            concurrency_limit=2,
            ttft_mu=mu_c,
            ttft_sigma=sig_c,
            tps=500.0,  # slow throughput -> slot held long
        ),
        SyntheticProvider(
            name="S_A_unlimited",
            tier=TIER_S_A,
            price_per_m_output=3.0,
            ttft_mu=mu_a,
            ttft_sigma=sig_a,
        ),
    ]

    # 5000 req in 300s = ~16.7 req/s. S_C capacity ~ 5 req/s => 3x over.
    requests = generate_bimodal_workload(
        n_requests=5000,
        duration_sec=300,
        short_tokens=50,
        long_tokens=1000,
        long_fraction=0.3,
        seed=seed,
    )

    return Scenario(
        name="S4_concurrency_saturation",
        description=(
            "3x over-saturation of S_C (2 slots). Joint spills earlier via "
            "congestion shadow price; two-layer only spills when fully saturated."
        ),
        providers=providers,
        requests=requests,
        slo_ms=2000,
    )


# -----------------------------------------------------------------------------
# Scenario 5: Tail-heavy subscription
# Tests: Hedging variants win; non-hedge Joint should choose based on tail.
# -----------------------------------------------------------------------------


def make_scenario_5_tail_heavy(seed: int = 0) -> Scenario:
    # S_Q has great P50 but bad tail.
    mu_q, sig_q = _ln_from_p50_p99(150, 3000)  # big sigma -> heavy tail
    # S_A has worse P50 but stable tail.
    mu_a, sig_a = _ln_from_p50_p99(400, 600)   # small sigma -> light tail

    providers = [
        SyntheticProvider(
            name="S_Q_spiky",
            tier=TIER_S_Q,
            daily_quota=10_000,
            ttft_mu=mu_q,
            ttft_sigma=sig_q,
        ),
        SyntheticProvider(
            name="S_A_stable",
            tier=TIER_S_A,
            price_per_m_output=3.0,
            ttft_mu=mu_a,
            ttft_sigma=sig_a,
        ),
    ]

    requests = generate_workload(n_requests=2000, duration_sec=86400, seed=seed)
    return Scenario(
        name="S5_tail_heavy",
        description=(
            "S_Q with excellent P50 (150ms) but heavy tail (P99=3s); S_A "
            "with worse P50 but stable. Tests value of hedging: primary=S_Q "
            "for cost, backup=S_A to catch tail."
        ),
        providers=providers,
        requests=requests,
        slo_ms=1000,
    )


# -----------------------------------------------------------------------------
# Scenario 6: Multi-S_A mix + subscription (realistic)
# Tests: Full Pareto frontier comparison in a realistic multi-provider setup.
# -----------------------------------------------------------------------------


def make_scenario_6_realistic_mix(seed: int = 0) -> Scenario:
    providers = [
        # Subscription: cheap but moderate latency, limited
        SyntheticProvider(
            name="S_Q_chutes",
            tier=TIER_S_Q,
            daily_quota=1000,
            ttft_mu=math.log(400),
            ttft_sigma=0.7,  # moderate tail
        ),
        SyntheticProvider(
            name="S_C_featherless",
            tier=TIER_S_C,
            concurrency_limit=4,
            ttft_mu=math.log(350),
            ttft_sigma=0.5,
            tps=1500.0,
        ),
        # API: cost-latency tradeoffs
        SyntheticProvider(
            name="S_A_cheap_slow",
            tier=TIER_S_A,
            price_per_m_output=1.5,
            ttft_mu=math.log(500),
            ttft_sigma=0.5,
        ),
        SyntheticProvider(
            name="S_A_mid",
            tier=TIER_S_A,
            price_per_m_output=3.0,
            ttft_mu=math.log(300),
            ttft_sigma=0.4,
        ),
        SyntheticProvider(
            name="S_A_fast_expensive",
            tier=TIER_S_A,
            price_per_m_output=5.0,
            ttft_mu=math.log(150),
            ttft_sigma=0.3,
        ),
    ]

    # 2500 req / day - exceeds S_Q quota, stresses S_C
    requests = generate_workload(
        n_requests=2500,
        duration_sec=86400,
        seed=seed,
    )
    return Scenario(
        name="S6_realistic_mix",
        description=(
            "5 providers across 3 tiers with cost/latency tradeoffs. Workload "
            "exceeds S_Q quota and stresses S_C concurrency. The full 5x2 "
            "strategy matrix should reveal Pareto frontier differences."
        ),
        providers=providers,
        requests=requests,
        slo_ms=1000,
    )


# -----------------------------------------------------------------------------
# Scenario registry
# -----------------------------------------------------------------------------


SCENARIO_BUILDERS = [
    make_scenario_1_benign,
    make_scenario_2_slow_subscription,
    make_scenario_3_quota_depletion,
    make_scenario_4_concurrency_saturation,
    make_scenario_5_tail_heavy,
    make_scenario_6_realistic_mix,
]


def all_scenarios(seed: int = 0) -> list[Scenario]:
    return [fn(seed=seed) for fn in SCENARIO_BUILDERS]
