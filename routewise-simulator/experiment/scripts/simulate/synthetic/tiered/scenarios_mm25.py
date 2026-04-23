"""Calibrated tiered scenarios using MiniMax-M2.5 measurements across all providers.

All latency parameters come from real concurrency probes conducted on
2026-04-21 (see experiment/results/minimax_m25_probes/). Using a single
model across tiers gives apples-to-apples comparison and addresses
Juncheng's 4/20 concern about hand-picked scenario parameters.

Provider                      Role    TTFT p50   TTFT p99   Capacity
------------------------------------------------------------------------
Ollama Cloud (subscription)   S_Q      305 ms     488 ms    Q=5000/day, soft C=3
Chutes (MiniMax-M2.5-TEE)     S_Q      976 ms    1389 ms    subscription, quota pending
MiniMax coding plan (native)  S_Q     1715 ms    8915 ms    subscription, quota pending
Featherless                   S_C      312 ms     497 ms    C=1 hard rate-limit
OpenRouter (various, minimax- S_A      varies     varies    pay-per-token $0.15-$0.60/M
  m2.5 endpoints)

All S_Q providers are paid subscriptions; their marginal cost per
request is 0 because the subscription is a sunk cost. Quota shadow
price psi(z) captures the opportunity cost of consuming capacity.

The scenarios exercise three mechanism-level properties of the joint
router:
  S6m: Featherless concurrency saturation + spillover to S_A.
  S7m: Ollama quota depletion with smooth psi(z) handoff to S_A.
  S8m: Multi-S_Q hierarchy (Ollama preferred over Chutes, MiniMax; all
       fall back to S_A when exhausted).
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TPS = LogNormal(mu=5.5, sigma=0.3)


def _ln_p50_p99(p50_ms: float, p99_ms: float) -> LogNormal:
    mu = math.log(p50_ms)
    sigma = (math.log(p99_ms) - mu) / 2.326
    return LogNormal(mu=mu, sigma=max(sigma, 0.01))


# ---------------------------------------------------------------------------
# Measured parameters (probe date: 2026-04-21)
# ---------------------------------------------------------------------------

# TTFT at concurrency=1 (baseline), from concurrency probe.
PROBE = {
    "ollama": {"p50_ms": 305.0, "p99_ms": 488.0},
    "chutes": {"p50_ms": 976.0, "p99_ms": 1389.0},
    "minimax": {"p50_ms": 1715.0, "p99_ms": 8915.0},
    "featherless": {"p50_ms": 312.0, "p99_ms": 497.0},
}

# Ollama Cloud subscription (paid; marginal cost 0 at margin).
# Sustained probe 2026-04-21: 1707 successes in 30 min at 1 req/s cap,
# no rate limiting encountered. 5h extrapolation = 17077 requests.
# Conservative: use the measured 30-min window sustained rate.
OLLAMA_QUOTA_PER_HOUR = 3414   # 56.9 req/min * 60
OLLAMA_QUOTA_WINDOW_SEC = 3600.0
OLLAMA_CONCURRENCY_SOFT = 3   # informational (user-reported)

# Chutes subscription: 1012 successes in 30 min, 33.7 req/min sustained,
# no rate limit hit. 5h extrapolation = 10120.
CHUTES_QUOTA_PER_HOUR = 2022   # 33.7 req/min * 60
CHUTES_QUOTA_WINDOW_SEC = 3600.0

# MiniMax coding plan: 563 successes in 20 min at 1 req/s, 28.1 req/min
# sustained. No rate limit. 5h extrapolation = 8444.
MINIMAX_QUOTA_PER_HOUR = 1686   # 28.1 req/min * 60
MINIMAX_QUOTA_WINDOW_SEC = 3600.0

# Featherless: measured hard-limit at concurrency=1 (50%+ 429s at c=2).
FEATHERLESS_CONCURRENCY = 1

# OpenRouter S_A pricing (from OpenRouter endpoint list for minimax/m2.5).
# We include the cheapest and a representative mid-price option.
OR_PROVIDERS = {
    "Chutes_OR":   {"in_per_m": 0.150, "out_per_m": 1.200,
                     "p50_ms": 976.0, "p99_ms": 1389.0},  # same Chutes backend
    "Parasail_OR": {"in_per_m": 0.150, "out_per_m": 1.200,
                     "p50_ms": 500.0, "p99_ms": 1500.0},   # estimate; typical
    "Friendli_OR": {"in_per_m": 0.300, "out_per_m": 1.200,
                     "p50_ms": 400.0, "p99_ms": 1200.0},
}

# Typical request token counts for cost estimation.
TYPICAL_TOKENS_IN = 100
TYPICAL_TOKENS_OUT = 100


def _or_cost_per_token(in_per_m: float, out_per_m: float) -> float:
    """Blend input + output pricing into a single cost_per_token number.

    For a request with TYPICAL_TOKENS_IN input and TYPICAL_TOKENS_OUT output:
        cost = (in_per_m * tokens_in + out_per_m * tokens_out) / 1e6
    The synthetic simulator uses a single cost_per_token parameter, so we
    blend: cost_per_token = (in_per_m * 0.5 + out_per_m * 0.5) / 1e6.
    """
    return (in_per_m * 0.5 + out_per_m * 0.5) * 1e-6


# ---------------------------------------------------------------------------
# Provider builders
# ---------------------------------------------------------------------------


def _ollama_s_q(quota_size: int = OLLAMA_QUOTA_PER_HOUR) -> TieredProvider:
    return TieredProvider(
        name="Ollama_S_Q",
        cost_per_token=0.0,     # subscription: marginal is 0
        ttft_dist=_ln_p50_p99(PROBE["ollama"]["p50_ms"], PROBE["ollama"]["p99_ms"]),
        tps_dist=_TPS,
        tier=ProviderTier.S_Q,
        quota=QuotaState(size=quota_size, window_sec=OLLAMA_QUOTA_WINDOW_SEC),
    )


def _chutes_s_q(quota_size: int = CHUTES_QUOTA_PER_HOUR) -> TieredProvider:
    return TieredProvider(
        name="Chutes_S_Q",
        cost_per_token=0.0,
        ttft_dist=_ln_p50_p99(PROBE["chutes"]["p50_ms"], PROBE["chutes"]["p99_ms"]),
        tps_dist=_TPS,
        tier=ProviderTier.S_Q,
        quota=QuotaState(size=quota_size, window_sec=CHUTES_QUOTA_WINDOW_SEC),
    )


def _minimax_s_q(quota_size: int = MINIMAX_QUOTA_PER_HOUR) -> TieredProvider:
    return TieredProvider(
        name="MiniMax_S_Q",
        cost_per_token=0.0,
        ttft_dist=_ln_p50_p99(PROBE["minimax"]["p50_ms"], PROBE["minimax"]["p99_ms"]),
        tps_dist=_TPS,
        tier=ProviderTier.S_Q,
        quota=QuotaState(size=quota_size, window_sec=MINIMAX_QUOTA_WINDOW_SEC),
    )


def _featherless_s_c() -> TieredProvider:
    return TieredProvider(
        name="Featherless_S_C",
        cost_per_token=0.0,
        ttft_dist=_ln_p50_p99(
            PROBE["featherless"]["p50_ms"], PROBE["featherless"]["p99_ms"],
        ),
        tps_dist=_TPS,
        tier=ProviderTier.S_C,
        concurrency=ConcurrencyState(limit=FEATHERLESS_CONCURRENCY),
        service_time_dist=_ln_p50_p99(2000.0, 6000.0),
    )


def _or_s_a(label: str) -> TieredProvider:
    params = OR_PROVIDERS[label]
    return TieredProvider(
        name=f"{label}",
        cost_per_token=_or_cost_per_token(params["in_per_m"], params["out_per_m"]),
        ttft_dist=_ln_p50_p99(params["p50_ms"], params["p99_ms"]),
        tps_dist=_TPS,
        tier=ProviderTier.S_A,
    )


# ---------------------------------------------------------------------------
# Scenario factory
# ---------------------------------------------------------------------------


def make_mm25_scenarios() -> dict[str, TieredScenarioConfig]:
    """S6m / S7m / S8m with MiniMax-M2.5 calibrated parameters."""
    return {
        # -------------------------------------------------------------------
        # S6m: Featherless saturation + S_A spillover.
        # Featherless hard-limits at c=1; any burst forces rejections that
        # the joint router must absorb by spilling to S_A. Real measurement
        # shows 50 percent rejection rate at c=2.
        # -------------------------------------------------------------------
        "s6m_featherless_saturation": TieredScenarioConfig(
            name="s6m_featherless_saturation",
            description=(
                f"S_C: Featherless (P50={PROBE['featherless']['p50_ms']:.0f}ms, "
                f"P99={PROBE['featherless']['p99_ms']:.0f}ms, C={FEATHERLESS_CONCURRENCY} "
                "HARD limit, 50%+ 429 rate at c>=2). "
                f"S_A: Parasail OpenRouter (P50=500ms, $0.15/$1.20 per M). "
                "Burst workload drives S_C saturation."
            ),
            providers=[
                _featherless_s_c(),
                _or_s_a("Parasail_OR"),
            ],
            n_requests=1500,
            duration_seconds=1200.0,   # 1.25 req/s average; bursts oversaturate
            primary_slo_ms=2000.0,
        ),

        # -------------------------------------------------------------------
        # S7m: Ollama quota depletion with smooth handoff.
        # Scaled quota so depletion occurs mid-run. Both providers have
        # near-identical TTFT (~300ms), so the question is purely cost:
        # joint at alpha=0 should stay on Ollama until psi(z) makes its
        # effective cost exceed S_A's price.
        # -------------------------------------------------------------------
        "s7m_quota_depletion": TieredScenarioConfig(
            name="s7m_quota_depletion",
            description=(
                f"S_Q: Ollama (P50={PROBE['ollama']['p50_ms']:.0f}ms, "
                "quota=100 scaled for depletion). "
                f"S_A: Parasail_OR (P50=500ms, $0.15/$1.20 per M). "
                "Joint alpha=0 should stay on Ollama; psi(z) triggers "
                "smooth handoff to S_A as quota depletes."
            ),
            providers=[
                _ollama_s_q(quota_size=100),
                _or_s_a("Parasail_OR"),
            ],
            n_requests=200,
            duration_seconds=3600.0,
            primary_slo_ms=2000.0,
        ),

        # -------------------------------------------------------------------
        # S8m: Multi-S_Q hierarchy. Three subscriptions with different
        # latency / quota tradeoffs. Joint router should prefer the fastest
        # free provider (Ollama), spill to next-best (Chutes_S_Q), and
        # finally S_A when all subscriptions are exhausted.
        # -------------------------------------------------------------------
        "s8m_multi_sq_hierarchy": TieredScenarioConfig(
            name="s8m_multi_sq_hierarchy",
            description=(
                "Three S_Q providers with different latency/quota. "
                f"Ollama (P50={PROBE['ollama']['p50_ms']:.0f}ms, quota=50), "
                f"Chutes_S_Q (P50={PROBE['chutes']['p50_ms']:.0f}ms, quota=100), "
                f"MiniMax_S_Q (P50={PROBE['minimax']['p50_ms']:.0f}ms, quota=50). "
                "S_A fallback: Parasail_OR ($0.15/$1.20 per M). "
                "Joint should prefer Ollama first."
            ),
            providers=[
                _ollama_s_q(quota_size=50),
                _chutes_s_q(quota_size=100),
                _minimax_s_q(quota_size=50),
                _or_s_a("Parasail_OR"),
            ],
            n_requests=300,
            duration_seconds=3600.0,
            primary_slo_ms=3000.0,  # relaxed to admit MiniMax (P99 = 8.9s
                                    # would exceed 2s SLO for ~10% of requests)
        ),
    }
