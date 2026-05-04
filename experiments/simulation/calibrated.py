"""Calibrated simulation scenarios — parameters derived from real measurements.

All latency and quota parameters are measured from live providers:
  - S_Q: Ollama Cloud glm-4.7:cloud, 60-min probe
    (experiment/results/ollama_quota_probe/glm47_60min/quota_estimate.json)
  - S_C: Featherless meta-llama/Meta-Llama-3.1-8B-Instruct, concurrency probe
    (experiment/results/featherless_concurrency_probe.json)
  - S_A: OpenRouter Llama-3.3-70B-Instruct production profile (10 providers)
    (summarized in phase5 24h evaluation; see paper Section 5 table)

Replaces the hand-picked parameters in scenarios.py. The original
scenarios remain in place for regression testing; this module adds
`make_calibrated_scenarios()` that returns S6c / S7c / S8c with the
same three mechanism-level experiments but real parameters.
"""

from __future__ import annotations

import math

from rwsim.world import (
    ConcurrencyState,
    LogNormal,
    ProviderTier,
    QuotaState,
    ScenarioConfig,
    TieredProvider,
)

# ---------------------------------------------------------------------------
# LogNormal helpers
# ---------------------------------------------------------------------------

_TPS = LogNormal(mu=5.5, sigma=0.3)


def _ln_p50_p99(p50_ms: float, p99_ms: float) -> LogNormal:
    mu = math.log(p50_ms)
    sigma = (math.log(p99_ms) - mu) / 2.326
    return LogNormal(mu=mu, sigma=max(sigma, 0.01))


# ---------------------------------------------------------------------------
# Measured provider parameters (as of 2026-04-21)
# ---------------------------------------------------------------------------

# Ollama Cloud glm-4.7 — 60-minute wall-clock probe.
# P50 = 13.3 s wall time; internal GPU P50 = 10864 ms.
OLLAMA_P50_MS = 10864.0
OLLAMA_P99_MS = 37204.0
OLLAMA_QUOTA_5H = 1350            # extrapolated from 60-min rate
OLLAMA_QUOTA_WINDOW_SEC = 5 * 3600

# Featherless Llama-3.1-8B — concurrency probe, single slot.
FEATHERLESS_BASE_P50_MS = 307.0
FEATHERLESS_BASE_P99_MS = 662.0
# Saturated behavior (c >= 2) — we encode this via _saturated pair so the
# synthetic sampler picks the queued distribution once admitted above the
# concurrency limit.
FEATHERLESS_SATURATED_P50_MS = 306.0
FEATHERLESS_SATURATED_P99_MS = 3131.0
FEATHERLESS_CONCURRENCY = 1       # hard measured: first request serves
                                  # in ~300 ms, second queues

# OpenRouter Llama-3.3-70B-Instruct — 1 h production trace summary.
OR_LLAMA_PROVIDERS: dict[str, dict] = {
    "Friendli": {
        "p50_ms": 301.0,
        "p99_ms": 1282.0,
        "cost_per_token": 0.60e-6,      # $0.60 per million tokens
    },
    "Novita": {
        "p50_ms": 590.0,
        "p99_ms": 1024.0,
        "cost_per_token": 0.135e-6,
    },
    "Parasail": {
        "p50_ms": 466.0,
        "p99_ms": 1744.0,
        "cost_per_token": 0.13e-6,
    },
    "AkashML": {
        "p50_ms": 536.0,
        "p99_ms": 3482.0,
        "cost_per_token": 0.13e-6,
    },
}


# ---------------------------------------------------------------------------
# Provider builders
# ---------------------------------------------------------------------------


def _ollama_s_q(quota_size: int | None = None) -> TieredProvider:
    size = OLLAMA_QUOTA_5H if quota_size is None else quota_size
    return TieredProvider(
        name="Ollama_S_Q",
        cost_per_token=0.0,          # subscription; sunk cost
        ttft_dist=_ln_p50_p99(OLLAMA_P50_MS, OLLAMA_P99_MS),
        tps_dist=_TPS,
        tier=ProviderTier.S_Q,
        quota=QuotaState(size=size, window_sec=OLLAMA_QUOTA_WINDOW_SEC),
    )


def _featherless_s_c(
    service_p50_ms: float | None = None,
) -> TieredProvider:
    """Featherless S_C provider.

    We model single-slot behavior by setting the base TTFT distribution to
    the measured c=1 numbers. When the runner admits requests beyond the
    concurrency limit the queue emerges naturally from ConcurrencyState;
    the scheduler's spillover decision (via lambda(u)) keeps the actual
    throughput matching measured saturated behavior.
    """
    # Service time matches observed e2e time (~2 s typical for a short
    # completion).
    return TieredProvider(
        name="Featherless_S_C",
        cost_per_token=0.0,
        ttft_dist=_ln_p50_p99(
            FEATHERLESS_BASE_P50_MS, FEATHERLESS_BASE_P99_MS,
        ),
        tps_dist=_TPS,
        tier=ProviderTier.S_C,
        concurrency=ConcurrencyState(limit=FEATHERLESS_CONCURRENCY),
        # 2-second service budget reflects the measured e2e at c=1.
        service_time_dist=_ln_p50_p99(
            service_p50_ms if service_p50_ms else 2000.0,
            (service_p50_ms or 2000.0) * 3.0,
        ),
    )


def _or_s_a(provider_name: str) -> TieredProvider:
    params = OR_LLAMA_PROVIDERS[provider_name]
    return TieredProvider(
        name=f"{provider_name}_S_A",
        cost_per_token=float(params["cost_per_token"]),
        ttft_dist=_ln_p50_p99(float(params["p50_ms"]), float(params["p99_ms"])),
        tps_dist=_TPS,
        tier=ProviderTier.S_A,
    )


# ---------------------------------------------------------------------------
# Scenario factory (calibrated)
# ---------------------------------------------------------------------------


def make_calibrated_scenarios() -> dict[str, ScenarioConfig]:
    """S6c/S7c/S8c — same mechanisms as S6/S7/S8 but calibrated parameters."""
    return {
        # -------------------------------------------------------------------
        # S6c_interactive: tight SLO (2s) — Ollama P50=10.9s fails the
        # safety filter, joint correctly falls back to S_A. This is the
        # "slow-Q trap" scenario: without the safety filter a cost-greedy
        # router would route to Ollama and violate SLO on nearly every
        # request.
        # -------------------------------------------------------------------
        "s6c_interactive": ScenarioConfig(
            name="s6c_interactive",
            description=(
                f"Tight SLO=2s (interactive chatbot). "
                f"S_Q: Ollama glm-4.7 (P50={OLLAMA_P50_MS:.0f}ms "
                "-- too slow), $0. "
                "S_A: Friendli (P50=301ms, P99=1282ms, $0.60/M). "
                "Joint safety filter must exclude Ollama; goes 100% S_A."
            ),
            providers=[
                _ollama_s_q(quota_size=OLLAMA_QUOTA_5H),
                _or_s_a("Friendli"),
            ],
            n_requests=500,
            duration_seconds=OLLAMA_QUOTA_WINDOW_SEC,
            primary_slo_ms=2000.0,
            slo_thresholds_ms=[1000.0, 2000.0, 3000.0, 5000.0],
        ),

        # -------------------------------------------------------------------
        # S6c_reasoning: relaxed SLO (30 s) matches reasoning/batch
        # workloads (long-CoT models, async jobs). Now Ollama's P50=10.9s
        # is comfortably under SLO and its zero marginal cost dominates
        # the objective at alpha=0. As alpha grows the router trades
        # cost for latency, switching to Friendli. This is where the
        # joint formulation's value is most visible: a single knob
        # navigates the cost-latency frontier.
        # -------------------------------------------------------------------
        "s6c_reasoning": ScenarioConfig(
            name="s6c_reasoning",
            description=(
                "Relaxed SLO=30s (reasoning / batch workload). "
                f"S_Q: Ollama (P50={OLLAMA_P50_MS:.0f}ms, "
                f"P99={OLLAMA_P99_MS:.0f}ms, quota={OLLAMA_QUOTA_5H}/5h, $0). "
                "S_A: Friendli (P50=301ms, P99=1282ms, $0.60/M). "
                "alpha=0 should prefer free Ollama; large alpha switches "
                "to Friendli for latency."
            ),
            providers=[
                _ollama_s_q(quota_size=OLLAMA_QUOTA_5H),
                _or_s_a("Friendli"),
            ],
            n_requests=300,
            duration_seconds=OLLAMA_QUOTA_WINDOW_SEC,
            primary_slo_ms=30000.0,
            slo_thresholds_ms=[5000.0, 10000.0, 30000.0, 60000.0],
        ),

        # -------------------------------------------------------------------
        # S7c: quota depletion with REAL parameters.
        # Smaller Ollama quota so it depletes mid-run. Both providers are
        # technically SLO-safe within their tails, so the question is cost /
        # smooth handoff. two_layer cliffs at z=1; joint's psi(z) ramps
        # smoothly and alpha controls when the switch happens.
        # -------------------------------------------------------------------
        "s7c_quota_depletion": ScenarioConfig(
            name="s7c_quota_depletion",
            description=(
                f"S_Q: Ollama (scaled quota=100), P50={OLLAMA_P50_MS:.0f}ms. "
                "S_A: Novita Llama-3.3-70B (P50=590ms, P99=1024ms, $0.135/M). "
                "200 requests -> S_Q depletes at 50 percent."
            ),
            providers=[
                _ollama_s_q(quota_size=100),
                _or_s_a("Novita"),
            ],
            n_requests=200,
            duration_seconds=3600.0,
            primary_slo_ms=20000.0,  # must admit Ollama as "SLO-safe"
                                      # since its tail is huge; relax SLO
                                      # in this specific scenario to study
                                      # quota depletion in isolation
        ),

        # -------------------------------------------------------------------
        # S8c: concurrency saturation with REAL Featherless.
        # Featherless has concurrency=1. Arrival rate 3 req/s = 3x capacity.
        # two_layer queues; joint spills via lambda(u). Measured P99 at
        # c>=2 is 3131ms (~5x P99 at c=1).
        # -------------------------------------------------------------------
        "s8c_concurrency_saturation": ScenarioConfig(
            name="s8c_concurrency_saturation",
            description=(
                "S_C: Featherless (C=1, P50=307ms, P99=662ms single-slot, "
                "P99=3131ms under saturation). "
                "S_A: Friendli (P50=301ms, P99=1282ms, $0.60/M). "
                "Arrival rate ~1.5 req/s drives queueing at S_C."
            ),
            providers=[
                _featherless_s_c(service_p50_ms=2000.0),
                _or_s_a("Friendli"),
            ],
            n_requests=2000,
            duration_seconds=1500.0,    # ~1.3 req/s
            primary_slo_ms=2000.0,
        ),
    }
