"""Tiered routing strategies.

Implements five strategies that operate on a TieredScenarioConfig:

two_layer
    Current RouteWise design: Layer 1 picks the tier by subscription priority
    (S_Q if quota available, else S_C if slot available, else S_A). Layer 2
    picks the lowest-P50 provider within that tier.

joint_nohedge
    Proposed joint design (corrected). Computes unified effective cost c_eff
    across all providers, filters by SLO-anchored P95 safety, then picks the
    cheapest by effective cost. Allows a slower-but-cheaper provider to win
    as long as its P95 is comfortably below the SLO.

joint_hedge
    joint_nohedge + cross-tier SMART_ECONOMIC hedging.

joint_p50band_nohedge  (ablation)
    Joint design using the *incorrect* P50-band filter (a P50 near-best
    band + cheapest-in-band). Shown alongside the corrected design to
    demonstrate that the filter choice matters for cross-tier correctness.

joint_p50band_hedge  (ablation)
    joint_p50band_nohedge + cross-tier SMART_ECONOMIC hedging.

Design lineage (see LESSONS_LEARNED.md):
    V1 (discarded): ported V2 router's P50-band selection directly to the
        cross-tier setting. The filter was too aggressive: in scenarios where
        a subscription tier has moderately higher P50 but is well within the
        SLO, the band filter excluded it and forced 100% routing to S_A.
    V2 (current): SLO-anchored P95 filter. Any provider whose P95 is within
        the SLO (scaled by a safety margin) enters the candidate set; the
        cheapest by effective cost wins.

Each strategy returns a StrategyRun with per-request results.
"""

from __future__ import annotations

import importlib.util as _ilu
import math
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Module loading (mirrors the pattern in ../runner.py)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_STRATEGIES_DIR = os.path.join(_PROJECT_ROOT, "experiment", "strategies")


def _load_mod(name: str, filename: str):
    path = os.path.join(_STRATEGIES_DIR, filename)
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_v2_mod = _load_mod("_tiered_v2", "v2_router.py")
_hedge_mod = _load_mod("_tiered_hedge", "smart_hedging.py")

V2Router = _v2_mod.V2Router
SmartHedger = _hedge_mod.SmartHedger
HedgingParams = _hedge_mod.HedgingParams
HedgingStrategy = _hedge_mod.HedgingStrategy
BackupSelectionMethod = _hedge_mod.BackupSelectionMethod

from rwsim.schemas import Request  # noqa: E402

from rwsim.world.metrics import StrategyRun  # noqa: E402

from rwsim.world.distributions import LogNormal  # noqa: E402
from rwsim.world.providers import ProviderTier, TieredProvider  # noqa: E402
from rwsim.world.scenarios import ScenarioConfig as TieredScenarioConfig  # noqa: E402
from rwsim.world.shadow_price import (  # noqa: E402
    calibrate_envelopes,
    concurrency_shadow_price,
    effective_cost,
    quota_shadow_price,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIERED_STRATEGIES = [
    "two_layer",
    "joint_nohedge",
    "joint_hedge",
    "joint_p50band_nohedge",
    "joint_p50band_hedge",
]

# Tokens used to convert shadow price (per request) into router "costs" dict.
_TYPICAL_TOKENS = 200

# V1 (ablation): P50 near-best band width.
_P50_BAND_MARGIN = 0.10

# V2 (current): P95 must be <= SLO * P95_SAFETY_MARGIN to admit a provider.
# 0.8 gives a 20 % headroom over the stated SLO.
_P95_SAFETY_MARGIN = 0.8

def _sample_ttft(
    provider: TieredProvider,
    rng: np.random.Generator,
    t: float,
) -> float:
    return float(provider.sample_ttft(rng, t))


def _sample_service_time(
    provider: TieredProvider,
    rng: np.random.Generator,
    t: float,
    output_tokens: int,
) -> float:
    """Service time in seconds (for S_C slot occupancy)."""
    if provider.tier != ProviderTier.S_C:
        return 0.0
    if provider.service_time_dist is not None:
        return float(provider.service_time_dist.sample(rng)[0]) / 1000.0
    ttft_ms, e2e_ms = provider.sample_request(output_tokens, rng, t)
    return e2e_ms / 1000.0


def _reset_all_state(providers: list[TieredProvider]) -> None:
    for p in providers:
        p.reset_state()


def _lognormal_p95(dist: LogNormal) -> float:
    """Analytical P95 of a LogNormal. Phi^{-1}(0.95) = 1.645."""
    return math.exp(dist.mu + 1.645 * dist.sigma)


def _provider_p95_at(provider: TieredProvider, now: float) -> float:
    """P95 TTFT for a provider at simulated time `now`.

    Supports both stationary SyntheticProvider and ShiftingProvider. Falls
    back to the base `ttft_dist` if no shift mechanism is present.
    """
    if hasattr(provider, "_active_dist"):
        dist = provider._active_dist(now)  # noqa: SLF001 — shift helper
    else:
        dist = provider.ttft_dist
    return _lognormal_p95(dist)


# ---------------------------------------------------------------------------
# Strategy: two_layer (current RouteWise design)
# ---------------------------------------------------------------------------


def _pick_tier(providers: list[TieredProvider], now: float) -> ProviderTier:
    """Layer-1 tier selection: subscription priority S_Q > S_C > S_A."""
    for tier in (ProviderTier.S_Q, ProviderTier.S_C, ProviderTier.S_A):
        if any(p.tier == tier and p.is_available(now) for p in providers):
            return tier
    return ProviderTier.S_A


def _run_two_layer(
    scenario: TieredScenarioConfig,
    requests: list[Request],
    rng: np.random.Generator,
) -> StrategyRun:
    """Tier-first routing with lowest-P50 within the chosen tier."""
    _reset_all_state(scenario.providers)

    ttft_ms: list[float] = []
    cost_usd: list[float] = []
    provider_sel: list[str] = []
    tier_sel: list[str] = []
    timestamps: list[float] = []
    z_series: list[float] = []
    u_series: list[float] = []

    for req in requests:
        t = float(req.timestamp)

        tier = _pick_tier(scenario.providers, t)
        tier_providers = [p for p in scenario.providers if p.tier == tier]
        if not tier_providers:
            tier_providers = [
                p for p in scenario.providers if p.tier == ProviderTier.S_A
            ]

        primary = min(tier_providers, key=lambda p: p.true_p50_ms(t))

        tm = _sample_ttft(primary, rng, t)
        service_time = _sample_service_time(primary, rng, t, req.response_tokens)
        primary.account_request(req.id, t, service_time)

        ttft_ms.append(tm)
        cost_usd.append(primary.marginal_cost(req.total_tokens, t))
        provider_sel.append(primary.name)
        tier_sel.append(primary.tier.value)
        timestamps.append(t)

        q_provider = next(
            (p for p in scenario.providers if p.tier == ProviderTier.S_Q), None
        )
        c_provider = next(
            (p for p in scenario.providers if p.tier == ProviderTier.S_C), None
        )
        z_series.append(q_provider.quota.fraction_used(t) if q_provider else 0.0)
        u_series.append(c_provider.concurrency.utilization(t) if c_provider else 0.0)

    return StrategyRun(
        strategy="two_layer",
        ttft_ms=np.array(ttft_ms),
        cost_usd=np.array(cost_usd),
        provider=provider_sel,
        tier=tier_sel,
        timestamp=np.array(timestamps),
        hedge_triggered=np.zeros(len(ttft_ms), dtype=bool),
        quota_fraction_used=np.array(z_series),
        concurrency_utilization=np.array(u_series),
    )


# ---------------------------------------------------------------------------
# Joint selectors: two filter rules to compare
# ---------------------------------------------------------------------------


def _joint_select_p50band(
    providers: list[TieredProvider],
    req: Request,
    now: float,
    slo_ms: float,
    U: float,
    L: float,
    band_margin: float = _P50_BAND_MARGIN,
) -> tuple[TieredProvider, dict[str, float]]:
    """V1 (naive / ablation): P50 rank + near-best band + cheapest-in-band.

    This is the rule inherited from Yiyan's V2 router for within-tier
    selection. When pushed to the cross-tier setting it fails in scenarios
    where the subscription tier has moderately higher P50 but is comfortably
    within the SLO — the band excludes it and forces 100% S_A.
    """
    del slo_ms  # P50-band is not SLO-aware
    candidates = [p for p in providers if p.is_available(now)]
    if not candidates:
        s_a = [p for p in providers if p.tier == ProviderTier.S_A]
        return (min(s_a, key=lambda p: p.cost_per_token), {})

    c_eff = {
        p.name: effective_cost(p, req.total_tokens, now, U=U, L=L)
        for p in candidates
    }

    sorted_by_p50 = sorted(candidates, key=lambda p: p.true_p50_ms(now))
    best_p50 = sorted_by_p50[0].true_p50_ms(now)

    band = [
        p for p in candidates
        if p.true_p50_ms(now) <= best_p50 * (1.0 + band_margin)
    ]
    if not band:
        band = [sorted_by_p50[0]]

    primary = min(band, key=lambda p: c_eff[p.name])
    return primary, c_eff


def _joint_select_slo_safe(
    providers: list[TieredProvider],
    req: Request,
    now: float,
    slo_ms: float,
    U: float,
    L: float,
    safety_margin: float = _P95_SAFETY_MARGIN,
) -> tuple[TieredProvider, dict[str, float]]:
    """V2 (corrected): SLO-anchored P95 filter + cheapest by effective cost.

    A provider enters the candidate set if its P95 TTFT is below
    SLO * safety_margin (default 80 %). Among the SLO-safe providers, the
    one with the lowest effective cost wins. Effective cost folds in the
    quota shadow price psi(z) and the concurrency price lambda(u), so
    subscriptions are preferred while capacity is cheap and S_A naturally
    takes over as shadow prices rise.
    """
    candidates = [p for p in providers if p.is_available(now)]
    if not candidates:
        s_a = [p for p in providers if p.tier == ProviderTier.S_A]
        return (min(s_a, key=lambda p: p.cost_per_token), {})

    threshold_ms = slo_ms * safety_margin
    safe = [p for p in candidates if _provider_p95_at(p, now) <= threshold_ms]
    if not safe:
        # No provider passes the safety filter — this is a problem case,
        # but we still have to route somewhere. Pick the provider with the
        # lowest P95 (best shot at meeting SLO) and let the hedger decide
        # whether to back it up.
        safe = [min(candidates, key=lambda p: _provider_p95_at(p, now))]

    c_eff = {
        p.name: effective_cost(p, req.total_tokens, now, U=U, L=L)
        for p in candidates
    }
    primary = min(safe, key=lambda p: c_eff[p.name])
    return primary, c_eff


# ---------------------------------------------------------------------------
# Joint runner (parameterized over selector and hedge flag)
# ---------------------------------------------------------------------------


def _pick_cross_tier_backup(
    providers: list[TieredProvider],
    primary: TieredProvider,
    now: float,
) -> TieredProvider | None:
    """Fastest provider in any tier other than `primary`'s."""
    others = [
        p for p in providers
        if p.tier != primary.tier and p.is_available(now)
    ]
    if not others:
        others = [p for p in providers if p.name != primary.name]
    if not others:
        return None
    return min(others, key=lambda p: p.true_p50_ms(now))


def _run_joint(
    scenario: TieredScenarioConfig,
    requests: list[Request],
    rng: np.random.Generator,
    *,
    strategy_name: str,
    selector,
    use_hedge: bool,
) -> StrategyRun:
    """Generic joint runner. Dispatches the selector and optionally hedges."""
    _reset_all_state(scenario.providers)

    L, U = calibrate_envelopes(scenario.providers, _TYPICAL_TOKENS)
    V_penalty = U * 10.0

    ttft_ms: list[float] = []
    cost_usd: list[float] = []
    provider_sel: list[str] = []
    tier_sel: list[str] = []
    timestamps: list[float] = []
    hedge_flags: list[bool] = []
    z_series: list[float] = []
    u_series: list[float] = []
    psi_q_series: list[float] = []

    for req in requests:
        t = float(req.timestamp)

        primary, c_eff = selector(
            scenario.providers,
            req,
            t,
            scenario.primary_slo_ms,
            U,
            L,
        )

        T_primary_ms = _sample_ttft(primary, rng, t)
        primary.account_request(
            req.id, t,
            _sample_service_time(primary, rng, t, req.response_tokens),
        )

        hedged = False
        final_ttft_ms = T_primary_ms
        c = primary.marginal_cost(req.total_tokens, t)

        if use_hedge:
            backup = _pick_cross_tier_backup(scenario.providers, primary, t)
            if backup is not None:
                # Hedge trigger: must be slow relative to BOTH the primary's
                # own P50 (signal that this request is on the tail) AND the
                # SLO budget (signal that violation is actually at risk).
                # max(.) chooses whichever threshold says "slow" more
                # conservatively. Without the SLO-budget floor, we fire
                # hedges on the natural right tail of a fast subscription
                # that poses zero SLO risk — which is how an earlier
                # iteration wasted 30 % of cost in S7.
                primary_p50 = primary.true_p50_ms(t)
                slo_ms = scenario.primary_slo_ms
                wait_threshold_ms = max(
                    primary_p50 * 1.5,   # 1.5x the primary's typical latency
                    slo_ms * 0.5,        # at least half the SLO budget used
                )
                if T_primary_ms > wait_threshold_ms:
                    P_viol = 0.5
                    remaining_ms = slo_ms - wait_threshold_ms
                    F_backup = (
                        1.0 if backup.true_p50_ms(t) < remaining_ms else 0.3
                    )
                    c_backup_eff = c_eff.get(
                        backup.name,
                        effective_cost(backup, req.total_tokens, t, U=U, L=L),
                    )
                    if P_viol * F_backup > c_backup_eff / V_penalty:
                        T_backup_ms = _sample_ttft(backup, rng, t)
                        backup.account_request(
                            req.id + 10_000_000, t,
                            _sample_service_time(
                                backup, rng, t, req.response_tokens
                            ),
                        )
                        final_ttft_ms = min(
                            T_primary_ms, wait_threshold_ms + T_backup_ms
                        )
                        c += backup.marginal_cost(req.total_tokens, t)
                        hedged = True

        ttft_ms.append(final_ttft_ms)
        cost_usd.append(c)
        provider_sel.append(primary.name)
        tier_sel.append(primary.tier.value)
        timestamps.append(t)
        hedge_flags.append(hedged)

        q_provider = next(
            (p for p in scenario.providers if p.tier == ProviderTier.S_Q), None
        )
        c_provider = next(
            (p for p in scenario.providers if p.tier == ProviderTier.S_C), None
        )
        z_series.append(q_provider.quota.fraction_used(t) if q_provider else 0.0)
        u_series.append(
            c_provider.concurrency.utilization(t) if c_provider else 0.0
        )
        psi_q_series.append(
            quota_shadow_price(q_provider, t, U=U, L=L) if q_provider else 0.0
        )

    return StrategyRun(
        strategy=strategy_name,
        ttft_ms=np.array(ttft_ms),
        cost_usd=np.array(cost_usd),
        provider=provider_sel,
        tier=tier_sel,
        timestamp=np.array(timestamps),
        hedge_triggered=np.array(hedge_flags, dtype=bool),
        quota_fraction_used=np.array(z_series),
        concurrency_utilization=np.array(u_series),
        shadow_price_q=np.array(psi_q_series),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_tiered_strategy(
    scenario: TieredScenarioConfig,
    requests: list[Request],
    strategy: str,
    seed: int = 42,
) -> StrategyRun:
    """Dispatch to the correct strategy runner."""
    if strategy not in TIERED_STRATEGIES:
        raise ValueError(
            f"Unknown tiered strategy: {strategy!r}. Choose from {TIERED_STRATEGIES}"
        )
    from rwsim.strategies.registry import run_registered_strategy

    return run_registered_strategy(scenario, requests, strategy, seed=seed)


__all__ = [
    "BackupSelectionMethod",
    "HedgingParams",
    "HedgingStrategy",
    "SmartHedger",
    "StrategyRun",
    "TIERED_STRATEGIES",
    "V2Router",
    "_joint_select_p50band",
    "_joint_select_slo_safe",
    "_lognormal_p95",
    "_pick_cross_tier_backup",
    "_pick_tier",
    "_provider_p95_at",
    "_reset_all_state",
    "_run_joint",
    "_run_two_layer",
    "_sample_service_time",
    "_sample_ttft",
    "run_tiered_strategy",
]
