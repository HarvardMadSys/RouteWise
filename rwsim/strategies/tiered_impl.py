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

import numpy as np

from rwsim.policies.cost_routers.tiered import (
    pick_available_tier as _pick_tier,
    pick_two_layer_provider,
)
from rwsim.policies.hedgers import (
    BackupSelectionMethod,
    HedgingParams,
    HedgingStrategy,
    SmartHedger,
)
from rwsim.policies.latency_routers.tiered_filters import (
    lognormal_p95 as _lognormal_p95,
    provider_p95_at as _provider_p95_at,
    select_p50_band,
    select_slo_safe,
)
from rwsim.policies.latency_routers import V2Router
from rwsim.schemas import Request
from rwsim.world.metrics import StrategyRun
from rwsim.world.providers import ProviderTier, TieredProvider
from rwsim.world.scenarios import ScenarioConfig as TieredScenarioConfig
from rwsim.world.shadow_price import (
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


# ---------------------------------------------------------------------------
# Strategy: two_layer (current RouteWise design)
# ---------------------------------------------------------------------------


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

        primary = pick_two_layer_provider(scenario.providers, t)

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
    return select_p50_band(providers, req, now, slo_ms, U, L, band_margin)


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
    return select_slo_safe(providers, req, now, slo_ms, U, L, safety_margin)


# ---------------------------------------------------------------------------
# Joint runner (parameterized over selector and hedge flag)
# ---------------------------------------------------------------------------


def _pick_cross_tier_backup(
    providers: list[TieredProvider],
    primary: TieredProvider,
    now: float,
) -> TieredProvider | None:
    """Fastest available provider in any tier other than `primary`'s."""
    others = [
        p for p in providers
        if p.tier != primary.tier and p.is_available(now)
    ]
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
                        if not backup.is_available(t):
                            raise RuntimeError(
                                f"Selected unavailable hedge backup {backup.name!r} "
                                f"at t={t:.3f}"
                            )
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
