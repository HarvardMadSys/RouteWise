"""Policy implementations for offline counterfactual simulation.

Simple baselines are implemented directly here.
LP routing and hedging reuse production classes from:
  - experiment/strategies/online_latency_router.py (OnlineLatencyRouter)
  - experiment/strategies/smart_hedging.py (SmartHedger)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RequestResult:
    """Result of simulating a single request."""

    provider: str
    ttft_ms: float
    cost_usd: float
    slo_violated: bool
    hedge_triggered: bool = False
    hedge_provider: str | None = None
    hedge_winner: str | None = None


# ---------------------------------------------------------------------------
# Simple fixed-provider baselines
# ---------------------------------------------------------------------------


def cheapest_fixed_provider(costs: dict[str, float]) -> str:
    """Return the cheapest provider by per-request cost."""
    return min(costs, key=lambda p: costs[p])


def fastest_fixed_provider(
    all_ttfts: dict[str, np.ndarray],
) -> str:
    """Return the provider with the lowest overall P50 TTFT.

    This is a static hindsight baseline: uses all data to pick the
    single best provider by median latency. Not achievable online.

    Args:
        all_ttfts: Provider -> array of all TTFT values in ms.
    """
    best = None
    best_p50 = float("inf")
    for prov, ttfts in all_ttfts.items():
        if len(ttfts) == 0:
            continue
        p50 = float(np.median(ttfts))
        if p50 < best_p50:
            best_p50 = p50
            best = prov
    if best is None:
        raise ValueError("No provider has any latency data.")
    return best


def simulate_fixed_policy(
    provider: str,
    latency_pool: np.ndarray | None,
    cost: float,
    slo_ms: float,
    rng: np.random.Generator,
) -> RequestResult | None:
    """Simulate a single request routed to a fixed provider.

    Args:
        provider: Target provider name.
        latency_pool: TTFT samples in ms for this provider in this window.
        cost: Per-request cost in USD.
        slo_ms: SLO threshold in milliseconds.
        rng: Random number generator.

    Returns:
        RequestResult, or None if provider is unavailable (empty pool).
    """
    if latency_pool is None or len(latency_pool) == 0:
        return None
    ttft = float(rng.choice(latency_pool))
    return RequestResult(
        provider=provider,
        ttft_ms=ttft,
        cost_usd=cost,
        slo_violated=ttft > slo_ms,
    )


def simulate_round_robin(
    providers: list[str],
    request_idx: int,
    latency_pools: dict[str, np.ndarray | None],
    costs: dict[str, float],
    slo_ms: float,
    rng: np.random.Generator,
) -> RequestResult | None:
    """Simulate round-robin routing across providers.

    Skips unavailable providers (empty pool) and tries the next one.

    Args:
        providers: Sorted list of provider names.
        request_idx: Global request index for round-robin cycling.
        latency_pools: Provider -> TTFT array for current window.
        costs: Provider -> per-request cost.
        slo_ms: SLO threshold in ms.
        rng: Random number generator.

    Returns:
        RequestResult, or None if all providers are unavailable.
    """
    n = len(providers)
    for offset in range(n):
        prov = providers[(request_idx + offset) % n]
        pool = latency_pools.get(prov)
        if pool is not None and len(pool) > 0:
            return simulate_fixed_policy(prov, pool, costs.get(prov, 0.0), slo_ms, rng)
    return None


def simulate_oracle_best_per_window(
    latency_pools: dict[str, np.ndarray | None],
    costs: dict[str, float],
    slo_ms: float,
    rng: np.random.Generator,
) -> RequestResult | None:
    """Route to the provider with the lowest P50 in the current window.

    This is an oracle upper-bound baseline (not achievable online).
    """
    best_prov = None
    best_p50 = float("inf")
    for prov, pool in latency_pools.items():
        if pool is None or len(pool) == 0:
            continue
        p50 = float(np.median(pool))
        if p50 < best_p50:
            best_p50 = p50
            best_prov = prov

    if best_prov is None:
        return None

    return simulate_fixed_policy(
        best_prov, latency_pools[best_prov], costs.get(best_prov, 0.0), slo_ms, rng
    )


# ---------------------------------------------------------------------------
# Hedge simulation helper (correct causal semantics)
# ---------------------------------------------------------------------------


def simulate_hedge_request(
    primary_ttft_ms: float,
    hedge_time_sec: float,
    backup_pool: np.ndarray | None,
    primary_provider: str,
    backup_provider: str | None,
    primary_cost: float,
    backup_cost: float,
    slo_ms: float,
    dispatch_overhead_sec: float,
    rng: np.random.Generator,
) -> RequestResult:
    """Simulate a single hedged request with correct time-based triggering.

    The hedge triggers at wall-clock time hedge_time_sec. If the primary
    hasn't responded by then, a backup is launched. The final TTFT is
    min(T_primary, hedge_time + dispatch_overhead + T_backup).

    Args:
        primary_ttft_ms: Primary provider TTFT in ms (already drawn).
        hedge_time_sec: Hedge trigger time in seconds.
        backup_pool: Backup provider TTFT samples in ms.
        primary_provider: Primary provider name.
        backup_provider: Backup provider name (None if no backup).
        primary_cost: Cost for primary request.
        backup_cost: Cost for backup request.
        slo_ms: SLO threshold in ms.
        dispatch_overhead_sec: Overhead for launching backup.
        rng: Random number generator.

    Returns:
        RequestResult with hedge details.
    """
    hedge_time_ms = hedge_time_sec * 1000.0
    dispatch_ms = dispatch_overhead_sec * 1000.0

    # No backup available or hedge time is infinite -> no hedging.
    if (
        backup_provider is None
        or backup_pool is None
        or len(backup_pool) == 0
        or hedge_time_sec == float("inf")
    ):
        return RequestResult(
            provider=primary_provider,
            ttft_ms=primary_ttft_ms,
            cost_usd=primary_cost,
            slo_violated=primary_ttft_ms > slo_ms,
        )

    # Primary responds before hedge trigger -> no hedge.
    if primary_ttft_ms <= hedge_time_ms:
        return RequestResult(
            provider=primary_provider,
            ttft_ms=primary_ttft_ms,
            cost_usd=primary_cost,
            slo_violated=primary_ttft_ms > slo_ms,
        )

    # Hedge triggers at hedge_time_ms. Backup launched.
    backup_ttft_ms = float(rng.choice(backup_pool))
    backup_finish_ms = hedge_time_ms + dispatch_ms + backup_ttft_ms

    if primary_ttft_ms <= backup_finish_ms:
        final_ttft = primary_ttft_ms
        winner = "primary"
    else:
        final_ttft = backup_finish_ms
        winner = "backup"

    return RequestResult(
        provider=primary_provider,
        ttft_ms=final_ttft,
        cost_usd=primary_cost + backup_cost,
        slo_violated=final_ttft > slo_ms,
        hedge_triggered=True,
        hedge_provider=backup_provider,
        hedge_winner=winner,
    )
