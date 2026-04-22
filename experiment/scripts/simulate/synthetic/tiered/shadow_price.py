"""Shadow-price functions for tiered provider capacity.

These functions convert capacity-constraint state into a per-request
opportunity cost. Adding the shadow price to the marginal cost yields
the `effective cost` used by the joint router.

Quota (S_Q) shadow price
------------------------
Uses the primal-dual exponential threshold from the RouteWise paper:

    psi(z) = L * (U / L) ** z

where:
    z = fraction of quota used in the current window (in [0, 1])
    L = smallest positive API price among S_A providers (near-zero floor)
    U = largest API price among S_A providers (the worst-case opportunity cost)

Intuition: when the quota is nearly empty, psi(z) ~ L (cheap, so S_Q wins).
When the quota is nearly exhausted, psi(z) ~ U (expensive, so S_A takes over).
The exponential schedule is monotonically increasing and smooth, so the
downstream V2 or LP decision does not jump discontinuously at depletion.

Concurrency (S_C) shadow price
------------------------------
Congestion price proportional to current slot utilization. The rigorous
form from the Stage-2 primal-dual is:

    lambda = min(value_density of queued requests)   if saturated
    lambda = 0                                       otherwise

Without an explicit queue in our simulator we use a simplified proxy:

    lambda(u) = U * u^alpha

where u = active/limit is the slot utilization and alpha controls how
aggressively spillover kicks in. alpha=2 means congestion is ignored until
the provider is fairly full, matching the "hard spill" production behavior.
Set alpha=1 for a linear schedule.
"""

from __future__ import annotations

import math

from .providers import ProviderTier, TieredProvider


# ---------------------------------------------------------------------------
# Quota shadow price
# ---------------------------------------------------------------------------


def quota_shadow_price(
    provider: TieredProvider,
    now: float,
    *,
    U: float,
    L: float,
) -> float:
    """Exponential threshold for a quota-limited provider.

    Parameters
    ----------
    provider:
        Must have tier == S_Q.
    now:
        Simulated timestamp.
    U, L:
        Upper and lower cost envelopes. `U` is the maximum API cost in the
        provider set; `L` is a small positive floor (e.g. min API cost / 1e3).

    Returns
    -------
    Shadow price in USD per request. Always non-negative and in [L, U].
    """
    if provider.tier != ProviderTier.S_Q:
        return 0.0
    assert provider.quota is not None

    z = provider.quota.fraction_used(now)
    # Clamp to [0, 1) to keep psi finite.
    z = min(max(z, 0.0), 0.9999)

    if L <= 0 or U <= 0 or U <= L:
        raise ValueError(f"Require 0 < L < U; got L={L}, U={U}")

    return L * math.pow(U / L, z)


# ---------------------------------------------------------------------------
# Concurrency shadow price
# ---------------------------------------------------------------------------


def concurrency_shadow_price(
    provider: TieredProvider,
    now: float,
    *,
    U: float,
    alpha: float = 2.0,
) -> float:
    """Congestion price for a concurrency-limited provider.

    Parameters
    ----------
    provider:
        Must have tier == S_C.
    now:
        Simulated timestamp.
    U:
        Upper cost envelope (same definition as `quota_shadow_price`).
    alpha:
        Convexity of the congestion schedule. alpha=1 gives a linear ramp
        from 0 to U as utilization goes 0 -> 1; alpha=2 keeps lambda small
        until the provider is nearly full, matching production hard-spill.

    Returns
    -------
    Shadow price in USD per request. Always non-negative and in [0, U].
    """
    if provider.tier != ProviderTier.S_C:
        return 0.0
    assert provider.concurrency is not None

    u = provider.concurrency.utilization(now)
    return U * math.pow(u, alpha)


# ---------------------------------------------------------------------------
# Effective cost aggregator
# ---------------------------------------------------------------------------


def effective_cost(
    provider: TieredProvider,
    total_tokens: int,
    now: float,
    *,
    U: float,
    L: float,
    concurrency_alpha: float = 2.0,
    latency_alpha: float = 0.0,
) -> float:
    """Unified effective cost used by the joint router.

    c_eff = marginal_cost + quota_shadow_price + concurrency_shadow_price
            + latency_alpha * P50_ms

    For S_A: c_eff = price * tokens           (quota/concurrency prices are 0).
    For S_Q: c_eff = 0 + psi(z)               (marginal is free).
    For S_C: c_eff = 0 + lambda(u)            (marginal is free).

    The optional ``latency_alpha`` term adds a latency-weighted penalty so
    the router can trade cost for faster providers without a separate V2
    algorithm. ``latency_alpha = 0`` preserves the original cost-first
    joint objective; larger values bias the selection toward lower P50.
    Units: USD per ms.

    Args:
        provider: The TieredProvider to price.
        total_tokens: Request size for S_A marginal cost.
        now: Simulated time (seconds).
        U: Upper envelope for shadow price normalization.
        L: Lower envelope for shadow price normalization.
        concurrency_alpha: Exponent for the concurrency penalty curve.
        latency_alpha: Weight on the provider's P50 (in ms). Default 0.
    """
    marginal = provider.marginal_cost(total_tokens, now)
    q_sp = quota_shadow_price(provider, now, U=U, L=L)
    c_sp = concurrency_shadow_price(provider, now, U=U, alpha=concurrency_alpha)
    lat_term = 0.0
    if latency_alpha > 0.0:
        lat_term = latency_alpha * provider.true_p50_ms(now)
    return marginal + q_sp + c_sp + lat_term


# ---------------------------------------------------------------------------
# Envelope calibration
# ---------------------------------------------------------------------------


def calibrate_envelopes(
    providers: list[TieredProvider],
    typical_tokens: int = 200,
    floor_ratio: float = 1e-3,
) -> tuple[float, float]:
    """Compute (L, U) envelopes from the S_A providers in the scenario.

    U is the maximum API cost per request; L is `U * floor_ratio`.
    If no S_A provider exists this falls back to (1e-6, 1e-3).
    """
    api_costs = [
        p.cost_per_token * typical_tokens
        for p in providers
        if p.tier == ProviderTier.S_A and p.cost_per_token > 0
    ]
    if not api_costs:
        return (1e-6, 1e-3)
    U = max(api_costs)
    L = max(U * floor_ratio, 1e-9)
    return (L, U)
