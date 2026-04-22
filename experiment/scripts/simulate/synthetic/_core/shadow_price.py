"""Shared shadow-price functions for tiered provider capacity."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .capacity import ProviderTier

if TYPE_CHECKING:
    from ..tiered.providers import TieredProvider


def quota_shadow_price(
    provider: "TieredProvider",
    now: float,
    *,
    U: float,
    L: float,
) -> float:
    """Exponential threshold for a quota-limited provider."""
    if provider.tier != ProviderTier.S_Q:
        return 0.0
    assert provider.quota is not None

    z = provider.quota.fraction_used(now)
    z = min(max(z, 0.0), 0.9999)

    if L <= 0 or U <= 0 or U <= L:
        raise ValueError(f"Require 0 < L < U; got L={L}, U={U}")

    return L * math.pow(U / L, z)


def concurrency_shadow_price(
    provider: "TieredProvider",
    now: float,
    *,
    U: float,
    alpha: float = 2.0,
) -> float:
    """Congestion price for a concurrency-limited provider."""
    if provider.tier != ProviderTier.S_C:
        return 0.0
    assert provider.concurrency is not None

    u = provider.concurrency.utilization(now)
    return U * math.pow(u, alpha)


def effective_cost(
    provider: "TieredProvider",
    total_tokens: int,
    now: float,
    *,
    U: float,
    L: float,
    concurrency_alpha: float = 2.0,
    latency_alpha: float = 0.0,
) -> float:
    """Unified effective cost used by the joint router."""
    marginal = provider.marginal_cost(total_tokens, now)
    q_sp = quota_shadow_price(provider, now, U=U, L=L)
    c_sp = concurrency_shadow_price(provider, now, U=U, alpha=concurrency_alpha)
    lat_term = 0.0
    if latency_alpha > 0.0:
        lat_term = latency_alpha * provider.true_p50_ms(now)
    return marginal + q_sp + c_sp + lat_term


def calibrate_envelopes(
    providers: list["TieredProvider"],
    typical_tokens: int = 200,
    floor_ratio: float = 1e-3,
) -> tuple[float, float]:
    """Compute (L, U) envelopes from the S_A providers in the scenario."""
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


__all__ = [
    "calibrate_envelopes",
    "concurrency_shadow_price",
    "effective_cost",
    "quota_shadow_price",
]
