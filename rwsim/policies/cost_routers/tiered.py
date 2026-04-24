"""Tier-aware cost-router helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from rwsim.schemas import Request
from rwsim.world.capacity import ProviderTier
from rwsim.world.shadow_price import effective_cost

if TYPE_CHECKING:
    from rwsim.world.providers import TieredProvider


def pick_available_tier(
    providers: Sequence["TieredProvider"],
    now: float,
) -> ProviderTier:
    """Pick the first available tier in subscription-priority order."""
    for tier in (ProviderTier.S_Q, ProviderTier.S_C, ProviderTier.S_A):
        if any(provider.tier == tier and provider.is_available(now) for provider in providers):
            return tier
    return ProviderTier.S_A


def pick_two_layer_provider(
    providers: Sequence["TieredProvider"],
    now: float,
) -> TieredProvider:
    """Pick tier by priority, then lowest-P50 provider within that tier."""
    if not providers:
        raise ValueError("No providers available")

    tier = pick_available_tier(providers, now)
    tier_providers = [
        provider for provider in providers
        if provider.tier == tier and provider.is_available(now)
    ]
    if not tier_providers:
        tier_providers = [
            provider for provider in providers
            if provider.tier == ProviderTier.S_A and provider.is_available(now)
        ]
    if not tier_providers:
        raise ValueError("No API providers available for fallback")
    return min(tier_providers, key=lambda provider: provider.true_p50_ms(now))


def effective_costs(
    providers: Sequence["TieredProvider"],
    request: Request,
    now: float,
    *,
    U: float,
    L: float,
) -> dict[str, float]:
    """Return effective request cost for each candidate provider."""
    return {
        provider.name: effective_cost(provider, request.total_tokens, now, U=U, L=L)
        for provider in providers
    }


def cheapest_effective_provider(
    providers: Sequence["TieredProvider"],
    request: Request,
    now: float,
    *,
    U: float,
    L: float,
) -> tuple["TieredProvider", dict[str, float]]:
    """Pick available provider with lowest effective cost."""
    candidates = [provider for provider in providers if provider.is_available(now)]
    if not candidates:
        candidates = [provider for provider in providers if provider.tier == ProviderTier.S_A]
    if not candidates:
        raise ValueError("No providers available")

    c_eff = effective_costs(candidates, request, now, U=U, L=L)
    return min(candidates, key=lambda provider: c_eff[provider.name]), c_eff


__all__ = [
    "cheapest_effective_provider",
    "effective_costs",
    "pick_available_tier",
    "pick_two_layer_provider",
]
