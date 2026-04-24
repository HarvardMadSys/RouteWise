"""Tiered latency-filter helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence

from rwsim.policies.cost_routers.tiered import effective_costs
from rwsim.schemas import Request
from rwsim.world.distributions import LogNormal
from rwsim.world.providers import ProviderTier, TieredProvider


def lognormal_p95(dist: LogNormal) -> float:
    """Analytical P95 for a lognormal distribution."""
    return math.exp(dist.mu + 1.645 * dist.sigma)


def provider_p95_at(provider: TieredProvider, now: float) -> float:
    """P95 TTFT for a provider at simulated time."""
    if hasattr(provider, "_active_dist"):
        dist = provider._active_dist(now)  # noqa: SLF001 - shift helper
    else:
        dist = provider.ttft_dist
    return lognormal_p95(dist)


def select_p50_band(
    providers: Sequence[TieredProvider],
    request: Request,
    now: float,
    slo_ms: float,
    U: float,
    L: float,
    band_margin: float,
) -> tuple[TieredProvider, dict[str, float]]:
    """P50 rank + near-best band + cheapest-in-band selector."""
    del slo_ms
    candidates = [provider for provider in providers if provider.is_available(now)]
    if not candidates:
        api_providers = [provider for provider in providers if provider.tier == ProviderTier.S_A]
        return min(api_providers, key=lambda provider: provider.cost_per_token), {}

    c_eff = effective_costs(candidates, request, now, U=U, L=L)
    sorted_by_p50 = sorted(candidates, key=lambda provider: provider.true_p50_ms(now))
    best_p50 = sorted_by_p50[0].true_p50_ms(now)

    band = [
        provider
        for provider in candidates
        if provider.true_p50_ms(now) <= best_p50 * (1.0 + band_margin)
    ]
    if not band:
        band = [sorted_by_p50[0]]

    return min(band, key=lambda provider: c_eff[provider.name]), c_eff


def select_slo_safe(
    providers: Sequence[TieredProvider],
    request: Request,
    now: float,
    slo_ms: float,
    U: float,
    L: float,
    safety_margin: float,
) -> tuple[TieredProvider, dict[str, float]]:
    """SLO-anchored P95 filter + cheapest effective cost selector."""
    candidates = [provider for provider in providers if provider.is_available(now)]
    if not candidates:
        api_providers = [provider for provider in providers if provider.tier == ProviderTier.S_A]
        return min(api_providers, key=lambda provider: provider.cost_per_token), {}

    threshold_ms = slo_ms * safety_margin
    safe = [provider for provider in candidates if provider_p95_at(provider, now) <= threshold_ms]
    if not safe:
        safe = [min(candidates, key=lambda provider: provider_p95_at(provider, now))]

    c_eff = effective_costs(candidates, request, now, U=U, L=L)
    return min(safe, key=lambda provider: c_eff[provider.name]), c_eff


__all__ = [
    "lognormal_p95",
    "provider_p95_at",
    "select_p50_band",
    "select_slo_safe",
]
