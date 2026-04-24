"""Fixed provider-selection cost routers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, TypeVar


class CostedProvider(Protocol):
    """Provider shape needed by fixed cost routers."""

    name: str
    cost_per_token: float


T_CostedProvider = TypeVar("T_CostedProvider", bound=CostedProvider)


def cheapest_provider(providers: Sequence[T_CostedProvider]) -> T_CostedProvider:
    """Return the provider with the lowest token cost."""
    if not providers:
        raise ValueError("No providers available")
    return min(providers, key=lambda provider: provider.cost_per_token)


def cheapest_provider_name(providers: Sequence[CostedProvider]) -> str:
    """Return the name of the provider with the lowest token cost."""
    return cheapest_provider(providers).name


def hindsight_fastest_provider(
    providers: Sequence[Any],
    *,
    seed: int = 0,
    samples_per_phase: int = 5000,
) -> SyntheticProvider:
    """Return the lowest time-averaged P50 provider.

    Shifting providers are evaluated with half the samples from each phase,
    preserving the historical `fastest_fixed` behavior.
    """
    if not providers:
        raise ValueError("No providers available")

    import numpy as np

    rng = np.random.default_rng(seed)
    best_p50: dict[str, float] = {}
    provider_by_name = {provider.name: provider for provider in providers}
    for provider in providers:
        ttft_dist_after = getattr(provider, "ttft_dist_after", None)
        if ttft_dist_after is not None:
            before = provider.ttft_dist.sample(rng, samples_per_phase)
            after = ttft_dist_after.sample(rng, samples_per_phase)
            best_p50[provider.name] = float(np.median(np.concatenate([before, after])))
        else:
            samples = provider.ttft_dist.sample(rng, samples_per_phase * 2)
            best_p50[provider.name] = float(np.median(samples))

    return provider_by_name[min(best_p50, key=lambda name: best_p50[name])]


__all__ = [
    "CostedProvider",
    "cheapest_provider",
    "cheapest_provider_name",
    "hindsight_fastest_provider",
]
