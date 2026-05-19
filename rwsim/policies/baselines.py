"""Paper-facing baseline policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from rwsim.policies.base import NoOpObserveMixin, NoOpTickMixin
from rwsim.policies.prefix_cache import cache_aware_marginal_cost
from rwsim.schemas import Request, RoutingDecision
from rwsim.world.capacity import ProviderTier

if TYPE_CHECKING:
    from rwsim.engine.state import SimulationState
    from rwsim.world.providers import Provider


# Tie-break order for greedy_cost when multiple providers share the same real
# marginal cost (typically S_C and S_Q both at 0). Concurrency slots are
# perishable — unused slots cannot be banked — so burn them first; quota is
# bankable, so save it for when concurrency saturates; paid API last.
_GREEDY_COST_TIER_RANK: dict[ProviderTier, int] = {
    ProviderTier.S_C: 0,
    ProviderTier.S_Q: 1,
    ProviderTier.S_A: 2,
}
_UNKNOWN_TIER_RANK = 99


_BASELINE_MODES = frozenset({"greedy_cost", "greedy_latency", "random"})


@dataclass
class BaselinePolicy(NoOpTickMixin, NoOpObserveMixin):
    """Paper-facing greedy-cost, greedy-latency, and random baselines."""

    mode: str
    seed: int = 0
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in _BASELINE_MODES:
            raise ValueError(f"Unknown baseline mode: {self.mode!r}")
        self.rng = np.random.default_rng(self.seed)

    def route(self, request: Request, state: SimulationState) -> RoutingDecision:
        """Choose one available provider."""
        providers = [provider for provider in state.providers.values() if provider.is_available(state.now)]
        if not providers:
            providers = list(state.providers.values())
        if not providers:
            raise ValueError("No providers configured for baseline policy.")

        if self.mode == "greedy_cost":
            primary = min(
                providers,
                key=lambda provider: (
                    cache_aware_marginal_cost(provider, request, state, now=state.now),
                    _GREEDY_COST_TIER_RANK.get(provider.tier, _UNKNOWN_TIER_RANK),
                    provider.true_mean_ms(state.now),
                    provider.name,
                ),
            )
        elif self.mode == "greedy_latency":
            primary = min(
                providers,
                key=lambda provider: (
                    provider.true_mean_ms(state.now),
                    cache_aware_marginal_cost(provider, request, state, now=state.now),
                    provider.name,
                ),
            )
        else:
            primary = providers[int(self.rng.integers(0, len(providers)))]

        return RoutingDecision(
            primary_provider=primary.name,
            metadata={"policy": self.mode},
        )


def cheapest_provider(providers: list[Provider] | tuple[Provider, ...]) -> Provider:
    """Return the cheapest provider by marginal token price."""
    return min(
        providers,
        key=lambda provider: (
            provider.effective_input_cost_per_token,
            provider.effective_output_cost_per_token,
            provider.name,
        ),
    )


__all__ = ["BaselinePolicy", "cheapest_provider"]
