"""Paper-facing baseline policies."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rwsim.engine.state import SimulationState
from rwsim.policies.base import NoOpObserveMixin, NoOpTickMixin
from rwsim.schemas import Request, RoutingDecision
from rwsim.world.providers import Provider


@dataclass
class BaselinePolicy(NoOpTickMixin, NoOpObserveMixin):
    """Greedy-cost, greedy-latency, and random baselines."""

    mode: str
    seed: int = 0
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in {"greedy_cost", "greedy_latency", "random"}:
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
                    provider.marginal_cost(request.total_tokens or 0, state.now),
                    provider.true_mean_ms(state.now),
                    provider.name,
                ),
            )
        elif self.mode == "greedy_latency":
            primary = min(
                providers,
                key=lambda provider: (
                    provider.true_mean_ms(state.now),
                    provider.marginal_cost(request.total_tokens or 0, state.now),
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
    return min(providers, key=lambda provider: (provider.cost_per_token, provider.name))


__all__ = ["BaselinePolicy", "cheapest_provider"]
