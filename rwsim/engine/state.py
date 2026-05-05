"""Runtime state owned by the simulation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any

from rwsim.world.providers import Provider


@dataclass
class CapacityState:
    """Read-only capacity view over provider runtime objects."""

    providers: Mapping[str, Provider] = field(default_factory=dict)

    def quota_fraction_used(self, provider_name: str, now: float) -> float:
        """Return quota usage for a provider, or 0 for non-quota providers."""
        provider = self.providers[provider_name]
        if provider.quota is None:
            return 0.0
        return provider.quota.fraction_used(now)

    def concurrency_utilization(self, provider_name: str, now: float) -> float:
        """Return concurrency utilization for a provider, or 0 otherwise."""
        provider = self.providers[provider_name]
        if provider.concurrency is None:
            return 0.0
        return provider.concurrency.utilization(now)


@dataclass
class SimulationState:
    """Mutable world state exposed to policies as a snapshot."""

    now: float = 0.0
    providers: Mapping[str, Provider] = field(default_factory=dict)
    capacity: CapacityState = field(default_factory=CapacityState)
    user_last_provider: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_providers(
        cls,
        providers: Mapping[str, Provider],
        *,
        now: float = 0.0,
    ) -> "SimulationState":
        """Build state with a matching capacity view."""
        return cls(now=now, providers=providers, capacity=CapacityState(providers))

    @property
    def current_time(self) -> float:
        """Compatibility alias for code that still reads current_time."""
        return self.now

    @current_time.setter
    def current_time(self, value: float) -> None:
        self.now = float(value)


__all__ = ["CapacityState", "SimulationState"]
