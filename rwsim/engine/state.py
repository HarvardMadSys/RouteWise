"""Runtime state owned by the simulation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderRuntimeState:
    """Mutable runtime state for one provider."""

    quota_used: int = 0
    quota_window_start: float = 0.0
    active_requests: list[tuple[float, int]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationState:
    """Mutable state passed through policy and execution stages."""

    current_time: float = 0.0
    providers: dict[str, ProviderRuntimeState] = field(default_factory=dict)
    policy_state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def provider(self, name: str) -> ProviderRuntimeState:
        """Return runtime state for a provider, creating it if needed."""
        return self.providers.setdefault(name, ProviderRuntimeState())


__all__ = ["ProviderRuntimeState", "SimulationState"]
