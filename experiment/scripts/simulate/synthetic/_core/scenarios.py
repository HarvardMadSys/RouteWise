"""Shared scenario configuration types for the synthetic simulator."""

from __future__ import annotations

from dataclasses import dataclass, field

from .providers import Provider


@dataclass
class ScenarioConfig:
    """Configuration for one synthetic simulation scenario."""

    name: str
    description: str
    providers: list[Provider]
    n_requests: int = 2000
    duration_seconds: float = 3600.0
    arrival_process: str = "poisson"
    primary_slo_ms: float = 2000.0
    slo_thresholds_ms: list[float] = field(
        default_factory=lambda: [1000.0, 2000.0, 3000.0, 5000.0]
    )

    @property
    def slo_sec(self) -> float:
        """Return the canonical primary SLO in seconds."""
        return self.primary_slo_ms / 1000.0

    @property
    def provider_names(self) -> list[str]:
        """Return provider names in declared order."""
        return [provider.name for provider in self.providers]


__all__ = ["ScenarioConfig"]
