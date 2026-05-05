"""Shared schemas for the target RouteWise simulator architecture.

These dataclasses are intentionally lightweight and dependency-free. Add fields
only when they are shared across module boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProviderTier(str, Enum):
    """Provider pricing or capacity regime."""

    API = "api"
    QUOTA = "quota"
    CONCURRENCY = "concurrency"


@dataclass(frozen=True)
class DistributionConfig:
    """Configuration for a named random distribution."""

    name: str
    params: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ShiftEvent:
    """Provider distribution change at simulated time."""

    time_sec: float
    ttft_distribution: DistributionConfig


@dataclass(frozen=True)
class ProviderConfig:
    """Static provider configuration loaded from an experiment config."""

    name: str
    tier: ProviderTier
    cost_per_token: float
    ttft_distribution: DistributionConfig
    tps_distribution: DistributionConfig | None = None
    quota_size: int | None = None
    quota_window_sec: float | None = None
    concurrency_limit: int | None = None
    service_time_distribution: DistributionConfig | None = None
    shift_events: tuple[ShiftEvent, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkloadConfig:
    """Static trace-workload configuration."""

    n_requests: int
    duration_seconds: float
    arrival_process: str = "trace"
    seed: int = 0
    start_time: float = 0.0
    input_tokens: int = 100
    output_token_distribution: DistributionConfig | None = None
    source: str = "trace"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioConfig:
    """Generic scenario config.

    Paper-specific scenarios should be expressed as external configs and
    loaded into this shape, not hard-coded in this module.
    """

    name: str
    description: str
    providers: tuple[ProviderConfig, ...]
    workload: WorkloadConfig
    primary_slo_ms: float
    slo_thresholds_ms: tuple[float, ...] = (1000.0, 2000.0, 3000.0, 5000.0)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Request:
    """One simulated inference request."""

    id: int
    timestamp: float
    request_tokens: int
    estimated_response_tokens: float | None = None
    response_tokens: int | None = None
    total_tokens: int | None = None
    model: str | None = None
    provider: str | None = None
    actual_cost: float | None = None
    latency_ms: int | None = None
    ttft_ms: int | None = None
    slo_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def day(self) -> int:
        """Return the day index for second-based timestamps."""
        return int(self.timestamp) // 86400

    @property
    def time_of_day(self) -> int:
        """Return seconds within the current day."""
        return int(self.timestamp) % 86400

    @property
    def latency_seconds(self) -> float:
        """Return observed or token-estimated latency in seconds."""
        if self.latency_ms is not None:
            return self.latency_ms / 1000.0
        if self.total_tokens is None:
            return 0.0
        return self.total_tokens / 2000.0


@dataclass(frozen=True)
class RoutingDecision:
    """Policy decision for one request."""

    primary_provider: str
    hedge_checkpoints: tuple[float, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HedgeDispatch:
    """In-flight hedge dispatch requested by a policy checkpoint."""

    backup_provider: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutingOutcome:
    """Realized execution result for one request."""

    request_id: int
    primary_provider: str
    final_provider: str
    ttft_ms: float
    cost_usd: float
    hedge_triggered: bool = False
    rejected: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricRecord:
    """Uniform per-request metric emitted by the engine."""

    request: Request
    decision: RoutingDecision
    outcome: RoutingOutcome
    state_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationResult:
    """Simulation output built from a uniform metrics stream."""

    records: tuple[MetricRecord, ...]
    aggregates: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "DistributionConfig",
    "MetricRecord",
    "ProviderConfig",
    "ProviderTier",
    "Request",
    "HedgeDispatch",
    "RoutingDecision",
    "RoutingOutcome",
    "ScenarioConfig",
    "ShiftEvent",
    "SimulationResult",
    "WorkloadConfig",
]
