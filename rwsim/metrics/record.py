"""Canonical per-request metric record shared by simulator and real eval."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Status(str, Enum):
    """Canonical request outcome status."""

    SUCCESS = "SUCCESS"
    REJECTED = "REJECTED"
    RATE_LIMITED = "RATE_LIMITED"
    ERROR = "ERROR"


@dataclass
class PerRequestRecord:
    """One user-visible request outcome in the unified metrics schema."""

    # Identity
    request_id: str
    elapsed_sec: float
    policy: str

    prompt_tokens: int
    completion_tokens_budget: int | float | None
    completion_tokens_actual: int | None

    primary_provider: str
    primary_tier: str
    final_provider: str
    final_tier: str
    backup_provider: str | None = None
    backup_tier: str | None = None

    # Identity extensions (cross-source disambiguation)
    model: str | None = None
    source: str | None = None
    timestamp_sec: float | None = None

    ttft_ms: float = 0.0
    e2e_ms: float | None = None
    primary_local_ttft_ms: float | None = None
    backup_local_ttft_ms: float | None = None
    slo_ms: float | None = None
    slo_violated: bool = False

    # Canonical accounting cost — default metrics read these.
    # total = primary + backup when backup exists.
    total_cost_usd: float = 0.0
    primary_cost_usd: float = 0.0
    backup_cost_usd: float | None = None

    # RouteWise decision-time cost estimate (optional, LP debug).
    routing_estimated_cost_usd: float | None = None
    primary_routing_estimated_cost_usd: float | None = None
    backup_routing_estimated_cost_usd: float | None = None

    # Upstream/provider-side physical cost (optional, ops / reconciliation / margin).
    physical_cost_usd: float | None = None
    primary_physical_cost_usd: float | None = None
    backup_physical_cost_usd: float | None = None

    hedge_triggered: bool = False
    hedge_delay_ms: float | None = None
    hedge_winner: str | None = None

    # Hedging algorithm + schedule (disambiguates hedge_delay_ms semantics).
    # hedge_algorithm: "probability_target" | "disabled"
    # hedge_schedule:  "slo_relative_checkpoints"
    hedge_algorithm: str | None = None
    hedge_schedule: str | None = None

    # LP state at routing time.
    # lp_status: "feasible" | "single_candidate" | "all_over_budget" | "no_candidates"
    lp_weights: dict[str, float] | None = None
    lp_budget_usd: float | None = None
    lp_status: str | None = None

    status: Status = Status.SUCCESS
    error_class: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["PerRequestRecord", "Status"]
