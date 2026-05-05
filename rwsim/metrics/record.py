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

    ttft_ms: float = 0.0
    e2e_ms: float | None = None
    primary_local_ttft_ms: float | None = None
    backup_local_ttft_ms: float | None = None
    slo_ms: float | None = None
    slo_violated: bool = False

    total_cost_usd: float = 0.0
    primary_cost_usd: float = 0.0
    backup_cost_usd: float | None = None

    hedge_triggered: bool = False
    hedge_delay_ms: float | None = None
    hedge_winner: str | None = None

    status: Status = Status.SUCCESS
    error_class: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["PerRequestRecord", "Status"]
