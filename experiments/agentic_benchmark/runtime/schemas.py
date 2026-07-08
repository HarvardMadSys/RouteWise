"""Data structures shared across the agentic benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ChatResult:
    """Outcome of one chat-completion call made through the gateway."""

    session_id: str
    request_id: str | None
    status_code: int
    started_at: float  # wall-clock epoch seconds at dispatch
    finished_at: float  # wall-clock epoch seconds at response
    content: str
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def wall_latency_ms(self) -> float:
        """Client-side round-trip latency in milliseconds."""
        return (self.finished_at - self.started_at) * 1000.0


@dataclass
class LLMRequestRecord:
    """One per-request log row read back from the gateway.

    Field names mirror the gateway's log schema. ``cost_usd`` is the
    gateway-side billed cost; ``provider`` is the backend RouteWise actually
    selected; ``session_id`` and ``routewise`` come from the row metadata when
    the admin export is queried with ``include_content``.
    """

    request_id: str
    model_id: str
    provider: str
    timestamp: datetime | None
    status_code: int | None = None
    latency_ms: int | None = None
    ttft_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    error: str | None = None
    session_id: str | None = None
    routewise: dict[str, Any] | None = None

    @classmethod
    def from_api(cls, row: dict[str, Any]) -> LLMRequestRecord:
        """Build a record from one log row.

        Handles both the ``/recent-requests`` shape (float cost, top-level
        ``routewise``) and the ``/admin/export/requests`` shape (string cost,
        ``session_id`` and ``routewise`` nested under ``metadata``).
        """
        ts = row.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        metadata = row.get("metadata")
        meta = metadata if isinstance(metadata, dict) else {}
        cost = row.get("cost_usd")
        return cls(
            request_id=row["request_id"],
            model_id=row.get("model_id", ""),
            provider=row.get("provider", ""),
            timestamp=ts,
            status_code=row.get("status_code"),
            latency_ms=row.get("latency_ms"),
            ttft_ms=row.get("ttft_ms"),
            prompt_tokens=row.get("prompt_tokens"),
            completion_tokens=row.get("completion_tokens"),
            reasoning_tokens=row.get("reasoning_tokens"),
            cache_read_tokens=row.get("cache_read_tokens"),
            cache_write_tokens=row.get("cache_write_tokens"),
            total_tokens=row.get("total_tokens"),
            cost_usd=float(cost) if cost is not None else None,
            error=row.get("error"),
            session_id=row.get("session_id") or meta.get("session_id"),
            routewise=row.get("routewise") or meta.get("routewise"),
        )
