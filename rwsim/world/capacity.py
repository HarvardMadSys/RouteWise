"""Shared capacity-state primitives for tiered synthetic providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ProviderTier(str, Enum):
    """Pricing/capacity regime for a provider."""

    S_A = "api"
    S_Q = "quota"
    S_C = "concurrency"


@dataclass
class QuotaState:
    """Mutable quota usage tracker for an S_Q provider."""

    size: int
    window_sec: float = 86400.0
    used: int = 0
    window_start: float = 0.0

    def roll_window(self, now: float) -> None:
        """Reset the counter if the current window has expired."""
        if now - self.window_start >= self.window_sec:
            self.window_start = now
            self.used = 0

    def fraction_used(self, now: float) -> float:
        """Return quota usage in the current window as a value in [0, 1]."""
        self.roll_window(now)
        if self.size <= 0:
            return 1.0
        return min(self.used / self.size, 1.0)

    def can_admit(self, now: float) -> bool:
        """Return whether one more request can use quota."""
        self.roll_window(now)
        return self.used < self.size

    def charge(self, now: float) -> None:
        """Consume one quota unit in the current window."""
        self.roll_window(now)
        self.used += 1


@dataclass
class ConcurrencyState:
    """Mutable concurrency tracker for an S_C provider."""

    limit: int
    active: list[tuple[float, int]] = field(default_factory=list)

    def _sweep(self, now: float) -> None:
        """Remove completed requests from the active list."""
        self.active = [(t_end, rid) for (t_end, rid) in self.active if t_end > now]

    def utilization(self, now: float) -> float:
        """Return active slot usage as a value in [0, 1]."""
        self._sweep(now)
        if self.limit <= 0:
            return 1.0
        return min(len(self.active) / self.limit, 1.0)

    def can_admit(self, now: float) -> bool:
        """Return whether one more request can enter immediately."""
        self._sweep(now)
        return len(self.active) < self.limit

    def admit(self, request_id: int, now: float, service_time_sec: float) -> None:
        """Mark one request as active until its service end time."""
        self._sweep(now)
        self.active.append((now + service_time_sec, request_id))


__all__ = [
    "ConcurrencyState",
    "ProviderTier",
    "QuotaState",
]
