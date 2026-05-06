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
        """Reset the counter on fixed ``window_sec`` boundaries."""
        if self.window_sec <= 0:
            self.window_start = now
            self.used = 0
            return
        if now - self.window_start >= self.window_sec:
            elapsed_windows = int((now - self.window_start) // self.window_sec)
            self.window_start += elapsed_windows * self.window_sec
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

    def reset(self) -> None:
        """Reset mutable quota state."""
        self.used = 0
        self.window_start = 0.0


@dataclass
class MultiWindowQuotaState:
    """Mutable quota tracker for plans with multiple simultaneous windows."""

    windows: tuple[QuotaState, ...]

    def __post_init__(self) -> None:
        if not self.windows:
            raise ValueError("MultiWindowQuotaState requires at least one window")

    def fraction_used(self, now: float) -> float:
        """Return the tightest quota usage fraction across all windows."""
        return max(window.fraction_used(now) for window in self.windows)

    def can_admit(self, now: float) -> bool:
        """Return whether every quota window has capacity."""
        return all(window.can_admit(now) for window in self.windows)

    def charge(self, now: float) -> None:
        """Consume one quota unit from every window."""
        if not self.can_admit(now):
            raise RuntimeError("Multi-window quota exceeded")
        for window in self.windows:
            window.charge(now)

    def reset(self) -> None:
        """Reset every quota window."""
        for window in self.windows:
            window.reset()


@dataclass
class ConcurrencyState:
    """Mutable concurrency tracker for an S_C provider."""

    limit: int
    active: list[tuple[float, float, int]] = field(default_factory=list)

    def _sweep(self, now: float) -> None:
        """Remove intervals that completed before ``now``."""
        self.active = [
            (t_start, t_end, rid)
            for (t_start, t_end, rid) in self.active
            if t_end > now
        ]

    def _active_count_at(self, now: float) -> int:
        """Return the number of intervals occupying a slot at ``now``."""
        return sum(1 for t_start, t_end, _ in self.active if t_start <= now < t_end)

    def utilization(self, now: float) -> float:
        """Return active slot usage as a value in [0, 1]."""
        self._sweep(now)
        if self.limit <= 0:
            return 1.0
        return min(self._active_count_at(now) / self.limit, 1.0)

    def can_admit(self, now: float) -> bool:
        """Return whether one more request can enter immediately."""
        self._sweep(now)
        return self._active_count_at(now) < self.limit

    def can_admit_interval(self, start: float, end: float) -> bool:
        """Return whether one more request fits throughout [start, end)."""
        self._sweep(start)
        if self.limit <= 0:
            return False
        if end <= start:
            return self.can_admit(start)

        event_points = {start}
        for t_start, t_end, _ in self.active:
            if t_start < end and start < t_end and start <= t_start < end:
                event_points.add(t_start)

        return all(self._active_count_at(point) < self.limit for point in event_points)

    def admit(self, request_id: int, now: float, service_time_sec: float) -> None:
        """Record one request's concurrency interval."""
        self._sweep(now)
        end = now + service_time_sec
        if not self.can_admit_interval(now, end):
            raise RuntimeError(
                "Concurrency limit exceeded for interval "
                f"[{now:.3f}, {end:.3f}) with limit={self.limit}"
            )
        self.active.append((now, end, request_id))


__all__ = [
    "ConcurrencyState",
    "MultiWindowQuotaState",
    "ProviderTier",
    "QuotaState",
]
