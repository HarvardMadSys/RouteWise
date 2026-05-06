"""Shared capacity-state primitives for tiered synthetic providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType


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

    def reset(self, *, window_start: float = 0.0) -> None:
        """Reset mutable quota state."""
        self.used = 0
        self.window_start = float(window_start)


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

    def reset(self, *, window_start: float = 0.0) -> None:
        """Reset every quota window."""
        for window in self.windows:
            window.reset(window_start=window_start)


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

    def reset(self) -> None:
        """Reset mutable concurrency state."""
        self.active = []


@dataclass
class WeightedConcurrencyState:
    """Mutable weighted-concurrency tracker for an S_C provider.

    ``capacity_units`` is the plan/account allotment, and each active request
    consumes the weighted cost associated with its resolved model class.
    """

    capacity_units: int
    model_concurrency_costs_by_class: MappingProxyType[str, int]
    fixed_model_class: str | None = None
    active: list[tuple[float, str, int]] = field(default_factory=list)
    peak_used_concurrency_cost: int = 0
    total_capacity_unit_seconds_used: float = 0.0

    def __post_init__(self) -> None:
        self.capacity_units = int(self.capacity_units)
        if self.capacity_units <= 0:
            raise ValueError("WeightedConcurrencyState capacity_units must be > 0")
        costs = {
            str(model_class): int(concurrency_cost)
            for model_class, concurrency_cost in self.model_concurrency_costs_by_class.items()
        }
        if not costs:
            raise ValueError(
                "WeightedConcurrencyState requires at least one model concurrency cost"
            )
        for model_class, concurrency_cost in costs.items():
            if concurrency_cost <= 0:
                raise ValueError(
                    "WeightedConcurrencyState concurrency costs must be > 0, "
                    f"got {concurrency_cost} for {model_class!r}"
                )
        self.model_concurrency_costs_by_class = MappingProxyType(costs)
        if self.fixed_model_class is not None:
            self.fixed_model_class = str(self.fixed_model_class)
            if self.fixed_model_class not in costs:
                raise ValueError(
                    "WeightedConcurrencyState fixed_model_class must exist in "
                    "model_concurrency_costs_by_class"
                )

    @property
    def limit(self) -> int:
        """Return the effective fixed-model slot limit when one is configured."""
        if self.fixed_model_class is None:
            return self.capacity_units
        cost = self.concurrency_cost(self.fixed_model_class)
        assert cost is not None
        return self.capacity_units // cost

    def concurrency_cost(self, model_class: str) -> int | None:
        """Return weighted capacity cost for a resolved model class."""
        return self.model_concurrency_costs_by_class.get(str(model_class))

    def _model_class_for_interval(self, model_class: str | None = None) -> str:
        resolved = self.fixed_model_class if model_class is None else str(model_class)
        if resolved is None:
            raise ValueError(
                "WeightedConcurrencyState interval admission requires a model class"
            )
        return resolved

    def release_finished(self, now: float) -> None:
        """Release requests whose finish time is at or before ``now``."""
        self.active = [
            (finish_time, model_class, cost)
            for finish_time, model_class, cost in self.active
            if finish_time > now
        ]

    def used_concurrency_cost(self, now: float | None = None) -> int:
        """Return currently occupied weighted capacity units."""
        if now is not None:
            self.release_finished(now)
        return sum(cost for _, _, cost in self.active)

    def utilization(self, now: float | None = None) -> float:
        """Return weighted utilization as a value in [0, 1]."""
        return min(self.used_concurrency_cost(now) / self.capacity_units, 1.0)

    def can_admit(self, model_class: str, now: float | None = None) -> bool:
        """Return whether a request with ``model_class`` can enter immediately."""
        cost = self.concurrency_cost(model_class)
        if cost is None:
            return False
        return self.used_concurrency_cost(now) + cost <= self.capacity_units

    def can_admit_interval(
        self,
        start: float,
        end: float,
        model_class: str | None = None,
    ) -> bool:
        """Return whether the fixed/scoped model can enter over ``[start, end)``."""
        del end
        return self.can_admit(self._model_class_for_interval(model_class), now=start)

    def admit(
        self,
        model_class: str,
        finish_time: float,
        *,
        now: float | None = None,
    ) -> bool:
        """Admit one request if weighted capacity is available.

        Returns ``True`` when the request was recorded and ``False`` when the
        resolved model class is incompatible or capacity is full.
        """
        if now is not None and finish_time <= now:
            raise ValueError("finish_time must be greater than now")
        cost = self.concurrency_cost(model_class)
        if cost is None:
            return False
        if self.used_concurrency_cost(now) + cost > self.capacity_units:
            return False
        self.active.append((float(finish_time), str(model_class), cost))
        if now is not None:
            # Current cost-layer S_C runs do not cancel admitted requests; if
            # cancellation is added, account this at completion/cancel events.
            self.total_capacity_unit_seconds_used += cost * (
                float(finish_time) - float(now)
            )
        self.peak_used_concurrency_cost = max(
            self.peak_used_concurrency_cost,
            self.used_concurrency_cost(),
        )
        return True

    def admit_interval(
        self,
        now: float,
        service_time_sec: float,
        model_class: str | None = None,
    ) -> bool:
        """Record one fixed/scoped-model request interval."""
        return self.admit(
            self._model_class_for_interval(model_class),
            now + service_time_sec,
            now=now,
        )

    def reset(self) -> None:
        """Reset mutable weighted concurrency state."""
        self.active = []
        self.peak_used_concurrency_cost = 0
        self.total_capacity_unit_seconds_used = 0.0


__all__ = [
    "ConcurrencyState",
    "MultiWindowQuotaState",
    "ProviderTier",
    "QuotaState",
    "WeightedConcurrencyState",
]
