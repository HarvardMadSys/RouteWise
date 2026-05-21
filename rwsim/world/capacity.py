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
    """Time-aware concurrency tracker for an S_C provider.

    The ``active`` list is an append-only ledger of admitted intervals.
    Read APIs (``utilization``, ``can_admit``, ``can_admit_interval``) are
    side-effect-free pure functions over the current ledger, so they are
    safe under non-monotonic time queries (e.g. when the simulator's hedge
    tick loop temporarily advances ``state.now`` to a future checkpoint
    and then reverts to outer trace time). Memory cleanup is explicit via
    :meth:`gc_before`, which the simulator engine calls between outer
    trace iterations using the previous monotonic trace timestamp.
    """

    limit: int
    active: list[tuple[float, float, int]] = field(default_factory=list)

    def _count_at(self, now: float) -> int:
        """Return the number of intervals occupying a slot at ``now``.

        Pure read; does not mutate ``active``.
        """
        return sum(1 for t_start, t_end, _ in self.active if t_start <= now < t_end)

    def utilization(self, now: float) -> float:
        """Return active slot usage at ``now`` as a value in [0, 1]."""
        if self.limit <= 0:
            return 1.0
        return min(self._count_at(now) / self.limit, 1.0)

    def can_admit(self, now: float) -> bool:
        """Return whether one more request can enter at ``now``."""
        return self._count_at(now) < self.limit

    def can_admit_interval(self, start: float, end: float) -> bool:
        """Return whether one more request fits throughout ``[start, end)``."""
        if self.limit <= 0:
            return False
        if end <= start:
            return self.can_admit(start)

        event_points = {start}
        for t_start, t_end, _ in self.active:
            if t_start < end and start < t_end and start <= t_start < end:
                event_points.add(t_start)

        return all(self._count_at(point) < self.limit for point in event_points)

    def admit(self, request_id: int, now: float, service_time_sec: float) -> None:
        """Record one request's concurrency interval ``[now, now+service_time_sec)``."""
        end = now + service_time_sec
        if not self.can_admit_interval(now, end):
            raise RuntimeError(
                "Concurrency limit exceeded for interval "
                f"[{now:.3f}, {end:.3f}) with limit={self.limit}"
            )
        self.active.append((now, end, request_id))

    def truncate(self, request_id: int, end_time: float) -> bool:
        """Shorten one admitted request interval to end no later than ``end_time``.

        Returns ``True`` when an interval was changed. If ``end_time`` is at
        or before the interval start, the interval is removed because it never
        occupied capacity in the modeled timeline.
        """
        changed = False
        active: list[tuple[float, float, int]] = []
        for t_start, t_end, rid in self.active:
            if rid != request_id or end_time >= t_end:
                active.append((t_start, t_end, rid))
                continue
            changed = True
            if end_time > t_start:
                active.append((t_start, float(end_time), rid))
        if changed:
            self.active = active
        return changed

    def gc_before(self, watermark: float) -> None:
        """Drop intervals ending at or before ``watermark``.

        Memory-only optimization. The caller (typically the simulator
        engine) must guarantee that no future read will query a time
        strictly less than ``watermark``; otherwise the dropped state
        would be needed for correctness.
        """
        self.active = [
            (t_start, t_end, rid) for (t_start, t_end, rid) in self.active if t_end > watermark
        ]

    def reset(self) -> None:
        """Reset mutable concurrency state."""
        self.active = []


@dataclass
class WeightedConcurrencyState:
    """Time-aware weighted-concurrency tracker for an S_C provider.

    ``capacity_units`` is the plan/account allotment, and each active request
    consumes the weighted cost associated with its resolved model class.

    The ``active`` list is an append-only ledger of admitted weighted
    intervals, each entry being ``(start, finish, sequence, request_id,
    model_class, cost)``. Read APIs (``used_concurrency_cost``, ``utilization``,
    ``can_admit``, ``can_admit_interval``) are side-effect-free pure
    functions over the current ledger, so they are safe under non-monotonic
    time queries (e.g. when the simulator's hedge tick loop temporarily
    advances ``state.now`` to a future checkpoint and then reverts to outer
    trace time). Memory cleanup is explicit via :meth:`gc_before`, which
    the simulator engine calls between outer trace iterations using the
    previous monotonic trace timestamp.

    The previous ``_current_used_cost`` counter and finish-time min-heap
    are gone: they could not represent intervals that have not yet started
    (future hedge backups) and were destructively mutated on read.
    """

    capacity_units: int
    model_concurrency_costs_by_class: MappingProxyType[str, int]
    fixed_model_class: str | None = None
    # Append-only ledger of (start, finish, sequence, request_id, model_class, cost).
    active: list[tuple[float, float, int, int, str, int]] = field(default_factory=list)
    # Peak reflects the currently modeled occupancy after cancellations.
    # Explicit GC preserves naturally completed intervals in the historical
    # component; truncate treats canceled tail occupancy as never consumed.
    peak_used_concurrency_cost: int = 0
    total_capacity_unit_seconds_used: float = 0.0
    _sequence: int = field(default=0, init=False, repr=False)
    _historical_peak_used_concurrency_cost: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        # Active entries must use the 6-tuple shape; old 3/4-tuple shapes
        # lacked a start_time and the 5-tuple interim shape lacked request_id,
        # so none can be safely upgraded.
        for entry in self.active:
            if len(entry) != 6:
                raise ValueError(
                    "WeightedConcurrencyState.active entries must be "
                    "(start, finish, sequence, request_id, model_class, cost); "
                    f"got {entry!r}"
                )
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
        # Seed _sequence past any pre-loaded entries so future admits remain unique.
        if self.active:
            self._sequence = max(entry[2] for entry in self.active) + 1

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
            raise ValueError("WeightedConcurrencyState interval admission requires a model class")
        return resolved

    def _used_at(self, now: float) -> int:
        """Return the total weighted cost occupying capacity at ``now``.

        Pure read; does not mutate ``active``.
        """
        return sum(cost for s, e, _, _, _, cost in self.active if s <= now < e)

    def _peak_over_active(self) -> int:
        """Return max active weighted cost over retained interval start points."""
        event_points = {start for start, _, _, _, _, _ in self.active}
        return max((self._used_at(point) for point in event_points), default=0)

    def _peak_over_entries_removed_by_gc(self, watermark: float) -> int:
        """Return observed peak contributed by entries GC will remove."""
        event_points = {
            start for start, finish, _, _, _, _ in self.active if finish <= watermark
        }
        return max((self._used_at(point) for point in event_points), default=0)

    def _refresh_peak(self) -> None:
        self.peak_used_concurrency_cost = max(
            self._historical_peak_used_concurrency_cost,
            self._peak_over_active(),
        )

    def used_concurrency_cost(self, now: float) -> int:
        """Return weighted capacity occupied at ``now``.

        ``now`` is required: after the move to a time-aware interval
        ledger there is no single "current used" value; the answer
        always depends on which point in time the caller is asking about.
        """
        if now is None:
            raise ValueError(
                "WeightedConcurrencyState.used_concurrency_cost requires `now`"
            )
        return self._used_at(now)

    def utilization(self, now: float) -> float:
        """Return weighted utilization at ``now`` as a value in [0, 1]."""
        if now is None:
            raise ValueError(
                "WeightedConcurrencyState.utilization requires `now`"
            )
        return min(self._used_at(now) / self.capacity_units, 1.0)

    def can_admit(self, model_class: str, now: float) -> bool:
        """Return whether a request with ``model_class`` can enter at ``now``."""
        cost = self.concurrency_cost(model_class)
        if cost is None:
            return False
        return self._used_at(now) + cost <= self.capacity_units

    def can_admit_interval(
        self,
        start: float,
        end: float,
        model_class: str | None = None,
    ) -> bool:
        """Return whether the fixed/scoped model can enter throughout ``[start, end)``."""
        resolved = self._model_class_for_interval(model_class)
        cost = self.concurrency_cost(resolved)
        if cost is None:
            return False
        if end <= start:
            return self.can_admit(resolved, start)
        event_points = {start}
        for s, _, _, _, _, _ in self.active:
            if start <= s < end:
                event_points.add(s)
        return all(self._used_at(p) + cost <= self.capacity_units for p in event_points)

    def admit(
        self,
        model_class: str,
        finish_time: float,
        *,
        now: float,
        request_id: int | None = None,
    ) -> bool:
        """Admit one request occupying ``[now, finish_time)``.

        Returns ``True`` when the request was recorded and ``False`` when the
        resolved model class is incompatible or the requested interval would
        exceed capacity at any point.
        """
        if finish_time <= now:
            raise ValueError("finish_time must be greater than now")
        cost = self.concurrency_cost(model_class)
        if cost is None:
            return False
        if not self.can_admit_interval(now, finish_time, model_class):
            return False
        sequence = self._sequence
        self.active.append(
            (
                float(now),
                float(finish_time),
                sequence,
                int(sequence if request_id is None else request_id),
                str(model_class),
                cost,
            )
        )
        self._sequence += 1
        self.total_capacity_unit_seconds_used += cost * (float(finish_time) - float(now))
        self._refresh_peak()
        return True

    def admit_interval(
        self,
        now: float,
        service_time_sec: float,
        model_class: str | None = None,
        request_id: int | None = None,
    ) -> bool:
        """Record one fixed/scoped-model request interval starting at ``now``."""
        return self.admit(
            self._model_class_for_interval(model_class),
            now + service_time_sec,
            now=now,
            request_id=request_id,
        )

    def truncate(self, request_id: int, end_time: float) -> bool:
        """Shorten one admitted request interval to end no later than ``end_time``."""
        changed = False
        active: list[tuple[float, float, int, int, str, int]] = []
        for start, finish, sequence, rid, model_class, cost in self.active:
            if rid != request_id or end_time >= finish:
                active.append((start, finish, sequence, rid, model_class, cost))
                continue
            changed = True
            new_finish = float(end_time)
            unused_seconds = finish - max(new_finish, start)
            self.total_capacity_unit_seconds_used -= cost * unused_seconds
            if end_time > start:
                active.append((start, new_finish, sequence, rid, model_class, cost))
        if changed:
            self.active = active
            self.total_capacity_unit_seconds_used = max(
                self.total_capacity_unit_seconds_used,
                0.0,
            )
            self._refresh_peak()
        return changed

    def gc_before(self, watermark: float) -> None:
        """Drop interval entries whose ``finish`` is at or before ``watermark``.

        Memory-only optimization. The caller (typically the simulator
        engine) must guarantee that no future read will query a time
        strictly less than ``watermark``; otherwise the dropped state
        would be needed for correctness.
        """
        self._historical_peak_used_concurrency_cost = max(
            self._historical_peak_used_concurrency_cost,
            self._peak_over_entries_removed_by_gc(watermark),
        )
        self.active = [entry for entry in self.active if entry[1] > watermark]
        self._refresh_peak()

    def reset(self) -> None:
        """Reset mutable weighted concurrency state."""
        self.active = []
        self.peak_used_concurrency_cost = 0
        self.total_capacity_unit_seconds_used = 0.0
        self._sequence = 0
        self._historical_peak_used_concurrency_cost = 0


__all__ = [
    "ConcurrencyState",
    "MultiWindowQuotaState",
    "ProviderTier",
    "QuotaState",
    "WeightedConcurrencyState",
]
