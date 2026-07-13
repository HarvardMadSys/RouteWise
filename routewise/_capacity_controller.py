"""Private capacity-admission seam used by the API-only facade."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class _CapacitySnapshot:
    """Immutable observation of one atomic capacity domain."""

    resource_key: str
    observed_at: float
    available: bool = True
    generation: int = 0


@runtime_checkable
class _Reservation(Protocol):
    """Atomic ownership of capacity for one attempt."""

    def commit(self) -> bool:
        """Commit immediately before I/O, returning whether ownership remains."""
        ...

    def release(self) -> None:
        """Close the reservation, returning capacity when appropriate."""
        ...


@runtime_checkable
class _CapacityController(Protocol):
    """Private admission interface implemented by each capacity backend."""

    def snapshot(self, *, resource_key: str, now: float) -> _CapacitySnapshot:
        """Return the current admission snapshot for one resource key."""
        ...

    def try_reserve(
        self,
        *,
        resource_key: str,
        attempt_id: str,
        snapshot: _CapacitySnapshot,
    ) -> _Reservation | None:
        """Atomically reserve against ``snapshot``, or report contention."""
        ...


class _ReservationState(str, Enum):
    RESERVED = "reserved"
    COMMITTED = "committed"
    CLOSED = "closed"


class _NoopReservation:
    """In-process reservation whose API-provider capacity is unlimited."""

    def __init__(self, *, resource_key: str, attempt_id: str) -> None:
        if not resource_key:
            raise ValueError("resource_key must be non-empty")
        if not attempt_id:
            raise ValueError("attempt_id must be non-empty")
        self.resource_key = resource_key
        self.attempt_id = attempt_id
        self._state = _ReservationState.RESERVED
        self._lock = Lock()

    def commit(self) -> bool:
        """Commit once; repeated commits succeed until the reservation closes."""
        with self._lock:
            if self._state is _ReservationState.CLOSED:
                return False
            self._state = _ReservationState.COMMITTED
            return True

    def release(self) -> None:
        """Close this reservation idempotently."""
        with self._lock:
            self._state = _ReservationState.CLOSED

    @property
    def committed(self) -> bool:
        """Return whether capacity was committed and has not yet closed."""
        with self._lock:
            return self._state is _ReservationState.COMMITTED

    @property
    def closed(self) -> bool:
        """Return whether this reservation has reached its terminal state."""
        with self._lock:
            return self._state is _ReservationState.CLOSED


class _NoopCapacityController:
    """Capacity controller for API providers; every valid reserve succeeds."""

    def snapshot(self, *, resource_key: str, now: float) -> _CapacitySnapshot:
        if not resource_key:
            raise ValueError("resource_key must be non-empty")
        if not math.isfinite(now):
            raise ValueError("now must be finite")
        return _CapacitySnapshot(resource_key=resource_key, observed_at=float(now))

    def try_reserve(
        self,
        *,
        resource_key: str,
        attempt_id: str,
        snapshot: _CapacitySnapshot,
    ) -> _NoopReservation | None:
        if not resource_key:
            raise ValueError("resource_key must be non-empty")
        if not attempt_id:
            raise ValueError("attempt_id must be non-empty")
        if snapshot.resource_key != resource_key:
            raise ValueError("snapshot resource_key does not match the reservation target")
        if not snapshot.available:
            return None
        return _NoopReservation(resource_key=resource_key, attempt_id=attempt_id)


__all__ = [
    "_CapacityController",
    "_CapacitySnapshot",
    "_NoopCapacityController",
    "_NoopReservation",
    "_Reservation",
]
