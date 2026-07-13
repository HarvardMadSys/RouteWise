"""Tests for the private API-provider capacity seam."""

from __future__ import annotations

import threading

import pytest

from routewise._capacity_controller import (
    _CapacityController,
    _CapacitySnapshot,
    _NoopCapacityController,
    _Reservation,
)


def test_noop_controller_implements_private_protocols() -> None:
    controller = _NoopCapacityController()
    snapshot = controller.snapshot(resource_key="provider-a", now=12.5)
    reservation = controller.try_reserve(
        resource_key="provider-a", attempt_id="attempt-1", snapshot=snapshot
    )

    assert isinstance(controller, _CapacityController)
    assert isinstance(reservation, _Reservation)
    assert snapshot == _CapacitySnapshot(resource_key="provider-a", observed_at=12.5)


def test_reserve_commit_and_terminal_release_are_idempotent() -> None:
    controller = _NoopCapacityController()
    snapshot = controller.snapshot(resource_key="provider-a", now=0.0)
    reservation = controller.try_reserve(
        resource_key="provider-a", attempt_id="attempt-1", snapshot=snapshot
    )
    assert reservation is not None

    assert reservation.commit() is True
    assert reservation.commit() is True
    assert reservation.committed

    reservation.release()
    reservation.release()

    assert reservation.closed
    assert not reservation.committed
    assert reservation.commit() is False


def test_release_before_dispatch_closes_uncommitted_reservation() -> None:
    controller = _NoopCapacityController()
    snapshot = controller.snapshot(resource_key="provider-a", now=0.0)
    reservation = controller.try_reserve(
        resource_key="provider-a", attempt_id="attempt-1", snapshot=snapshot
    )
    assert reservation is not None

    reservation.release()

    assert reservation.closed
    assert reservation.commit() is False


def test_unavailable_snapshot_declines_reservation() -> None:
    controller = _NoopCapacityController()
    snapshot = _CapacitySnapshot(resource_key="provider-a", observed_at=0.0, available=False)

    assert (
        controller.try_reserve(resource_key="provider-a", attempt_id="attempt-1", snapshot=snapshot)
        is None
    )


def test_snapshot_cannot_be_used_for_another_resource() -> None:
    controller = _NoopCapacityController()
    snapshot = controller.snapshot(resource_key="provider-a", now=0.0)

    with pytest.raises(ValueError, match="resource_key"):
        controller.try_reserve(resource_key="provider-b", attempt_id="attempt-1", snapshot=snapshot)


def test_concurrent_commit_and_release_reach_a_valid_terminal_state() -> None:
    controller = _NoopCapacityController()
    snapshot = controller.snapshot(resource_key="provider-a", now=0.0)
    reservation = controller.try_reserve(
        resource_key="provider-a", attempt_id="attempt-1", snapshot=snapshot
    )
    assert reservation is not None

    barrier = threading.Barrier(3)

    def commit() -> None:
        barrier.wait()
        reservation.commit()

    def release() -> None:
        barrier.wait()
        reservation.release()

    threads = [threading.Thread(target=commit), threading.Thread(target=release)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert reservation.closed
    assert reservation.commit() is False
