"""QuotaState rolling-window regressions for the real-eval harness."""

from __future__ import annotations

from experiments.real_evaluation.inventory import QuotaState


def test_first_charge_snaps_to_fixed_boundary() -> None:
    """A QuotaState constructed with default window_start=0.0 should snap
    to the largest ``window_sec`` multiple <= ``now`` on first tick.

    The earlier ``window_start = now`` shortcut anchored to first-request
    time, which extended the window beyond its configured length
    whenever the first request arrived mid-window.
    """
    q = QuotaState(window_sec=3600.0, limit=10)
    # First contact at 5000s into the run. Fixed-boundary semantics
    # demand window_start = floor(5000 / 3600) * 3600 = 3600.
    q.charge(now=5000.0)
    assert q.used == 1
    assert q.window_start == 3600.0


def test_advances_by_full_window_multiples_under_sparse_traffic() -> None:
    """If 2.5 windows pass between charges, ``window_start`` advances by
    2 windows (not 2.5), preserving fixed boundaries."""
    q = QuotaState(window_sec=3600.0, limit=10)
    q.charge(now=1000.0)  # snaps to 0.0
    q.charge(now=1100.0)
    assert q.used == 2 and q.window_start == 0.0
    q.tick(now=1000.0 + 2.5 * 3600.0)
    assert q.used == 0
    assert q.window_start == 2 * 3600.0


def test_zero_window_resets_used_on_every_tick() -> None:
    """Defensive: a non-positive ``window_sec`` resets ``used`` on every
    ``tick``, so each ``charge()`` returns to ``used == 1`` immediately
    afterwards (the prior charge has been wiped). Pin the behaviour so
    a future change can't silently degrade it."""
    q = QuotaState(window_sec=0.0, limit=10)
    q.charge(now=42.0)
    assert q.used == 1
    # Independent later tick — previous charge is wiped, this one re-enters.
    q.charge(now=43.0)
    assert q.used == 1


def test_fraction_used_within_window() -> None:
    q = QuotaState(window_sec=60.0, limit=4)
    q.charge(now=120.0)
    q.charge(now=121.0)
    assert q.fraction_used(now=121.0) == 0.5


def test_can_admit_blocks_at_limit() -> None:
    q = QuotaState(window_sec=60.0, limit=2)
    q.charge(now=10.0)
    q.charge(now=11.0)
    assert q.is_available(now=12.0) is False
    # New window: counter resets, slot frees.
    assert q.is_available(now=80.0) is True
