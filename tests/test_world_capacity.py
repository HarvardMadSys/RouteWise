"""Tests for dependency-light world capacity primitives."""

from __future__ import annotations

import random
import unittest

from rwsim.world.capacity import ConcurrencyState, QuotaState, WeightedConcurrencyState


class CapacityStateTest(unittest.TestCase):
    def test_quota_rolls_after_window(self) -> None:
        quota = QuotaState(size=2, window_sec=10.0)

        self.assertTrue(quota.can_admit(0.0))
        quota.charge(0.0)
        quota.charge(1.0)
        self.assertFalse(quota.can_admit(2.0))
        self.assertEqual(quota.fraction_used(2.0), 1.0)

        self.assertTrue(quota.can_admit(10.0))
        self.assertEqual(quota.used, 0)
        self.assertEqual(quota.window_start, 10.0)

    def test_quota_rolls_on_fixed_window_boundaries(self) -> None:
        quota = QuotaState(size=1, window_sec=10.0)

        quota.charge(0.0)
        self.assertFalse(quota.can_admit(9.0))

        self.assertTrue(quota.can_admit(13.0))
        self.assertEqual(quota.window_start, 10.0)
        quota.charge(13.0)

        self.assertTrue(quota.can_admit(21.0))
        self.assertEqual(quota.window_start, 20.0)

    def test_concurrency_pure_reads_at_different_times(self) -> None:
        """Reads at different ``now`` values return the right answer
        without mutating ledger state (no destructive sweep)."""
        concurrency = ConcurrencyState(limit=1)

        self.assertTrue(concurrency.can_admit(0.0))
        concurrency.admit(request_id=1, now=0.0, service_time_sec=5.0)
        self.assertFalse(concurrency.can_admit(1.0))
        self.assertEqual(concurrency.utilization(1.0), 1.0)

        self.assertTrue(concurrency.can_admit(5.0))
        self.assertEqual(concurrency.utilization(5.0), 0.0)

        # Ledger entry is still present after reads; only gc_before removes it.
        self.assertEqual(len(concurrency.active), 1)

    def test_future_concurrency_interval_does_not_occupy_before_start(self) -> None:
        concurrency = ConcurrencyState(limit=1)

        concurrency.admit(request_id=1, now=5.0, service_time_sec=5.0)

        self.assertTrue(concurrency.can_admit(4.0))
        self.assertEqual(concurrency.utilization(4.0), 0.0)
        self.assertFalse(concurrency.can_admit(5.0))
        self.assertEqual(concurrency.utilization(5.0), 1.0)
        self.assertFalse(concurrency.can_admit(9.0))
        self.assertTrue(concurrency.can_admit(10.0))

    def test_concurrency_interval_admission_checks_overlap(self) -> None:
        concurrency = ConcurrencyState(limit=1)

        concurrency.admit(request_id=1, now=5.0, service_time_sec=5.0)

        self.assertTrue(concurrency.can_admit_interval(0.0, 4.0))
        self.assertFalse(concurrency.can_admit_interval(4.0, 6.0))
        self.assertFalse(concurrency.can_admit_interval(6.0, 8.0))
        self.assertTrue(concurrency.can_admit_interval(10.0, 12.0))

    def test_concurrency_admit_rejects_overlapping_future_reservation(self) -> None:
        concurrency = ConcurrencyState(limit=1)

        concurrency.admit(request_id=1, now=10.0, service_time_sec=10.0)

        self.assertTrue(concurrency.can_admit(5.0))
        with self.assertRaises(RuntimeError):
            concurrency.admit(request_id=2, now=5.0, service_time_sec=10.0)
        self.assertEqual(concurrency._count_at(12.0), 1)

    def test_concurrency_state_pure_under_time_travel(self) -> None:
        """Regression: simulator hedge tick can query a future ``now`` then
        revert to outer trace time. Reads must not destroy ledger state
        in a way that breaks later queries at earlier times."""
        concurrency = ConcurrencyState(limit=1)
        concurrency.admit(request_id=1, now=0.0, service_time_sec=0.3)

        self.assertFalse(concurrency.can_admit(0.1))
        # Query a future time (this used to destructively sweep [0, 0.3)).
        self.assertTrue(concurrency.can_admit(0.5))
        # Query past time again; must still see the original interval.
        self.assertFalse(concurrency.can_admit(0.1))

    def test_concurrency_gc_before_drops_finished_intervals(self) -> None:
        concurrency = ConcurrencyState(limit=2)
        concurrency.admit(request_id=1, now=0.0, service_time_sec=2.0)  # [0, 2)
        concurrency.admit(request_id=2, now=3.0, service_time_sec=4.0)  # [3, 7)

        self.assertEqual(len(concurrency.active), 2)
        concurrency.gc_before(2.5)  # keep entries whose end > 2.5
        self.assertEqual(len(concurrency.active), 1)
        self.assertEqual(concurrency.active[0][2], 2)

    def test_weighted_concurrency_cost_four_fills_capacity(self) -> None:
        concurrency = WeightedConcurrencyState(
            capacity_units=4,
            model_concurrency_costs_by_class={"ge_70b": 4, "le_15b": 1},
        )

        self.assertTrue(concurrency.admit("ge_70b", finish_time=10.0, now=0.0))
        self.assertEqual(concurrency.used_concurrency_cost(1.0), 4)
        self.assertEqual(concurrency.utilization(1.0), 1.0)
        self.assertEqual(concurrency.total_capacity_unit_seconds_used, 40.0)
        self.assertFalse(concurrency.can_admit("le_15b", now=1.0))
        self.assertFalse(concurrency.admit("le_15b", finish_time=10.0, now=1.0))
        self.assertEqual(concurrency.total_capacity_unit_seconds_used, 40.0)

    def test_weighted_concurrency_cost_one_admits_four_requests(self) -> None:
        concurrency = WeightedConcurrencyState(
            capacity_units=4,
            model_concurrency_costs_by_class={"le_15b": 1},
        )

        for request_id in range(4):
            self.assertTrue(concurrency.admit("le_15b", finish_time=10.0, now=0.0))
            self.assertEqual(concurrency.used_concurrency_cost(0.0), request_id + 1)

        self.assertFalse(concurrency.can_admit("le_15b", now=0.0))
        self.assertFalse(concurrency.admit("le_15b", finish_time=10.0, now=0.0))
        self.assertEqual(concurrency.peak_used_concurrency_cost, 4)

    def test_weighted_concurrency_releases_at_finish_time(self) -> None:
        concurrency = WeightedConcurrencyState(
            capacity_units=4,
            model_concurrency_costs_by_class={"ge_70b": 4},
        )

        self.assertTrue(concurrency.admit("ge_70b", finish_time=5.0, now=0.0))
        self.assertFalse(concurrency.can_admit("ge_70b", now=4.0))
        self.assertTrue(concurrency.can_admit("ge_70b", now=5.0))
        # At t=5 the interval [0, 5) is no longer occupying capacity.
        self.assertEqual(concurrency.used_concurrency_cost(5.0), 0)

    def test_weighted_concurrency_rejects_unknown_model_class(self) -> None:
        concurrency = WeightedConcurrencyState(
            capacity_units=4,
            model_concurrency_costs_by_class={"ge_70b": 4},
        )

        self.assertFalse(concurrency.can_admit("unknown", now=0.0))
        self.assertFalse(concurrency.admit("unknown", finish_time=5.0, now=0.0))
        self.assertEqual(concurrency.used_concurrency_cost(0.0), 0)

    def test_weighted_concurrency_fixed_model_interval_compatibility(self) -> None:
        concurrency = WeightedConcurrencyState(
            capacity_units=8,
            model_concurrency_costs_by_class={"ge_70b": 4, "24_34b": 2},
            fixed_model_class="ge_70b",
        )

        self.assertEqual(concurrency.limit, 2)
        self.assertTrue(concurrency.can_admit_interval(0.0, 5.0))
        self.assertTrue(concurrency.admit_interval(now=0.0, service_time_sec=5.0))
        self.assertTrue(concurrency.admit_interval(now=0.0, service_time_sec=5.0))
        self.assertEqual(concurrency.total_capacity_unit_seconds_used, 40.0)
        self.assertFalse(concurrency.can_admit_interval(1.0, 6.0))
        self.assertTrue(concurrency.can_admit_interval(5.0, 6.0))

    def test_weighted_concurrency_used_decreases_over_time(self) -> None:
        concurrency = WeightedConcurrencyState(
            capacity_units=8,
            model_concurrency_costs_by_class={"small": 1},
        )

        self.assertTrue(concurrency.admit("small", finish_time=10.0, now=0.0))
        self.assertTrue(concurrency.admit("small", finish_time=20.0, now=0.0))
        self.assertTrue(concurrency.admit("small", finish_time=30.0, now=0.0))

        self.assertEqual(concurrency.used_concurrency_cost(0.0), 3)
        self.assertEqual(concurrency.used_concurrency_cost(10.0), 2)
        self.assertEqual(concurrency.used_concurrency_cost(20.0), 1)
        self.assertEqual(concurrency.used_concurrency_cost(30.0), 0)

        # Ledger entries still present until gc_before is called.
        self.assertEqual(len(concurrency.active), 3)
        concurrency.gc_before(30.0)
        self.assertEqual(concurrency.active, [])

    def test_weighted_concurrency_running_used_cost_and_utilization(self) -> None:
        concurrency = WeightedConcurrencyState(
            capacity_units=8,
            model_concurrency_costs_by_class={"small": 1, "medium": 2},
        )

        self.assertTrue(concurrency.admit("small", finish_time=10.0, now=0.0))
        self.assertEqual(concurrency.used_concurrency_cost(0.0), 1)
        self.assertEqual(concurrency.utilization(0.0), 0.125)

        self.assertTrue(concurrency.admit("medium", finish_time=20.0, now=0.0))
        self.assertEqual(concurrency.used_concurrency_cost(0.0), 3)
        self.assertEqual(concurrency.utilization(0.0), 0.375)

        self.assertEqual(concurrency.used_concurrency_cost(10.0), 2)
        self.assertEqual(concurrency.utilization(10.0), 0.25)
        self.assertEqual(concurrency.used_concurrency_cost(20.0), 0)

    def test_weighted_concurrency_rejects_when_capacity_full_after_finish(self) -> None:
        concurrency = WeightedConcurrencyState(
            capacity_units=4,
            model_concurrency_costs_by_class={"large": 4},
        )

        self.assertTrue(concurrency.admit("large", finish_time=10.0, now=0.0))
        self.assertFalse(concurrency.can_admit("large", now=9.0))
        self.assertFalse(concurrency.admit("large", finish_time=12.0, now=9.0))
        self.assertTrue(concurrency.can_admit("large", now=10.0))
        self.assertTrue(concurrency.admit("large", finish_time=20.0, now=10.0))

    def test_weighted_concurrency_peak_and_capacity_seconds_preserved(self) -> None:
        concurrency = WeightedConcurrencyState(
            capacity_units=8,
            model_concurrency_costs_by_class={"small": 1, "large": 4},
        )

        self.assertTrue(concurrency.admit("large", finish_time=10.0, now=0.0))
        self.assertTrue(concurrency.admit("small", finish_time=5.0, now=0.0))
        self.assertEqual(concurrency.peak_used_concurrency_cost, 5)
        self.assertEqual(concurrency.total_capacity_unit_seconds_used, 45.0)

        self.assertEqual(concurrency.used_concurrency_cost(5.0), 4)
        self.assertEqual(concurrency.peak_used_concurrency_cost, 5)
        self.assertEqual(concurrency.total_capacity_unit_seconds_used, 45.0)

    def test_weighted_concurrency_reset_clears_ledger_and_counters(self) -> None:
        concurrency = WeightedConcurrencyState(
            capacity_units=8,
            model_concurrency_costs_by_class={"small": 1, "large": 4},
        )

        self.assertTrue(concurrency.admit("large", finish_time=10.0, now=0.0))
        self.assertTrue(concurrency.admit("small", finish_time=5.0, now=0.0))

        concurrency.reset()

        self.assertEqual(concurrency.active, [])
        self.assertEqual(concurrency.used_concurrency_cost(0.0), 0)
        self.assertEqual(concurrency.peak_used_concurrency_cost, 0)
        self.assertEqual(concurrency.total_capacity_unit_seconds_used, 0.0)

    def test_weighted_concurrency_interval_helpers_use_start_time(self) -> None:
        concurrency = WeightedConcurrencyState(
            capacity_units=8,
            model_concurrency_costs_by_class={"ge_70b": 4, "small": 1},
            fixed_model_class="ge_70b",
        )

        self.assertTrue(concurrency.admit_interval(now=0.0, service_time_sec=10.0))
        self.assertTrue(concurrency.admit_interval(now=0.0, service_time_sec=10.0))
        self.assertFalse(concurrency.can_admit_interval(9.0, 100.0))
        self.assertTrue(concurrency.can_admit_interval(10.0, 100.0))

        self.assertTrue(concurrency.admit_interval(now=10.0, service_time_sec=5.0))
        self.assertEqual(concurrency.used_concurrency_cost(10.0), 4)
        self.assertEqual(concurrency.total_capacity_unit_seconds_used, 100.0)

    def test_weighted_concurrency_future_start_does_not_pre_occupy(self) -> None:
        """Regression: a future-start interval (e.g. a hedge backup
        planned at a future checkpoint) must not occupy capacity before
        its start time. The previous _current_used_cost counter took
        effect on admit, which made future intervals look "already
        occupied" to queries at earlier times."""
        concurrency = WeightedConcurrencyState(
            capacity_units=4,
            model_concurrency_costs_by_class={"m": 4},
            fixed_model_class="m",
        )

        concurrency.admit_interval(now=10.0, service_time_sec=5.0)

        # Before the interval starts, no capacity is used.
        self.assertEqual(concurrency.used_concurrency_cost(0.0), 0)
        self.assertEqual(concurrency.utilization(0.0), 0.0)
        self.assertTrue(concurrency.can_admit_interval(0.0, 1.0))

        # Inside the interval, capacity is fully occupied.
        self.assertEqual(concurrency.used_concurrency_cost(12.0), 4)
        self.assertFalse(concurrency.can_admit_interval(11.0, 13.0))

        # After it ends, capacity is free again.
        self.assertEqual(concurrency.used_concurrency_cost(15.0), 0)

    def test_weighted_concurrency_peak_under_overlap(self) -> None:
        """Regression: peak occupied cost is the maximum over event
        points, not a running counter that double-counts non-overlapping
        intervals."""
        concurrency = WeightedConcurrencyState(
            capacity_units=12,
            model_concurrency_costs_by_class={"m": 4},
            fixed_model_class="m",
        )
        concurrency.admit_interval(now=0.0, service_time_sec=5.0)  # [0, 5)
        concurrency.admit_interval(now=3.0, service_time_sec=5.0)  # [3, 8)
        concurrency.admit_interval(now=6.0, service_time_sec=4.0)  # [6, 10)

        # Two intervals overlap at most: [3,5) and [6,8). Peak = 8.
        self.assertEqual(concurrency.peak_used_concurrency_cost, 8)
        self.assertEqual(concurrency.used_concurrency_cost(4.0), 8)
        self.assertEqual(concurrency.used_concurrency_cost(7.0), 8)
        # At t=5.5 only the second interval is active.
        self.assertEqual(concurrency.used_concurrency_cost(5.5), 4)

    def test_weighted_concurrency_requires_now_on_reads(self) -> None:
        """No-arg reads were a remnant of the old counter API. After the
        move to a time-aware interval ledger, ``now`` is required so the
        caller cannot accidentally read a meaningless 'current' value."""
        concurrency = WeightedConcurrencyState(
            capacity_units=4,
            model_concurrency_costs_by_class={"m": 1},
        )

        with self.assertRaises(ValueError):
            concurrency.used_concurrency_cost(None)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            concurrency.utilization(None)  # type: ignore[arg-type]

    def test_weighted_concurrency_rejects_legacy_active_shape(self) -> None:
        """Pre-existing 3/4-tuple active entries had no start_time and
        cannot be safely upgraded; the constructor rejects them."""
        with self.assertRaises(ValueError):
            WeightedConcurrencyState(
                capacity_units=4,
                model_concurrency_costs_by_class={"m": 1},
                active=[(10.0, "m", 1)],  # legacy 3-tuple shape  # type: ignore[list-item]
            )
        with self.assertRaises(ValueError):
            WeightedConcurrencyState(
                capacity_units=4,
                model_concurrency_costs_by_class={"m": 1},
                active=[(10.0, 0, "m", 1)],  # legacy 4-tuple shape  # type: ignore[list-item]
            )

    def test_weighted_concurrency_gc_before_drops_finished_intervals(self) -> None:
        concurrency = WeightedConcurrencyState(
            capacity_units=8,
            model_concurrency_costs_by_class={"m": 1},
        )
        concurrency.admit("m", finish_time=10.0, now=0.0)
        concurrency.admit("m", finish_time=20.0, now=5.0)

        self.assertEqual(len(concurrency.active), 2)
        concurrency.gc_before(15.0)  # keep entries whose finish > 15.0
        self.assertEqual(len(concurrency.active), 1)
        self.assertEqual(concurrency.active[0][1], 20.0)

    def test_weighted_concurrency_matches_reference_under_monotonic_time(
        self,
    ) -> None:
        """Under monotonically-advancing ``now``, the interval-ledger
        implementation must give the same answers as a destructive-
        counter reference. Non-monotonic behavior is covered by the
        dedicated future-start and time-travel regressions above."""
        rng = random.Random(7)
        impl = WeightedConcurrencyState(
            capacity_units=8,
            model_concurrency_costs_by_class={"small": 1, "medium": 2, "large": 4},
        )
        reference = _WeightedConcurrencyReference(
            capacity_units=8,
            costs={"small": 1, "medium": 2, "large": 4},
        )
        now = 0.0

        for _ in range(250):
            now += rng.random() * 2.0  # strictly increasing
            model_class = rng.choice(["small", "medium", "large", "unknown"])
            operation = rng.choice(["can_admit", "admit", "used", "utilization"])

            if operation == "can_admit":
                self.assertEqual(
                    impl.can_admit(model_class, now=now),
                    reference.can_admit(model_class, now=now),
                )
            elif operation == "admit":
                finish_time = now + 0.1 + rng.random() * 6.0
                self.assertEqual(
                    impl.admit(model_class, finish_time=finish_time, now=now),
                    reference.admit(model_class, finish_time=finish_time, now=now),
                )
            elif operation == "used":
                self.assertEqual(
                    impl.used_concurrency_cost(now),
                    reference.used_concurrency_cost(now),
                )
            else:
                self.assertAlmostEqual(
                    impl.utilization(now),
                    reference.utilization(now),
                )


class _WeightedConcurrencyReference:
    """Destructive-counter reference used by the monotonic-time parity test.

    Only valid when callers query ``now`` monotonically; this matches the
    pre-refactor implementation's hidden assumption.
    """

    def __init__(self, *, capacity_units: int, costs: dict[str, int]) -> None:
        self.capacity_units = capacity_units
        self.costs = costs
        self.active: list[tuple[float, str, int]] = []

    def release_finished(self, now: float) -> None:
        self.active = [
            (finish_time, model_class, cost)
            for finish_time, model_class, cost in self.active
            if finish_time > now
        ]

    def used_concurrency_cost(self, now: float) -> int:
        self.release_finished(now)
        return sum(cost for _, _, cost in self.active)

    def utilization(self, now: float) -> float:
        return min(self.used_concurrency_cost(now) / self.capacity_units, 1.0)

    def can_admit(self, model_class: str, now: float) -> bool:
        cost = self.costs.get(model_class)
        if cost is None:
            return False
        return self.used_concurrency_cost(now) + cost <= self.capacity_units

    def admit(self, model_class: str, *, finish_time: float, now: float) -> bool:
        cost = self.costs.get(model_class)
        if cost is None:
            return False
        if self.used_concurrency_cost(now) + cost > self.capacity_units:
            return False
        self.active.append((finish_time, model_class, cost))
        return True


if __name__ == "__main__":
    unittest.main()
