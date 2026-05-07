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

    def test_concurrency_sweeps_completed_requests(self) -> None:
        concurrency = ConcurrencyState(limit=1)

        self.assertTrue(concurrency.can_admit(0.0))
        concurrency.admit(request_id=1, now=0.0, service_time_sec=5.0)
        self.assertFalse(concurrency.can_admit(1.0))
        self.assertEqual(concurrency.utilization(1.0), 1.0)

        self.assertTrue(concurrency.can_admit(5.0))
        self.assertEqual(concurrency.utilization(5.0), 0.0)

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
        self.assertEqual(concurrency._active_count_at(12.0), 1)

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

    def test_weighted_concurrency_releases_finished_requests(self) -> None:
        concurrency = WeightedConcurrencyState(
            capacity_units=4,
            model_concurrency_costs_by_class={"ge_70b": 4},
        )

        self.assertTrue(concurrency.admit("ge_70b", finish_time=5.0, now=0.0))
        self.assertFalse(concurrency.can_admit("ge_70b", now=4.0))
        self.assertTrue(concurrency.can_admit("ge_70b", now=5.0))
        self.assertEqual(concurrency.used_concurrency_cost(), 0)

    def test_weighted_concurrency_rejects_unknown_model_class(self) -> None:
        concurrency = WeightedConcurrencyState(
            capacity_units=4,
            model_concurrency_costs_by_class={"ge_70b": 4},
        )

        self.assertFalse(concurrency.can_admit("unknown", now=0.0))
        self.assertFalse(concurrency.admit("unknown", finish_time=5.0, now=0.0))
        self.assertEqual(concurrency.used_concurrency_cost(), 0)

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

    def test_weighted_concurrency_heap_release_semantics(self) -> None:
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
        self.assertEqual(concurrency.active, [])

    def test_weighted_concurrency_running_used_cost_and_utilization(self) -> None:
        concurrency = WeightedConcurrencyState(
            capacity_units=8,
            model_concurrency_costs_by_class={"small": 1, "medium": 2},
        )

        self.assertTrue(concurrency.admit("small", finish_time=10.0, now=0.0))
        self.assertEqual(concurrency.used_concurrency_cost(), 1)
        self.assertEqual(concurrency.utilization(), 0.125)

        self.assertTrue(concurrency.admit("medium", finish_time=20.0, now=0.0))
        self.assertEqual(concurrency.used_concurrency_cost(), 3)
        self.assertEqual(concurrency.utilization(), 0.375)

        self.assertEqual(concurrency.used_concurrency_cost(10.0), 2)
        self.assertEqual(concurrency.utilization(10.0), 0.25)
        self.assertEqual(concurrency.used_concurrency_cost(20.0), 0)

    def test_weighted_concurrency_rejects_when_capacity_full_after_heap_change(self) -> None:
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

    def test_weighted_concurrency_reset_clears_heap_and_counters(self) -> None:
        concurrency = WeightedConcurrencyState(
            capacity_units=8,
            model_concurrency_costs_by_class={"small": 1, "large": 4},
        )

        self.assertTrue(concurrency.admit("large", finish_time=10.0, now=0.0))
        self.assertTrue(concurrency.admit("small", finish_time=5.0, now=0.0))

        concurrency.reset()

        self.assertEqual(concurrency.active, [])
        self.assertEqual(concurrency.used_concurrency_cost(), 0)
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

    def test_weighted_concurrency_heap_matches_list_reference_randomized(self) -> None:
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
            now += rng.random() * 2.0
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

            self.assertEqual(impl.used_concurrency_cost(), reference.used_concurrency_cost())
            self.assertAlmostEqual(impl.utilization(), reference.utilization())


class _WeightedConcurrencyReference:
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

    def used_concurrency_cost(self, now: float | None = None) -> int:
        if now is not None:
            self.release_finished(now)
        return sum(cost for _, _, cost in self.active)

    def utilization(self, now: float | None = None) -> float:
        return min(self.used_concurrency_cost(now) / self.capacity_units, 1.0)

    def can_admit(self, model_class: str, now: float | None = None) -> bool:
        cost = self.costs.get(model_class)
        if cost is None:
            return False
        return self.used_concurrency_cost(now) + cost <= self.capacity_units

    def admit(self, model_class: str, *, finish_time: float, now: float | None = None) -> bool:
        cost = self.costs.get(model_class)
        if cost is None:
            return False
        if self.used_concurrency_cost(now) + cost > self.capacity_units:
            return False
        self.active.append((finish_time, model_class, cost))
        return True


if __name__ == "__main__":
    unittest.main()
