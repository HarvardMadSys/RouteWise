"""Tests for dependency-light world capacity primitives."""

from __future__ import annotations

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
        self.assertFalse(concurrency.can_admit("le_15b", now=1.0))
        self.assertFalse(concurrency.admit("le_15b", finish_time=10.0, now=1.0))

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


if __name__ == "__main__":
    unittest.main()
