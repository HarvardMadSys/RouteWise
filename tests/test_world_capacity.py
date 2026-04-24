"""Tests for dependency-light world capacity primitives."""

from __future__ import annotations

import unittest

from rwsim.world.capacity import ConcurrencyState, QuotaState


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

    def test_concurrency_sweeps_completed_requests(self) -> None:
        concurrency = ConcurrencyState(limit=1)

        self.assertTrue(concurrency.can_admit(0.0))
        concurrency.admit(request_id=1, now=0.0, service_time_sec=5.0)
        self.assertFalse(concurrency.can_admit(1.0))
        self.assertEqual(concurrency.utilization(1.0), 1.0)

        self.assertTrue(concurrency.can_admit(5.0))
        self.assertEqual(concurrency.utilization(5.0), 0.0)


if __name__ == "__main__":
    unittest.main()
