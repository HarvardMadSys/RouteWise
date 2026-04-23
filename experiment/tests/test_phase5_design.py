#!/usr/bin/env python3
"""Test Phase 5 evaluation design.

Tests:
1. Single window design (no dual window/prior)
2. Shared router across policies
3. All policies per request (not round-robin)
"""

import os
import sys
import threading
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import using same pattern as phase5_online_evaluation.py
import importlib.util

_script_dir = os.path.dirname(os.path.abspath(__file__))
_module_path = os.path.join(_script_dir, "..", "strategies", "online_latency_router.py")
_module_path = os.path.normpath(_module_path)

_spec = importlib.util.spec_from_file_location("online_latency_router", _module_path)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

OnlineLatencyRouter = _module.OnlineLatencyRouter
ProviderProfile = _module.ProviderProfile

_hedging_path = os.path.join(_script_dir, "..", "strategies", "smart_hedging.py")
_hedging_path = os.path.normpath(_hedging_path)

_hedging_spec = importlib.util.spec_from_file_location("smart_hedging", _hedging_path)
_hedging_module = importlib.util.module_from_spec(_hedging_spec)
sys.modules[_hedging_spec.name] = _hedging_module
_hedging_spec.loader.exec_module(_hedging_module)

SmartHedger = _hedging_module.SmartHedger
HedgingParams = _hedging_module.HedgingParams
HedgingStrategy = _hedging_module.HedgingStrategy
get_survival_for_hedging = _hedging_module.get_survival_for_hedging


class TestSingleWindowDesign:
    """Test single moving window design."""

    def test_provider_profile_single_window(self):
        """Provider profile should have single window, not dual."""
        profile = ProviderProfile(provider="Test", window_sec=60)

        # Should have single window
        assert hasattr(profile, "samples"), "Should have 'samples' field"
        assert hasattr(profile, "window_sec"), "Should have 'window_sec' field"

        # Should NOT have dual window fields
        assert not hasattr(
            profile, "short_window_samples"
        ), "Should not have 'short_window_samples'"
        assert not hasattr(profile, "long_window_samples"), "Should not have 'long_window_samples'"
        assert not hasattr(profile, "short_window_sec"), "Should not have 'short_window_sec'"
        assert not hasattr(profile, "long_window_sec"), "Should not have 'long_window_sec'"
        assert not hasattr(profile, "prior_strength"), "Should not have 'prior_strength'"
        assert not hasattr(profile, "prior_samples_sec"), "Should not have 'prior_samples_sec'"

        print("  PASSED: Single window design verified")

    def test_window_pruning(self):
        """Samples outside window should be pruned."""
        profile = ProviderProfile(provider="Test", window_sec=60)
        now = time.time()

        # Add samples: some inside window, some outside
        profile.add_sample(now - 30, 500)  # 30s ago, inside
        profile.add_sample(now - 50, 600)  # 50s ago, inside
        profile.add_sample(now - 100, 700)  # 100s ago, outside

        # Get samples (triggers pruning)
        samples = profile.get_samples_before(now)

        # Should only have 2 samples (inside window)
        assert len(samples) == 2, f"Expected 2 samples, got {len(samples)}"
        print("  PASSED: Window pruning works")

    def test_cdf_computation(self):
        """CDF should be computed from single window."""
        profile = ProviderProfile(provider="Test", window_sec=60)
        now = time.time()

        # Add samples: 500, 600, 700, 800, 900 ms
        for i, ttft in enumerate([500, 600, 700, 800, 900]):
            profile.add_sample(now - 30 + i, ttft)

        # CDF at 750ms should be 0.6 (3 out of 5 samples <= 750ms)
        cdf = profile.get_cdf_at(0.75, now)
        assert 0.55 < cdf < 0.65, f"Expected CDF ~0.6, got {cdf}"
        print("  PASSED: CDF computation works")


class TestSharedRouter:
    """Test that router can be shared across policies."""

    def test_router_shared_profiles(self):
        """Router profiles should be accessible and modifiable."""
        costs = {"ProviderA": 0.001, "ProviderB": 0.002}
        router = OnlineLatencyRouter(costs=costs, slo_sec=2.0, window_sec=60)

        now = time.time()

        # Add samples from "different policies" (same router)
        router.add_sample("ProviderA", now - 10, 400)  # From lp_mix
        router.add_sample("ProviderA", now - 5, 450)  # From smart_hedge
        router.add_sample("ProviderB", now - 8, 600)  # From openrouter_auto

        # All samples should be in the same profiles
        samples_a = router.profiles["ProviderA"].get_samples_before(now)
        samples_b = router.profiles["ProviderB"].get_samples_before(now)

        assert len(samples_a) == 2, f"ProviderA should have 2 samples, got {len(samples_a)}"
        assert len(samples_b) == 1, f"ProviderB should have 1 sample, got {len(samples_b)}"
        print("  PASSED: Shared router profiles work")

    def test_concurrent_router_access(self):
        """Router should handle concurrent access from multiple threads."""
        costs = {"ProviderA": 0.001, "ProviderB": 0.002, "ProviderC": 0.003}
        router = OnlineLatencyRouter(costs=costs, slo_sec=2.0, window_sec=60)

        now = time.time()
        errors = []

        def add_samples(provider: str, count: int):
            try:
                for i in range(count):
                    router.add_sample(provider, now + i * 0.001, 400 + i * 10)
            except Exception as e:
                errors.append(e)

        # Simulate 3 policies adding samples concurrently
        threads = [
            threading.Thread(target=add_samples, args=("ProviderA", 10)),
            threading.Thread(target=add_samples, args=("ProviderB", 10)),
            threading.Thread(target=add_samples, args=("ProviderC", 10)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent access errors: {errors}"

        # Verify all samples were added
        total_samples = sum(len(router.profiles[p].samples) for p in costs)
        assert total_samples == 30, f"Expected 30 samples, got {total_samples}"
        print("  PASSED: Concurrent router access works")


class TestHedgingLogic:
    """Test smart hedging logic."""

    def test_hedge_triggers_on_slow_primary(self):
        """Hedge should trigger when primary is slow."""
        costs = {"Fast": 0.002, "Slow": 0.001}
        router = OnlineLatencyRouter(costs=costs, slo_sec=2.0, window_sec=60)

        now = time.time()

        # Fast provider: 200-300ms
        for i in range(10):
            router.add_sample("Fast", now - 30 + i, 200 + i * 10)

        # Slow provider: 1500-2500ms (near SLO)
        for i in range(10):
            router.add_sample("Slow", now - 30 + i, 1500 + i * 100)

        # Compute hedge time: should be finite (hedge makes sense)
        params = HedgingParams(strategy=HedgingStrategy.SMART_SURVIVAL, slo_sec=2.0)
        hedger = SmartHedger(params, costs)

        h = hedger.compute_hedge_time("Slow", "Fast", router.profiles, now)
        assert h < float("inf"), f"Hedge time should be finite, got {h}"
        assert h < 2.0, f"Hedge time should be < SLO, got {h}"
        print(f"  PASSED: Hedge triggers on slow primary (h={h:.2f}s)")

    def test_no_hedge_when_primary_fast(self):
        """Hedge should trigger just after primary's P99 when primary is fast.

        When all samples are < 400ms, the optimal hedge time is around 400ms.
        This is correct: if we wait 400ms and primary hasn't responded,
        we're already in the tail (beyond all observed samples), so hedge.

        The hedge time should be close to the max observed latency, not close to SLO.
        """
        costs = {"Fast": 0.001, "Backup": 0.002}
        router = OnlineLatencyRouter(costs=costs, slo_sec=2.0, window_sec=60)

        now = time.time()

        # Both providers fast: 200-380ms (max is 380ms = 0.38s)
        for i in range(10):
            router.add_sample("Fast", now - 30 + i, 200 + i * 20)  # 200-380ms
            router.add_sample("Backup", now - 30 + i, 250 + i * 15)  # 250-385ms

        # Compute hedge time
        params = HedgingParams(strategy=HedgingStrategy.SMART_SURVIVAL, slo_sec=2.0)
        hedger = SmartHedger(params, costs)

        h = hedger.compute_hedge_time("Fast", "Backup", router.profiles, now)
        # Hedge time should be around max observed latency (~0.4s), not close to SLO
        # Algorithm: if elapsed > all samples, we're in tail territory, so hedge
        assert h > 0.3, f"Hedge time should be > 0.3s, got {h}"
        assert h < 0.6, f"Hedge time should be < 0.6s (near max sample), got {h}"
        print(f"  PASSED: Hedge time near max sample when primary is fast (h={h:.2f}s)")

    def test_survival_function(self):
        """Survival function should compute P(T > t)."""
        profile = ProviderProfile(provider="Test", window_sec=60)
        now = time.time()

        # Add samples: 500, 600, 700, 800, 900 ms (in seconds: 0.5-0.9)
        for i, ttft in enumerate([500, 600, 700, 800, 900]):
            profile.add_sample(now - 30 + i, ttft)

        # S(0.75) = P(T > 0.75s) = 2/5 = 0.4 (800ms and 900ms > 750ms)
        survival = get_survival_for_hedging(profile, 0.75, now)
        assert 0.35 < survival < 0.45, f"Expected survival ~0.4, got {survival}"
        print(f"  PASSED: Survival function works (S(0.75s)={survival:.2f})")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Phase 5 Design Tests")
    print("=" * 60)

    test_classes = [
        TestSingleWindowDesign,
        TestSharedRouter,
        TestHedgingLogic,
    ]

    total_passed = 0
    total_failed = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    getattr(instance, method_name)()
                    total_passed += 1
                except AssertionError as e:
                    print(f"  FAILED: {method_name}: {e}")
                    total_failed += 1
                except Exception as e:
                    print(f"  ERROR: {method_name}: {e}")
                    total_failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {total_passed} passed, {total_failed} failed")
    print("=" * 60)

    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
