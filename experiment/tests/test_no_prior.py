#!/usr/bin/env python3
"""Test simplified single-window design."""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Use same import pattern as phase5_online_evaluation.py
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
load_pricing = _module.load_pricing

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


def test_router_basic():
    """Test OnlineLatencyRouter basic functionality."""
    print("Test 1: OnlineLatencyRouter basic functionality")
    costs = {"ProviderA": 0.001, "ProviderB": 0.002, "ProviderC": 0.0005}
    router = OnlineLatencyRouter(costs=costs, slo_sec=2.0, window_sec=15 * 60)

    # Add some samples
    now = time.time()
    for i in range(20):
        router.add_sample("ProviderA", now - 100 + i * 5, 500 + i * 10)  # 500-690ms
        router.add_sample("ProviderB", now - 100 + i * 5, 800 + i * 20)  # 800-1180ms
        router.add_sample("ProviderC", now - 100 + i * 5, 300 + i * 5)  # 300-395ms

    # Force LP update
    router.update_lp(now, force=True)
    print(f"  LP status: {router.last_lp_status}")
    print(f"  LP weights: {router.last_lp_weights}")

    # Test routing
    provider = router.route(now)
    print(f"  Routed to: {provider}")
    assert provider is not None, "Should route to a provider"
    print("  PASSED")
    return router


def test_single_window():
    """Test that ProviderProfile uses single window."""
    print()
    print("Test 2: Single window design")
    profile = ProviderProfile(provider="Test", window_sec=60)

    # Verify single window fields
    assert hasattr(profile, "samples"), "Should have 'samples' field"
    assert not hasattr(profile, "short_window_samples"), "Should not have 'short_window_samples'"
    assert not hasattr(profile, "long_window_samples"), "Should not have 'long_window_samples'"
    assert hasattr(profile, "window_sec"), "Should have 'window_sec' field"
    assert not hasattr(profile, "short_window_sec"), "Should not have 'short_window_sec'"
    assert not hasattr(profile, "long_window_sec"), "Should not have 'long_window_sec'"
    assert not hasattr(profile, "prior_strength"), "Should not have 'prior_strength'"
    print("  ProviderProfile has correct single-window structure")
    print("  PASSED")


def test_survival_function(router):
    """Test survival function with single window."""
    print()
    print("Test 3: Survival function")
    now = time.time()
    profile = router.profiles["ProviderA"]
    survival = get_survival_for_hedging(profile, 0.6, now)  # 600ms
    print(f"  S(0.6s) for ProviderA: {survival:.3f}")

    # Test with empty profile
    empty_profile = ProviderProfile(provider="Empty")
    survival_empty = get_survival_for_hedging(empty_profile, 1.0, now)
    print(f"  S(1.0s) for empty profile: {survival_empty:.3f} (should be 1.0)")
    assert survival_empty == 1.0, "Empty profile should return 1.0"
    print("  PASSED")


def test_hedger(router):
    """Test SmartHedger with single window."""
    print()
    print("Test 4: SmartHedger compute_hedge_time")
    now = time.time()
    costs = {"ProviderA": 0.001, "ProviderB": 0.002, "ProviderC": 0.0005}
    params = HedgingParams(strategy=HedgingStrategy.SMART_SURVIVAL, slo_sec=2.0)
    hedger = SmartHedger(params, costs)

    h = hedger.compute_hedge_time("ProviderA", "ProviderC", router.profiles, now)
    print(f"  Hedge time (A->C): {h:.3f}s")
    print("  PASSED")


def test_cdf():
    """Test CDF computation with single window."""
    print()
    print("Test 5: CDF computation")
    profile = ProviderProfile(provider="Test", window_sec=60)
    now = time.time()

    # Add samples
    for i in range(10):
        profile.add_sample(now - 30 + i, 500 + i * 50)  # 500-950ms

    cdf = profile.get_cdf_at(0.7, now)  # 700ms
    print(f"  CDF at 0.7s: {cdf:.3f}")

    # Should be roughly 0.4-0.5 (4-5 samples out of 10 are <= 700ms)
    assert 0.3 < cdf < 0.6, f"CDF should be ~0.4-0.5, got {cdf}"
    print("  PASSED")


def test_no_legacy():
    """Verify legacy code is removed."""
    print()
    print("Test 6: Verify legacy code removed")
    costs = {"ProviderA": 0.001}
    router = OnlineLatencyRouter(costs=costs, slo_sec=2.0)

    # Check no load_offline_profile
    assert not hasattr(router, "load_offline_profile"), "load_offline_profile should not exist"
    print("  No load_offline_profile method")

    # Check no prior_strength
    assert not hasattr(router, "prior_strength"), "prior_strength should not exist"
    print("  No prior_strength attribute")

    print("  PASSED")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing simplified single-window design")
    print("=" * 60)
    print()

    router = test_router_basic()
    test_single_window()
    test_survival_function(router)
    test_hedger(router)
    test_cdf()
    test_no_legacy()

    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
