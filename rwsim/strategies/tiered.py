"""Tiered strategy wrappers."""

from __future__ import annotations

import numpy as np


def _tiered_module():
    from rwsim.strategies import tiered_impl as tiered_strategies

    return tiered_strategies


def run_two_layer(scenario, requests, seed: int = 42):
    """Run the current two-layer tiered baseline."""
    return _tiered_module()._run_two_layer(
        scenario, requests, np.random.default_rng(seed)
    )


def run_joint_nohedge(scenario, requests, seed: int = 42):
    """Run joint routing without hedging."""
    mod = _tiered_module()
    return mod._run_joint(
        scenario,
        requests,
        np.random.default_rng(seed),
        strategy_name="joint_nohedge",
        selector=mod._joint_select_slo_safe,
        use_hedge=False,
    )


def run_joint_hedge(scenario, requests, seed: int = 42):
    """Run joint routing with smart economic hedging."""
    mod = _tiered_module()
    return mod._run_joint(
        scenario,
        requests,
        np.random.default_rng(seed),
        strategy_name="joint_hedge",
        selector=mod._joint_select_slo_safe,
        use_hedge=True,
    )


def run_joint_p50band_nohedge(scenario, requests, seed: int = 42):
    """Run joint routing with the P50-band latency ablation."""
    mod = _tiered_module()
    return mod._run_joint(
        scenario,
        requests,
        np.random.default_rng(seed),
        strategy_name="joint_p50band_nohedge",
        selector=mod._joint_select_p50band,
        use_hedge=False,
    )


def run_joint_p50band_hedge(scenario, requests, seed: int = 42):
    """Run joint P50-band routing with smart economic hedging."""
    mod = _tiered_module()
    return mod._run_joint(
        scenario,
        requests,
        np.random.default_rng(seed),
        strategy_name="joint_p50band_hedge",
        selector=mod._joint_select_p50band,
        use_hedge=True,
    )


TIERED_STRATEGIES = {
    "two_layer": run_two_layer,
    "joint_nohedge": run_joint_nohedge,
    "joint_hedge": run_joint_hedge,
    "joint_p50band_nohedge": run_joint_p50band_nohedge,
    "joint_p50band_hedge": run_joint_p50band_hedge,
}


__all__ = [
    "TIERED_STRATEGIES",
    "run_joint_hedge",
    "run_joint_nohedge",
    "run_joint_p50band_hedge",
    "run_joint_p50band_nohedge",
    "run_two_layer",
]
