"""Tiered strategy wrappers."""

from __future__ import annotations

import numpy as np


def _tiered_module():
    from ...tiered import strategies as tiered_strategies

    return tiered_strategies


def run_two_layer(scenario, requests, seed: int = 42):
    return _tiered_module()._run_two_layer(
        scenario, requests, np.random.default_rng(seed)
    )


def run_joint_nohedge(scenario, requests, seed: int = 42):
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
    mod = _tiered_module()
    return mod._run_joint(
        scenario,
        requests,
        np.random.default_rng(seed),
        strategy_name="joint_p50band_hedge",
        selector=mod._joint_select_p50band,
        use_hedge=True,
    )


def run_joint_ucb(scenario, requests, seed: int = 42):
    mod = _tiered_module()
    return mod._run_joint_ucb(
        scenario,
        requests,
        np.random.default_rng(seed),
        use_hedge=False,
        strategy_name="joint_ucb",
    )


def run_joint_ucb_hedge(scenario, requests, seed: int = 42):
    mod = _tiered_module()
    return mod._run_joint_ucb(
        scenario,
        requests,
        np.random.default_rng(seed),
        use_hedge=True,
        strategy_name="joint_ucb_hedge",
    )


TIERED_STRATEGIES = {
    "two_layer": run_two_layer,
    "joint_nohedge": run_joint_nohedge,
    "joint_hedge": run_joint_hedge,
    "joint_p50band_nohedge": run_joint_p50band_nohedge,
    "joint_p50band_hedge": run_joint_p50band_hedge,
    "joint_ucb": run_joint_ucb,
    "joint_ucb_hedge": run_joint_ucb_hedge,
}

