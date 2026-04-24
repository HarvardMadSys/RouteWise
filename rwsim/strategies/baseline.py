"""Baseline strategy wrappers."""

from __future__ import annotations

import numpy as np


def _runner_module():
    from experiment.scripts.simulate.synthetic import runner as latency_runner

    return latency_runner


def run_cheapest_fixed(scenario, requests, seed: int = 42):
    """Run cheapest-fixed routing."""
    return _runner_module()._run_cheapest_fixed(
        scenario, requests, np.random.default_rng(seed)
    )


def run_fastest_fixed(scenario, requests, seed: int = 42):
    """Run fastest-fixed routing."""
    return _runner_module()._run_fastest_fixed(
        scenario, requests, np.random.default_rng(seed)
    )


def run_round_robin(scenario, requests, seed: int = 42):
    """Run round-robin routing."""
    return _runner_module()._run_round_robin(
        scenario, requests, np.random.default_rng(seed)
    )


def run_oracle_per_window(scenario, requests, seed: int = 42):
    """Run hindsight oracle routing per time window."""
    return _runner_module()._run_oracle_per_window(
        scenario, requests, np.random.default_rng(seed)
    )


BASELINE_STRATEGIES = {
    "cheapest_fixed": run_cheapest_fixed,
    "fastest_fixed": run_fastest_fixed,
    "round_robin": run_round_robin,
    "oracle_per_window": run_oracle_per_window,
}


__all__ = [
    "BASELINE_STRATEGIES",
    "run_cheapest_fixed",
    "run_fastest_fixed",
    "run_oracle_per_window",
    "run_round_robin",
]
