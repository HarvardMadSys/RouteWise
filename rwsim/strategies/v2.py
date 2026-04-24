"""V2-family strategy wrappers."""

from __future__ import annotations

import numpy as np


def _runner_module():
    from rwsim.strategies import latency_impl as latency_runner

    return latency_runner


def _slo_sec(scenario) -> float:
    return scenario.primary_slo_ms / 1000.0


def run_v2_only(scenario, requests, seed: int = 42):
    """Run V2 routing without hedging."""
    return _runner_module()._run_v2_only(
        scenario, requests, np.random.default_rng(seed), _slo_sec(scenario)
    )


def run_v2_p50_hedge(scenario, requests, seed: int = 42):
    """Run V2 routing with P50-band hedging."""
    return _runner_module()._run_v2_p50_hedge(
        scenario, requests, np.random.default_rng(seed), _slo_sec(scenario)
    )


def run_v2_explorer(scenario, requests, seed: int = 42):
    """Run V2 routing with hedge-as-probe exploration."""
    return _runner_module()._run_v2_p50_hedge(
        scenario,
        requests,
        np.random.default_rng(seed),
        _slo_sec(scenario),
        hedge_as_probe=True,
        probe_rate=None,
        strategy_label="v2_explorer",
    )


def run_v2_explorer_no_probe(scenario, requests, seed: int = 42):
    """Run V2 explorer with probe dispatch disabled."""
    return _runner_module()._run_v2_p50_hedge(
        scenario,
        requests,
        np.random.default_rng(seed),
        _slo_sec(scenario),
        hedge_as_probe=True,
        probe_rate=0.0,
        strategy_label="v2_explorer_no_probe",
    )


V2_STRATEGIES = {
    "v2_only": run_v2_only,
    "v2_p50_hedge": run_v2_p50_hedge,
    "v2_explorer": run_v2_explorer,
    "v2_explorer_no_probe": run_v2_explorer_no_probe,
}


__all__ = [
    "V2_STRATEGIES",
    "run_v2_explorer",
    "run_v2_explorer_no_probe",
    "run_v2_only",
    "run_v2_p50_hedge",
]
