"""LP-family strategy wrappers."""

from __future__ import annotations

import numpy as np


def _runner_module():
    from rwsim.strategies import latency_impl as latency_runner

    return latency_runner


def _slo_sec(scenario) -> float:
    return scenario.primary_slo_ms / 1000.0


def run_lp_mix(scenario, requests, seed: int = 42):
    """Run LP mix routing."""
    return _runner_module()._run_lp_mix(
        scenario, requests, np.random.default_rng(seed), _slo_sec(scenario)
    )


def run_lp_hedge(scenario, requests, seed: int = 42):
    """Run LP routing with smart economic hedging."""
    return _runner_module()._run_lp_hedge(
        scenario, requests, np.random.default_rng(seed), _slo_sec(scenario)
    )


def run_lp_explorer(scenario, requests, seed: int = 42):
    """Run LP routing with hedge-as-probe exploration."""
    return _runner_module()._run_lp_hedge(
        scenario,
        requests,
        np.random.default_rng(seed),
        _slo_sec(scenario),
        hedge_as_probe=True,
        probe_rate=None,
        strategy_label="lp_explorer",
    )


def run_lp_explorer_no_probe(scenario, requests, seed: int = 42):
    """Run LP explorer with probe dispatch disabled."""
    return _runner_module()._run_lp_hedge(
        scenario,
        requests,
        np.random.default_rng(seed),
        _slo_sec(scenario),
        hedge_as_probe=True,
        probe_rate=0.0,
        strategy_label="lp_explorer_no_probe",
    )


LP_STRATEGIES = {
    "lp_mix": run_lp_mix,
    "lp_hedge": run_lp_hedge,
    "lp_explorer": run_lp_explorer,
    "lp_explorer_no_probe": run_lp_explorer_no_probe,
}


__all__ = [
    "LP_STRATEGIES",
    "run_lp_explorer",
    "run_lp_explorer_no_probe",
    "run_lp_hedge",
    "run_lp_mix",
]
