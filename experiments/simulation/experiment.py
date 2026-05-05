"""Thin config runner for simulation scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments._configs import (
    list_config_names,
    load_all_named_scenarios,
    load_named_scenario,
    summarize_scenario,
)
from rwsim.schemas import ScenarioConfig


EXPERIMENT_NAME = "simulation"
CONFIG_DIR = Path(__file__).with_name("configs")


def list_scenarios() -> tuple[str, ...]:
    """Return available simulation scenario names."""
    return list_config_names(CONFIG_DIR)


def load_scenario(name: str) -> ScenarioConfig:
    """Load one simulation scenario."""
    return load_named_scenario(CONFIG_DIR, name)


def load_all_scenarios() -> tuple[ScenarioConfig, ...]:
    """Load all simulation scenarios."""
    return load_all_named_scenarios(CONFIG_DIR)


def load_world_scenario(name: str):
    """Load one simulation scenario as runnable world objects."""
    from experiments.simulation.materialize import scenario as materialize_scenario

    return materialize_scenario(load_scenario(name))


def load_all_world_scenarios() -> tuple[object, ...]:
    """Load all simulation scenarios as runnable world objects."""
    from experiments.simulation.materialize import scenario as materialize_scenario

    return tuple(materialize_scenario(item) for item in load_all_scenarios())


def run_policy(scenario_name: str, policy: str, seed: int = 42) -> dict[str, Any]:
    """Run one policy on one config-driven simulation scenario."""
    from rwsim.runner import run_policy as run_named_policy

    scenario = load_world_scenario(scenario_name)
    requests = _load_requests(scenario_name)
    for provider in scenario.providers:
        provider.reset_state()

    run = run_named_policy(scenario, requests, policy, seed=seed)
    return _summarize_run(scenario_name, policy, seed, run, scenario.primary_slo_ms)


def summarize(name: str) -> dict[str, Any]:
    """Load and summarize one simulation scenario."""
    return summarize_scenario(load_scenario(name))


def _summarize_run(
    scenario_name: str,
    policy: str,
    seed: int,
    run,
    primary_slo_ms: float,
) -> dict[str, Any]:
    return {
        "scenario": scenario_name,
        "policy": policy,
        "seed": seed,
        "n_requests": len(run.provider),
        "slo_violation_rate": run.slo_violation_rate(primary_slo_ms),
        "mean_cost_usd": run.mean_cost_usd(),
        "p50_ms": run.p50_ms(),
        "p99_ms": run.p99_ms(),
        "hedge_rate": run.hedge_rate(),
        "provider_fractions": run.provider_fractions(),
        "tier_fractions": run.tier_fractions() if hasattr(run, "tier_fractions") else {},
    }


def _load_requests(scenario_name: str):
    """Load requests for a config-driven scenario from trace data."""
    from experiments.simulation.lp_budget_eval import generate_scenario_workload

    schema_scenario = load_scenario(scenario_name)
    source = schema_scenario.workload.source
    return generate_scenario_workload(
        load_world_scenario(scenario_name),
        seed=schema_scenario.workload.seed,
        dataset_name=source,
    )


__all__ = [
    "CONFIG_DIR",
    "EXPERIMENT_NAME",
    "list_scenarios",
    "load_all_scenarios",
    "load_all_world_scenarios",
    "load_scenario",
    "load_world_scenario",
    "run_policy",
    "summarize",
]
