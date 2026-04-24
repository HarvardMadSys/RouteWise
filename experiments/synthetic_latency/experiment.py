"""Thin config runner for synthetic latency scenarios."""

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


EXPERIMENT_NAME = "synthetic_latency"
CONFIG_DIR = Path(__file__).with_name("configs")


def list_scenarios() -> tuple[str, ...]:
    """Return available synthetic latency scenario names."""
    return list_config_names(CONFIG_DIR)


def load_scenario(name: str) -> ScenarioConfig:
    """Load one synthetic latency scenario."""
    return load_named_scenario(CONFIG_DIR, name)


def load_all_scenarios() -> tuple[ScenarioConfig, ...]:
    """Load all synthetic latency scenarios."""
    return load_all_named_scenarios(CONFIG_DIR)


def load_world_scenario(name: str):
    """Load one synthetic latency scenario as runnable world objects."""
    from experiments.synthetic_latency.materialize import scenario as materialize_scenario

    return materialize_scenario(load_scenario(name))


def load_all_world_scenarios() -> tuple[object, ...]:
    """Load all synthetic latency scenarios as runnable world objects."""
    from experiments.synthetic_latency.materialize import scenario as materialize_scenario

    return tuple(materialize_scenario(item) for item in load_all_scenarios())


def run_strategy(scenario_name: str, strategy: str, seed: int = 42) -> dict[str, Any]:
    """Run one registered strategy on one config-driven latency scenario."""
    from experiments.synthetic_latency.materialize import distribution
    from rwsim.runner import run_registered_strategy
    from rwsim.world import generate_workload

    generic = load_scenario(scenario_name)
    scenario = load_world_scenario(scenario_name)
    requests = generate_workload(
        scenario.n_requests,
        scenario.duration_seconds,
        seed=generic.workload.seed,
        start_time=generic.workload.start_time,
        arrival_process=scenario.arrival_process,
        output_token_dist=distribution(generic.workload.output_token_distribution),
        input_tokens=generic.workload.input_tokens,
    )
    run = run_registered_strategy(scenario, requests, strategy, seed=seed)
    return _summarize_run(scenario_name, strategy, seed, run, scenario.primary_slo_ms)


def summarize(name: str) -> dict[str, Any]:
    """Load and summarize one synthetic latency scenario."""
    return summarize_scenario(load_scenario(name))


def _summarize_run(
    scenario_name: str,
    strategy: str,
    seed: int,
    run,
    primary_slo_ms: float,
) -> dict[str, Any]:
    return {
        "scenario": scenario_name,
        "strategy": strategy,
        "seed": seed,
        "n_requests": len(run.provider),
        "slo_violation_rate": run.slo_violation_rate(primary_slo_ms),
        "mean_cost_usd": run.mean_cost_usd(),
        "p50_ms": run.p50_ms(),
        "p99_ms": run.p99_ms(),
        "hedge_rate": run.hedge_rate(),
        "provider_fractions": run.provider_fractions(),
    }


__all__ = [
    "CONFIG_DIR",
    "EXPERIMENT_NAME",
    "list_scenarios",
    "load_all_scenarios",
    "load_all_world_scenarios",
    "load_scenario",
    "load_world_scenario",
    "run_strategy",
    "summarize",
]
