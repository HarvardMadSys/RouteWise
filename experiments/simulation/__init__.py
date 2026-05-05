"""Main simulation experiment configs."""

from experiments.simulation.experiment import (
    CONFIG_DIR,
    EXPERIMENT_NAME,
    list_scenarios,
    load_all_scenarios,
    load_all_world_scenarios,
    load_scenario,
    load_world_scenario,
    run_policy,
    summarize,
)

__all__ = [
    "CONFIG_DIR",
    "EXPERIMENT_NAME",
    "list_scenarios",
    "load_all_scenarios",
    "load_all_world_scenarios",
    "load_scenario",
    "load_world_scenario",
    "make_mm25_scenarios",
    "make_simple_scenarios",
    "make_calibrated_scenarios",
    "run_policy",
    "summarize",
]


def __getattr__(name: str):
    """Resolve optional experiment helpers lazily."""
    if name == "make_mm25_scenarios":
        from experiments.simulation import minimax_m25

        return minimax_m25.make_mm25_scenarios
    if name == "make_simple_scenarios":
        from experiments.simulation import simple_scenarios

        return simple_scenarios.make_simple_scenarios
    if name == "make_calibrated_scenarios":
        from experiments.simulation import calibrated

        return calibrated.make_calibrated_scenarios
    raise AttributeError(f"module 'experiments.simulation' has no attribute {name!r}")
