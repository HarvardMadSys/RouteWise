"""Tiered capacity experiment configs."""

from experiments.tiered_capacity.experiment import (
    CONFIG_DIR,
    EXPERIMENT_NAME,
    list_scenarios,
    load_all_scenarios,
    load_all_world_scenarios,
    load_scenario,
    load_world_scenario,
    run_strategy,
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
    "make_stress_scenarios",
    "run_strategy",
    "summarize",
]


def __getattr__(name: str):
    """Resolve optional experiment helpers lazily."""
    if name == "make_mm25_scenarios":
        from experiments.tiered_capacity import minimax_m25

        return minimax_m25.make_mm25_scenarios
    if name == "make_stress_scenarios":
        from experiments.tiered_capacity import stress

        return stress.make_stress_scenarios
    raise AttributeError(f"module 'experiments.tiered_capacity' has no attribute {name!r}")
