"""Thin config runner for tiered capacity scenarios."""

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


EXPERIMENT_NAME = "tiered_capacity"
CONFIG_DIR = Path(__file__).with_name("configs")


def list_scenarios() -> tuple[str, ...]:
    """Return available tiered capacity scenario names."""
    return list_config_names(CONFIG_DIR)


def load_scenario(name: str) -> ScenarioConfig:
    """Load one tiered capacity scenario."""
    return load_named_scenario(CONFIG_DIR, name)


def load_all_scenarios() -> tuple[ScenarioConfig, ...]:
    """Load all tiered capacity scenarios."""
    return load_all_named_scenarios(CONFIG_DIR)


def load_world_scenario(name: str):
    """Load one tiered capacity scenario as runnable world objects."""
    from experiments.tiered_capacity.materialize import scenario as materialize_scenario

    return materialize_scenario(load_scenario(name))


def load_all_world_scenarios() -> tuple[object, ...]:
    """Load all tiered capacity scenarios as runnable world objects."""
    from experiments.tiered_capacity.materialize import scenario as materialize_scenario

    return tuple(materialize_scenario(item) for item in load_all_scenarios())


def summarize(name: str) -> dict[str, Any]:
    """Load and summarize one tiered capacity scenario."""
    return summarize_scenario(load_scenario(name))


__all__ = [
    "CONFIG_DIR",
    "EXPERIMENT_NAME",
    "list_scenarios",
    "load_all_scenarios",
    "load_all_world_scenarios",
    "load_scenario",
    "load_world_scenario",
    "summarize",
]
