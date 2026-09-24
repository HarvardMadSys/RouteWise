"""Thin config runner for estimator ablation scenarios."""

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


EXPERIMENT_NAME = "estimator_ablation"
CONFIG_DIR = Path(__file__).with_name("configs")


def list_scenarios() -> tuple[str, ...]:
    """Return available estimator ablation scenario names."""
    return list_config_names(CONFIG_DIR)


def load_scenario(name: str) -> ScenarioConfig:
    """Load one estimator ablation scenario."""
    return load_named_scenario(CONFIG_DIR, name)


def load_all_scenarios() -> tuple[ScenarioConfig, ...]:
    """Load all estimator ablation scenarios."""
    return load_all_named_scenarios(CONFIG_DIR)


def summarize(name: str) -> dict[str, Any]:
    """Load and summarize one estimator ablation scenario."""
    return summarize_scenario(load_scenario(name))


__all__ = [
    "CONFIG_DIR",
    "EXPERIMENT_NAME",
    "list_scenarios",
    "load_all_scenarios",
    "load_scenario",
    "summarize",
]
