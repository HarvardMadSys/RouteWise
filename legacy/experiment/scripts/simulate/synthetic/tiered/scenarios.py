"""Compatibility wrapper for config-driven tiered capacity scenarios."""

from __future__ import annotations

from experiments.tiered_capacity import list_scenarios, load_world_scenario
from rwsim.world.scenarios import ScenarioConfig as TieredScenarioConfig


def make_tiered_scenarios() -> dict[str, TieredScenarioConfig]:
    """Create S6-S9 and unified_pool scenarios from experiment configs."""
    return {name: load_world_scenario(name) for name in list_scenarios()}


__all__ = ["TieredScenarioConfig", "make_tiered_scenarios"]
