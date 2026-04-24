"""Compatibility scenario factories for the tiered capacity experiment."""

from experiments.tiered_capacity import list_scenarios, load_world_scenario
from rwsim.world.scenarios import ScenarioConfig as TieredScenarioConfig


def make_tiered_scenarios() -> dict[str, TieredScenarioConfig]:
    """Create all tiered capacity scenarios from configs."""
    return {name: load_world_scenario(name) for name in list_scenarios()}


__all__ = ["TieredScenarioConfig", "make_tiered_scenarios"]
