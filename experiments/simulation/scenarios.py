"""Compatibility scenario factories for the simulation experiment."""

from experiments.simulation import list_scenarios, load_world_scenario
from rwsim.world.scenarios import ScenarioConfig


def make_simulation_scenarios() -> dict[str, ScenarioConfig]:
    """Create all simulation scenarios from configs."""
    return {name: load_world_scenario(name) for name in list_scenarios()}


__all__ = ["ScenarioConfig", "make_simulation_scenarios"]
