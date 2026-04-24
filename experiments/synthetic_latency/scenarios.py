"""Compatibility scenario factories for the synthetic latency experiment."""

from experiments.synthetic_latency import list_scenarios, load_world_scenario
from rwsim.world.scenarios import ScenarioConfig


def make_scenarios() -> dict[str, ScenarioConfig]:
    """Create all synthetic latency evaluation scenarios from configs."""
    return {name: load_world_scenario(name) for name in list_scenarios()}


__all__ = ["ScenarioConfig", "make_scenarios"]
