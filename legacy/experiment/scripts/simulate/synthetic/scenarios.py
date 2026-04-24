"""Legacy compatibility wrapper for synthetic latency scenarios.

The canonical S1-S5 definitions live in
``experiments/synthetic_latency/configs``.
"""

from experiments.synthetic_latency import list_scenarios, load_world_scenario
from rwsim.world.scenarios import ScenarioConfig


def make_scenarios() -> dict[str, ScenarioConfig]:
    """Create all synthetic latency evaluation scenarios from configs."""
    return {name: load_world_scenario(name) for name in list_scenarios()}


__all__ = ["ScenarioConfig", "make_scenarios"]
