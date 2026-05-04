"""Simulation metrics: result containers and aggregation primitives.

This package owns the simulator output protocol. World-model objects
(`rwsim/world/`) describe providers and scenarios; this package describes
*what comes out* of running a strategy on a scenario.
"""

from rwsim.metrics.run import StrategyRun

__all__ = ["StrategyRun"]
