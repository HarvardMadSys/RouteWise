"""Deprecated: import from `rwsim.metrics` instead.

`StrategyRun` was moved to `rwsim/metrics/run.py` so the `world` package can
own only world-model objects (providers, quotas, distributions, scenarios).
This shim preserves the old import path during the migration window.
"""

from rwsim.metrics.run import StrategyRun  # noqa: F401

__all__ = ["StrategyRun"]
