"""Baseline strategy implementation exports.

The implementations still live in the legacy synthetic modules during
migration. New code should import through this rwsim namespace.
"""

from experiment.scripts.simulate.synthetic._core.strategies.baseline import BASELINE_STRATEGIES

__all__ = ["BASELINE_STRATEGIES"]
