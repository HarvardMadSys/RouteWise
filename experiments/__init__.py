"""Experiment config packages.

Experiments own reproducible configs and thin runners. Core simulation behavior
belongs in :mod:`llm_routewise.sim`.
"""

from experiments.registry import available_experiments, get_experiment

__all__ = ["available_experiments", "get_experiment"]
