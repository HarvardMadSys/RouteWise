"""Registry for config-driven experiment packages."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


_EXPERIMENT_MODULES = {
    "estimator_ablation": "experiments.estimator_ablation.experiment",
    "tiered_capacity": "experiments.tiered_capacity.experiment",
}


def available_experiments() -> tuple[str, ...]:
    """Return known experiment package names."""
    return tuple(sorted(_EXPERIMENT_MODULES))


def get_experiment(name: str) -> ModuleType:
    """Import and return an experiment module by name."""
    try:
        module_name = _EXPERIMENT_MODULES[name]
    except KeyError as exc:
        known = ", ".join(available_experiments())
        raise ValueError(f"Unknown experiment {name!r}. Known: {known}") from exc
    return import_module(module_name)


__all__ = ["available_experiments", "get_experiment"]
