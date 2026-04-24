"""Offline counterfactual paper experiment."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "compute_metrics",
    "compute_summary",
    "load_openrouter_auto",
    "run_simulation",
]


def __getattr__(name: str):
    """Resolve experiment helpers lazily so `python -m` stays warning-free."""
    if name in __all__:
        module = import_module("experiments.offline_counterfactual.experiment")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
