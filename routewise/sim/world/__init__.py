"""Canonical world-model exports for provider and latency primitives.

Exports are resolved lazily so dependency-light modules such as
``routewise.capacity`` can be imported before the scientific stack is
installed.
"""

from __future__ import annotations

_EXPORT_MODULES = {
    "ConcurrencyState": "routewise.capacity",
    "MultiWindowQuotaState": "routewise.capacity",
    "ProviderTier": "routewise.capacity",
    "QuotaState": "routewise.capacity",
    "HeavyTail": "routewise.sim.world.distributions",
    "LATENCY_FAMILIES": "routewise.sim.world.distributions",
    "LatencyDistribution": "routewise.sim.world.distributions",
    "LogNormal": "routewise.sim.world.distributions",
    "Normal": "routewise.sim.world.distributions",
    "Uniform": "routewise.sim.world.distributions",
    "EmpiricalDistribution": "routewise.sim.world.empirical",
    "Provider": "routewise.sim.world.providers",
    "ShiftingProvider": "routewise.sim.world.providers",
    "SyntheticProvider": "routewise.sim.world.providers",
    "TieredProvider": "routewise.sim.world.providers",
    "ScenarioConfig": "routewise.sim.world.scenarios",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str):
    """Resolve public world-model exports lazily."""
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(f"module 'routewise.sim.world' has no attribute {name!r}") from exc

    from importlib import import_module

    module = import_module(module_name)
    return getattr(module, name)
