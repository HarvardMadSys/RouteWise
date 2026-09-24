"""Canonical world-model exports for provider and latency primitives.

Exports are resolved lazily so dependency-light modules such as
``llm_routewise.capacity`` can be imported before the scientific stack is
installed.
"""

from __future__ import annotations

_EXPORT_MODULES = {
    "ConcurrencyState": "llm_routewise.capacity",
    "MultiWindowQuotaState": "llm_routewise.capacity",
    "ProviderTier": "llm_routewise.capacity",
    "QuotaState": "llm_routewise.capacity",
    "HeavyTail": "llm_routewise.sim.world.distributions",
    "LATENCY_FAMILIES": "llm_routewise.sim.world.distributions",
    "LatencyDistribution": "llm_routewise.sim.world.distributions",
    "LogNormal": "llm_routewise.sim.world.distributions",
    "Normal": "llm_routewise.sim.world.distributions",
    "Uniform": "llm_routewise.sim.world.distributions",
    "EmpiricalDistribution": "llm_routewise.sim.world.empirical",
    "Provider": "llm_routewise.sim.world.providers",
    "ShiftingProvider": "llm_routewise.sim.world.providers",
    "SyntheticProvider": "llm_routewise.sim.world.providers",
    "TieredProvider": "llm_routewise.sim.world.providers",
    "ScenarioConfig": "llm_routewise.sim.world.scenarios",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str):
    """Resolve public world-model exports lazily."""
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(f"module 'llm_routewise.sim.world' has no attribute {name!r}") from exc

    from importlib import import_module

    module = import_module(module_name)
    return getattr(module, name)
