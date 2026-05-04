"""Canonical world-model exports for the synthetic simulator.

Exports are resolved lazily so dependency-light modules such as
``rwsim.world.capacity`` can be imported before the scientific stack is
installed.
"""

from __future__ import annotations


_EXPORT_MODULES = {
    "ConcurrencyState": "rwsim.world.capacity",
    "ProviderTier": "rwsim.world.capacity",
    "QuotaState": "rwsim.world.capacity",
    "HeavyTail": "rwsim.world.distributions",
    "LATENCY_FAMILIES": "rwsim.world.distributions",
    "LatencyDistribution": "rwsim.world.distributions",
    "LogNormal": "rwsim.world.distributions",
    "Normal": "rwsim.world.distributions",
    "Uniform": "rwsim.world.distributions",
    "StrategyRun": "rwsim.metrics.run",
    "Provider": "rwsim.world.providers",
    "ShiftingProvider": "rwsim.world.providers",
    "SyntheticProvider": "rwsim.world.providers",
    "TieredProvider": "rwsim.world.providers",
    "ScenarioConfig": "rwsim.world.scenarios",
    "calibrate_envelopes": "rwsim.world.shadow_price",
    "concurrency_shadow_price": "rwsim.world.shadow_price",
    "effective_cost": "rwsim.world.shadow_price",
    "quota_shadow_price": "rwsim.world.shadow_price",
    "generate_workload": "rwsim.world.workload",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str):
    """Resolve public world-model exports lazily."""
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(f"module 'rwsim.world' has no attribute {name!r}") from exc

    from importlib import import_module

    module = import_module(module_name)
    return getattr(module, name)
