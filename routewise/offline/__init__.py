"""Offline/stage simulation primitives.

Imports from this package are intentionally lazy so dependency-light checks can
inspect the architecture without importing NumPy, tqdm, or solver packages.
"""

_EXPORT_MODULES = {
    "Request": "routewise.offline.schemas",
    "ProviderConfig": "routewise.offline.schemas",
    "ProviderType": "routewise.offline.schemas",
    "RoutingDecision": "routewise.offline.schemas",
    "CostCalculator": "routewise.offline.cost",
    "QuotaManager": "routewise.offline.quota",
    "PlanConfig": "routewise.offline.window_quota",
    "WindowQuotaManager": "routewise.offline.window_quota",
    "build_model_plan_mapping": "routewise.offline.window_quota",
    "RoutingStrategy": "routewise.offline.strategy",
    "OfflineSimulator": "routewise.offline.simulator",
    "SimulationResult": "routewise.offline.simulator",
    "get_dataset_cache_path": "routewise.offline.cache",
    "load_cached_dataset": "routewise.offline.cache",
    "save_dataset_cache": "routewise.offline.cache",
    "get_ilp_cache_key": "routewise.offline.cache",
    "get_ilp_cache_path": "routewise.offline.cache",
    "load_cached_ilp_result": "routewise.offline.cache",
    "save_ilp_cache": "routewise.offline.cache",
    "clear_cache": "routewise.offline.cache",
}

__all__ = tuple(_EXPORT_MODULES)


def __getattr__(name: str):
    if name not in _EXPORT_MODULES:
        raise AttributeError(name)

    from importlib import import_module

    module = import_module(_EXPORT_MODULES[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
