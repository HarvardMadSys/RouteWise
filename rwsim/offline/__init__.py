"""Offline/stage simulation primitives.

Imports from this package are intentionally lazy so dependency-light checks can
inspect the architecture without importing NumPy, tqdm, or solver packages.
"""

_EXPORT_MODULES = {
    "Request": "rwsim.offline.schemas",
    "ProviderConfig": "rwsim.offline.schemas",
    "ProviderType": "rwsim.offline.schemas",
    "RoutingDecision": "rwsim.offline.schemas",
    "CostCalculator": "rwsim.offline.cost",
    "QuotaManager": "rwsim.offline.quota",
    "PlanConfig": "rwsim.offline.window_quota",
    "WindowQuotaManager": "rwsim.offline.window_quota",
    "build_model_plan_mapping": "rwsim.offline.window_quota",
    "RoutingStrategy": "rwsim.offline.strategy",
    "OfflineSimulator": "rwsim.offline.simulator",
    "SimulationResult": "rwsim.offline.simulator",
    "get_dataset_cache_path": "rwsim.offline.cache",
    "load_cached_dataset": "rwsim.offline.cache",
    "save_dataset_cache": "rwsim.offline.cache",
    "get_ilp_cache_key": "rwsim.offline.cache",
    "get_ilp_cache_path": "rwsim.offline.cache",
    "load_cached_ilp_result": "rwsim.offline.cache",
    "save_ilp_cache": "rwsim.offline.cache",
    "clear_cache": "rwsim.offline.cache",
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
