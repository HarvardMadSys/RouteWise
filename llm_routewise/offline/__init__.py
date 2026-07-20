"""Offline/stage simulation primitives.

Imports from this package are intentionally lazy so dependency-light checks can
inspect the architecture without importing NumPy, tqdm, or solver packages.
"""

_EXPORT_MODULES = {
    "Request": "llm_routewise.offline.schemas",
    "ProviderConfig": "llm_routewise.offline.schemas",
    "ProviderType": "llm_routewise.offline.schemas",
    "RoutingDecision": "llm_routewise.offline.schemas",
    "CostCalculator": "llm_routewise.offline.cost",
    "QuotaManager": "llm_routewise.offline.quota",
    "PlanConfig": "llm_routewise.offline.window_quota",
    "WindowQuotaManager": "llm_routewise.offline.window_quota",
    "build_model_plan_mapping": "llm_routewise.offline.window_quota",
    "RoutingStrategy": "llm_routewise.offline.strategy",
    "OfflineSimulator": "llm_routewise.offline.simulator",
    "SimulationResult": "llm_routewise.offline.simulator",
    "get_dataset_cache_path": "llm_routewise.offline.cache",
    "load_cached_dataset": "llm_routewise.offline.cache",
    "save_dataset_cache": "llm_routewise.offline.cache",
    "get_ilp_cache_key": "llm_routewise.offline.cache",
    "get_ilp_cache_path": "llm_routewise.offline.cache",
    "load_cached_ilp_result": "llm_routewise.offline.cache",
    "save_ilp_cache": "llm_routewise.offline.cache",
    "clear_cache": "llm_routewise.offline.cache",
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
