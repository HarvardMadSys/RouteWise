"""Compatibility exports for offline/stage routing strategies."""

_EXPORT_MODULES = {
    "RoutingStrategy": "rwsim.offline.strategy",
    "AllAPIStrategy": "legacy.experiment.strategies.all_api",
    "GreedyStrategy": "legacy.experiment.strategies.greedy",
    "OptimalStrategy": "legacy.experiment.strategies.stage1_optimal",
    "OnlineStrategy": "legacy.experiment.strategies.online",
    "GreedyOnlineStrategy": "legacy.experiment.strategies.online",
    "PrimalDualOnlineStrategy": "legacy.experiment.strategies.online",
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
