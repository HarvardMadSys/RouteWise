"""Routing strategies for offline simulation and online decision-making."""

_EXPORT_MODULES = {
    "RoutingStrategy": "rwsim.offline.strategy",
    "AllAPIStrategy": "experiments.offline_stage.strategies.all_api",
    "GreedyStrategy": "experiments.offline_stage.strategies.greedy",
    "OptimalStrategy": "experiments.offline_stage.strategies.stage1_optimal",
    "OnlineStrategy": "experiments.offline_stage.strategies.online",
    "GreedyOnlineStrategy": "experiments.offline_stage.strategies.online",
    "PrimalDualOnlineStrategy": "experiments.offline_stage.strategies.online",
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
