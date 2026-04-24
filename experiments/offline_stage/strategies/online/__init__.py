"""Online routing strategies for offline/stage experiments."""

_EXPORT_MODULES = {
    "OnlineStrategy": "experiments.offline_stage.strategies.online.base",
    "GreedyOnlineStrategy": "experiments.offline_stage.strategies.online.greedy",
    "GreedyCostAwareStrategy": "experiments.offline_stage.strategies.online.greedy",
    "PrimalDualQuotaManager": "experiments.offline_stage.strategies.online.primal_dual",
    "CAPQConcurrencyManager": "experiments.offline_stage.strategies.online.primal_dual",
    "PrimalDualOnlineStrategy": "experiments.offline_stage.strategies.online.primal_dual",
    "LAPDConfig": "experiments.offline_stage.strategies.online.learning_augmented",
    "LearningAugmentedPrimalDualStrategy": (
        "experiments.offline_stage.strategies.online.learning_augmented"
    ),
    "LearningAugmentedUnifiedStrategy": (
        "experiments.offline_stage.strategies.online.learning_augmented"
    ),
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
