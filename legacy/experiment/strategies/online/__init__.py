"""Compatibility exports for online offline-stage strategies."""

_EXPORT_MODULES = {
    "OnlineStrategy": "legacy.experiment.strategies.online.base",
    "GreedyOnlineStrategy": "legacy.experiment.strategies.online.greedy",
    "GreedyCostAwareStrategy": "legacy.experiment.strategies.online.greedy",
    "PrimalDualQuotaManager": "legacy.experiment.strategies.online.primal_dual",
    "CAPQConcurrencyManager": "legacy.experiment.strategies.online.primal_dual",
    "PrimalDualOnlineStrategy": "legacy.experiment.strategies.online.primal_dual",
    "LAPDConfig": "legacy.experiment.strategies.online.learning_augmented",
    "LearningAugmentedPrimalDualStrategy": (
        "legacy.experiment.strategies.online.learning_augmented"
    ),
    "LearningAugmentedUnifiedStrategy": "legacy.experiment.strategies.online.learning_augmented",
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
