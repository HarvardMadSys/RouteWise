"""Compatibility exports for experiment data models and loaders."""

_EXPORT_MODULES = {
    "DataLoader": "legacy.experiment.data.loader",
    "normalize_model_name": "legacy.experiment.data.loader",
    "Request": "legacy.experiment.data.schema",
    "ProviderConfig": "legacy.experiment.data.schema",
    "ProviderType": "legacy.experiment.data.schema",
    "RoutingDecision": "legacy.experiment.data.schema",
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
