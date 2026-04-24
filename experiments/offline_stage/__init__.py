"""Paper offline/stage experiment package."""

from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).with_name("configs") / "experiment.yaml"

__all__ = ["ExperimentConfig", "DEFAULT_CONFIG_PATH", "load_default_config"]


def __getattr__(name: str):
    if name in {"ExperimentConfig", "load_default_config"}:
        from experiments.offline_stage.config import ExperimentConfig, load_default_config

        return {
            "ExperimentConfig": ExperimentConfig,
            "load_default_config": load_default_config,
        }[name]
    raise AttributeError(name)
