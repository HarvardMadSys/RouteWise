"""Shared helpers for config-driven experiments."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from routewise.schemas import ScenarioConfig
from routewise.sim.scenarios import load_scenario_config


def list_config_paths(config_dir: Path) -> tuple[Path, ...]:
    """Return sorted YAML config paths for an experiment."""
    if not config_dir.exists():
        return ()
    return tuple(sorted(path for path in config_dir.glob("*.yaml") if path.is_file()))


def list_config_names(config_dir: Path) -> tuple[str, ...]:
    """Return sorted scenario names inferred from config filenames."""
    return tuple(path.stem for path in list_config_paths(config_dir))


def config_path(config_dir: Path, scenario_name: str) -> Path:
    """Return the YAML path for a scenario name."""
    path = config_dir / f"{scenario_name}.yaml"
    if not path.exists():
        known = ", ".join(list_config_names(config_dir))
        raise ValueError(f"Unknown scenario {scenario_name!r}. Known: {known}")
    return path


def load_named_scenario(config_dir: Path, scenario_name: str) -> ScenarioConfig:
    """Load one named scenario config."""
    return load_scenario_config(str(config_path(config_dir, scenario_name)))


def load_all_named_scenarios(config_dir: Path) -> tuple[ScenarioConfig, ...]:
    """Load all scenario configs under a config directory."""
    return tuple(load_scenario_config(str(path)) for path in list_config_paths(config_dir))


def summarize_scenario(scenario: ScenarioConfig) -> dict[str, Any]:
    """Return a stable, JSON-serializable scenario summary."""
    tier_counts = Counter(provider.tier.value for provider in scenario.providers)
    return {
        "name": scenario.name,
        "description": scenario.description,
        "primary_slo_ms": scenario.primary_slo_ms,
        "slo_thresholds_ms": list(scenario.slo_thresholds_ms),
        "workload": {
            "source": scenario.workload.source,
            "n_requests": scenario.workload.n_requests,
            "duration_seconds": scenario.workload.duration_seconds,
            "arrival_process": scenario.workload.arrival_process,
            "seed": scenario.workload.seed,
        },
        "providers": {
            "count": len(scenario.providers),
            "tiers": dict(sorted(tier_counts.items())),
            "names": [provider.name for provider in scenario.providers],
        },
    }


__all__ = [
    "config_path",
    "list_config_names",
    "list_config_paths",
    "load_all_named_scenarios",
    "load_named_scenario",
    "summarize_scenario",
]
