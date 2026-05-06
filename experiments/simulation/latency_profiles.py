"""Load committed real-world latency profile artifacts for simulator sections."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rwsim.world.empirical import EmpiricalDistribution

PROFILE_DIR = Path(__file__).resolve().with_name("latency_profiles")
DEFAULT_POOLS_PATH = PROFILE_DIR / "pools.yaml"


def load_profile_config(path: str | Path = DEFAULT_POOLS_PATH) -> dict[str, Any]:
    """Load the real-world latency profile pool configuration."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"profile config must be a mapping: {config_path}")
    return payload


def load_pool(
    pool_name: str,
    *,
    config_path: str | Path = DEFAULT_POOLS_PATH,
) -> dict[str, EmpiricalDistribution]:
    """Load provider-specific real-world latency distributions for one named pool."""
    config_path = Path(config_path)
    config = load_profile_config(config_path)
    pools = config.get("pools", {})
    try:
        pool = pools[pool_name]
    except KeyError as exc:
        known = ", ".join(sorted(pools))
        raise KeyError(f"unknown real-world profile pool {pool_name!r}; known pools: {known}") from exc

    artifact = _artifact_path(config, config_path)
    providers = tuple(pool["providers"])
    return {
        provider_name: EmpiricalDistribution.from_npz(artifact, provider_name)
        for provider_name in providers
    }


def load_pooled_distribution(
    pooled_name: str,
    *,
    config_path: str | Path = DEFAULT_POOLS_PATH,
) -> EmpiricalDistribution:
    """Load one pooled real-world latency distribution from configured providers."""
    config_path = Path(config_path)
    config = load_profile_config(config_path)
    pooled = config.get("pooled", {})
    try:
        pool = pooled[pooled_name]
    except KeyError as exc:
        known = ", ".join(sorted(pooled))
        raise KeyError(
            f"unknown pooled real-world profile {pooled_name!r}; known pooled profiles: {known}"
        ) from exc

    artifact = _artifact_path(config, config_path)
    providers = tuple(pool["source_providers"])
    return EmpiricalDistribution.pooled_from_npz(artifact, providers, label=pooled_name)


def _artifact_path(config: dict[str, Any], config_path: Path) -> Path:
    artifact = Path(config["artifact"])
    if artifact.is_absolute():
        return artifact
    return config_path.parent / artifact


__all__ = [
    "DEFAULT_POOLS_PATH",
    "PROFILE_DIR",
    "load_pool",
    "load_pooled_distribution",
    "load_profile_config",
]
