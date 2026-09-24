"""Load committed real-world latency profile artifacts for simulator sections."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rwsim.world.empirical import EmpiricalDistribution

PROFILE_DIR = Path(__file__).resolve().with_name("latency_profiles")
DEFAULT_POOLS_PATH = PROFILE_DIR / "pools.yaml"
DEFAULT_SUBSCRIPTION_PROFILE = "minimax_m25_subscriptions"


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
    config, pool = _load_pool_entry(pool_name, config_path)

    artifact = _entry_artifact_path(config, pool, config_path)
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

    artifact = _entry_artifact_path(config, pool, config_path)
    providers = tuple(pool["source_providers"])
    return EmpiricalDistribution.pooled_from_npz(artifact, providers, label=pooled_name)


def load_empirical_profile_metadata(
    profile_name: str | Path = DEFAULT_SUBSCRIPTION_PROFILE,
) -> dict[str, Any]:
    """Load a committed empirical profile metadata sidecar."""
    metadata_path = _metadata_path(profile_name)
    with metadata_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"profile metadata must be a mapping: {metadata_path}")
    if "artifact" not in payload:
        raise ValueError(f"profile metadata must define artifact: {metadata_path}")
    return payload


def load_empirical_distribution(
    profile_name: str | Path,
    provider_name: str,
) -> EmpiricalDistribution:
    """Load one provider distribution from a committed empirical profile."""
    metadata_path = _metadata_path(profile_name)
    metadata = load_empirical_profile_metadata(metadata_path)
    artifact = _artifact_path(metadata, metadata_path)
    return EmpiricalDistribution.from_npz(artifact, provider_name)


def load_empirical_profile(
    profile_name: str | Path = DEFAULT_SUBSCRIPTION_PROFILE,
    *,
    provider_names: tuple[str, ...] | None = None,
) -> dict[str, EmpiricalDistribution]:
    """Load provider-specific distributions from a committed empirical profile."""
    metadata_path = _metadata_path(profile_name)
    metadata = load_empirical_profile_metadata(metadata_path)
    providers = metadata.get("providers", {})
    if not isinstance(providers, dict):
        raise ValueError(f"profile metadata providers must be a mapping: {metadata_path}")
    names = tuple(provider_names) if provider_names is not None else tuple(providers)
    artifact = _artifact_path(metadata, metadata_path)
    return {
        provider_name: EmpiricalDistribution.from_npz(artifact, provider_name)
        for provider_name in names
    }


def _artifact_path(config: dict[str, Any], config_path: Path) -> Path:
    artifact = Path(config["artifact"])
    if artifact.is_absolute():
        return artifact
    return config_path.parent / artifact


def _load_pool_entry(
    pool_name: str,
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_profile_config(config_path)
    pools = config.get("pools", {})
    try:
        pool = pools[pool_name]
    except KeyError as exc:
        known = ", ".join(sorted(pools))
        raise KeyError(f"unknown real-world profile pool {pool_name!r}; known pools: {known}") from exc
    if not isinstance(pool, dict):
        raise ValueError(f"profile pool {pool_name!r} must be a mapping: {config_path}")
    return config, pool


def _entry_artifact_path(
    config: dict[str, Any],
    entry: dict[str, Any],
    config_path: Path,
) -> Path:
    return _artifact_path(_entry_profile_config(config, entry, config_path), config_path)


def _entry_metadata_path(
    config: dict[str, Any],
    entry: dict[str, Any],
    config_path: Path,
) -> Path | None:
    metadata = _entry_profile_config(config, entry, config_path).get("metadata")
    if metadata is None:
        return None
    path = Path(metadata)
    if path.is_absolute():
        return path
    return config_path.parent / path


def _entry_profile_config(
    config: dict[str, Any],
    entry: dict[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    if "profile" not in entry:
        return entry if "artifact" in entry else config

    profiles = config.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError(f"profile config must define profiles mapping: {config_path}")
    profile_name = entry["profile"]
    try:
        profile = profiles[profile_name]
    except KeyError as exc:
        known = ", ".join(sorted(profiles))
        raise KeyError(
            f"unknown latency profile {profile_name!r}; known profiles: {known}"
        ) from exc
    if not isinstance(profile, dict):
        raise ValueError(f"latency profile {profile_name!r} must be a mapping: {config_path}")
    return profile


def _metadata_path(profile_name: str | Path) -> Path:
    path = Path(profile_name)
    if path.suffix:
        return path
    return PROFILE_DIR / f"{path}.json"


__all__ = [
    "DEFAULT_POOLS_PATH",
    "DEFAULT_SUBSCRIPTION_PROFILE",
    "PROFILE_DIR",
    "load_empirical_distribution",
    "load_empirical_profile",
    "load_empirical_profile_metadata",
    "load_pool",
    "load_pooled_distribution",
    "load_profile_config",
]
