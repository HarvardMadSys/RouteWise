"""Load priced provider pools from empirical latency profile metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from experiments.simulation.latency_profiles import (
    DEFAULT_POOLS_PATH,
    load_pool,
    load_profile_config,
)
from rwsim.world.empirical import EmpiricalDistribution


@dataclass(frozen=True)
class ProviderPoolEntry:
    """One provider's empirical latency distribution and request prices."""

    name: str
    ttft_dist: EmpiricalDistribution
    input_per_m: float
    output_per_m: float
    cached_input_per_m: float | None
    price_source: str


@dataclass(frozen=True)
class ProviderPool:
    """A configured provider pool with resolved empirical distributions and prices."""

    name: str
    profile_name: str
    price_source: str
    providers: tuple[ProviderPoolEntry, ...]

    def by_name(self) -> dict[str, ProviderPoolEntry]:
        return {provider.name: provider for provider in self.providers}


def load_provider_pool(
    pool_name: str,
    *,
    config_path: str | Path = DEFAULT_POOLS_PATH,
) -> ProviderPool:
    """Load one configured pool with latency distributions and resolved prices."""
    config_path = Path(config_path)
    config = load_profile_config(config_path)
    pool = _pool_config(config, pool_name, config_path)
    profile_name = str(pool.get("profile", ""))
    distributions = load_pool(pool_name, config_path=config_path)
    prices, price_source = _load_prices(config, pool, tuple(distributions), config_path)
    entries = tuple(
        ProviderPoolEntry(
            name=provider_name,
            ttft_dist=distributions[provider_name],
            input_per_m=prices[provider_name][0],
            output_per_m=prices[provider_name][1],
            cached_input_per_m=prices[provider_name][2],
            price_source=price_source,
        )
        for provider_name in distributions
    )
    return ProviderPool(
        name=pool_name,
        profile_name=profile_name,
        price_source=price_source,
        providers=entries,
    )


def _load_prices(
    config: dict[str, Any],
    pool: dict[str, Any],
    provider_names: tuple[str, ...],
    config_path: Path,
) -> tuple[dict[str, tuple[float, float, float | None]], str]:
    pricing = pool.get("pricing")
    if not isinstance(pricing, dict):
        raise ValueError(f"provider pool {pool!r} must define pricing: {config_path}")
    source = str(pricing.get("source", ""))
    if source == "static":
        return _load_static_prices(pricing, provider_names, config_path), source
    if source == "metadata_openrouter_price":
        return _load_openrouter_prices_from_metadata(
            config,
            pool,
            provider_names,
            config_path=config_path,
        ), source
    raise ValueError(f"unsupported pool pricing source {source!r}: {config_path}")


def _load_static_prices(
    pricing: dict[str, Any],
    provider_names: tuple[str, ...],
    config_path: Path,
) -> dict[str, tuple[float, float, float | None]]:
    default_input = pricing.get("default_input_per_m")
    default_output = pricing.get("default_output_per_m")
    overrides = pricing.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError(f"static pricing overrides must be a mapping: {config_path}")
    prices: dict[str, tuple[float, float, float | None]] = {}
    for provider_name in provider_names:
        override = overrides.get(provider_name, {})
        if override is None:
            override = {}
        if not isinstance(override, dict):
            raise ValueError(
                f"static pricing override for {provider_name!r} must be a mapping: "
                f"{config_path}"
            )
        input_per_m = override.get("input_per_m", default_input)
        output_per_m = override.get("output_per_m", default_output)
        if input_per_m is None or output_per_m is None:
            raise ValueError(
                f"static pricing missing input/output price for {provider_name!r}: "
                f"{config_path}"
            )
        prices[provider_name] = (float(input_per_m), float(output_per_m), None)
    return prices


def _load_openrouter_prices_from_metadata(
    config: dict[str, Any],
    pool: dict[str, Any],
    provider_names: tuple[str, ...],
    *,
    config_path: Path,
) -> dict[str, tuple[float, float, float | None]]:
    metadata = _load_pool_profile_metadata(config, pool, config_path)
    provider_metadata = metadata.get("providers", {})
    if not isinstance(provider_metadata, dict):
        raise ValueError(f"profile metadata providers must be a mapping: {config_path}")

    prices: dict[str, tuple[float, float, float | None]] = {}
    missing: list[str] = []
    for provider_name in provider_names:
        raw_provider = provider_metadata.get(provider_name, {})
        price = raw_provider.get("openrouter_price") if isinstance(raw_provider, dict) else None
        if not isinstance(price, dict):
            missing.append(provider_name)
            continue
        input_per_m = price.get("input_price_per_m")
        output_per_m = price.get("output_price_per_m")
        if input_per_m is None or output_per_m is None:
            missing.append(provider_name)
            continue
        cached_input_per_m = price.get("input_cache_read_price_per_m")
        prices[provider_name] = (
            float(input_per_m),
            float(output_per_m),
            None if cached_input_per_m is None else float(cached_input_per_m),
        )

    if missing:
        raise ValueError(
            "missing OpenRouter price metadata for providers "
            f"{missing!r}: {config_path}"
        )
    return prices


def _load_pool_profile_metadata(
    config: dict[str, Any],
    pool: dict[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    profile = _profile_config(config, pool, config_path)
    metadata = profile.get("metadata")
    if metadata is None:
        raise ValueError(f"profile for pool must define metadata: {config_path}")
    metadata_path = Path(metadata)
    if not metadata_path.is_absolute():
        metadata_path = config_path.parent / metadata_path
    with metadata_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"profile metadata must be a mapping: {metadata_path}")
    return payload


def _pool_config(
    config: dict[str, Any],
    pool_name: str,
    config_path: Path,
) -> dict[str, Any]:
    pools = config.get("pools", {})
    if not isinstance(pools, dict):
        raise ValueError(f"profile config pools must be a mapping: {config_path}")
    try:
        pool = pools[pool_name]
    except KeyError as exc:
        known = ", ".join(sorted(pools))
        raise KeyError(f"unknown provider pool {pool_name!r}; known pools: {known}") from exc
    if not isinstance(pool, dict):
        raise ValueError(f"provider pool {pool_name!r} must be a mapping: {config_path}")
    return pool


def _profile_config(
    config: dict[str, Any],
    pool: dict[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    profiles = config.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError(f"profile config profiles must be a mapping: {config_path}")
    profile_name = pool.get("profile")
    if not isinstance(profile_name, str):
        raise ValueError(f"provider pool must name a profile: {config_path}")
    try:
        profile = profiles[profile_name]
    except KeyError as exc:
        known = ", ".join(sorted(profiles))
        raise KeyError(f"unknown profile {profile_name!r}; known profiles: {known}") from exc
    if not isinstance(profile, dict):
        raise ValueError(f"profile {profile_name!r} must be a mapping: {config_path}")
    return profile


__all__ = [
    "ProviderPool",
    "ProviderPoolEntry",
    "load_provider_pool",
]
