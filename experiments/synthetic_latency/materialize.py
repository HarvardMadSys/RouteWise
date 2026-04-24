"""Materialize synthetic latency configs into runnable world scenarios."""

from __future__ import annotations

import math

from rwsim.schemas import (
    DistributionConfig,
    ProviderConfig,
    ProviderTier as ConfigProviderTier,
    ScenarioConfig as GenericScenarioConfig,
)
from rwsim.world import LogNormal, ScenarioConfig, ShiftingProvider, SyntheticProvider


def distribution(config: DistributionConfig | None) -> LogNormal | None:
    """Build a world distribution from a generic distribution config."""
    if config is None:
        return None

    params = config.params
    if config.name == "lognormal":
        return LogNormal(mu=float(params["mu"]), sigma=float(params["sigma"]))
    if config.name == "lognormal_p50_sigma":
        return LogNormal(
            mu=math.log(float(params["p50_ms"])),
            sigma=float(params.get("sigma", 0.5)),
        )
    if config.name == "lognormal_p50_p99":
        p50 = float(params["p50_ms"])
        p99 = float(params["p99_ms"])
        mu = math.log(p50)
        sigma = (math.log(p99) - mu) / 2.326
        return LogNormal(mu=mu, sigma=max(sigma, 0.01))

    raise ValueError(f"Unsupported distribution {config.name!r}")


def provider(config: ProviderConfig) -> SyntheticProvider:
    """Build a runnable synthetic provider from generic provider config."""
    if config.tier != ConfigProviderTier.API:
        raise ValueError(f"Synthetic latency provider {config.name!r} must use api tier")
    if len(config.shift_events) > 1:
        raise ValueError(f"Provider {config.name!r} has multiple shift events")

    ttft_dist = distribution(config.ttft_distribution)
    tps_dist = distribution(config.tps_distribution)
    if ttft_dist is None:
        raise ValueError(f"Provider {config.name!r} requires ttft_distribution")
    if tps_dist is None:
        raise ValueError(f"Provider {config.name!r} requires tps_distribution")

    if config.shift_events:
        shift = config.shift_events[0]
        ttft_dist_after = distribution(shift.ttft_distribution)
        if ttft_dist_after is None:
            raise ValueError(f"Provider {config.name!r} requires shifted ttft distribution")
        return ShiftingProvider(
            name=config.name,
            cost_per_token=config.cost_per_token,
            ttft_dist=ttft_dist,
            tps_dist=tps_dist,
            shift_time=shift.time_sec,
            ttft_dist_after=ttft_dist_after,
        )

    return SyntheticProvider(
        name=config.name,
        cost_per_token=config.cost_per_token,
        ttft_dist=ttft_dist,
        tps_dist=tps_dist,
    )


def scenario(config: GenericScenarioConfig) -> ScenarioConfig:
    """Build a runnable world scenario from a generic scenario config."""
    return ScenarioConfig(
        name=config.name,
        description=config.description,
        providers=[provider(item) for item in config.providers],
        n_requests=config.workload.n_requests,
        duration_seconds=config.workload.duration_seconds,
        arrival_process=config.workload.arrival_process,
        primary_slo_ms=config.primary_slo_ms,
        slo_thresholds_ms=list(config.slo_thresholds_ms),
    )


__all__ = ["distribution", "provider", "scenario"]
