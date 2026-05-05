"""Materialize simulation configs into runnable world scenarios."""

from __future__ import annotations

import math

from rwsim.schemas import (
    DistributionConfig,
    ProviderConfig,
    ProviderTier as ConfigProviderTier,
    ScenarioConfig as GenericScenarioConfig,
)
from rwsim.world import (
    ConcurrencyState,
    LogNormal,
    ProviderTier,
    QuotaState,
    ScenarioConfig,
    TieredProvider,
)


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


def provider(config: ProviderConfig) -> TieredProvider:
    """Build a runnable tiered provider from generic provider config."""
    tier = _provider_tier(config.tier)
    ttft_dist = distribution(config.ttft_distribution)
    tps_dist = distribution(config.tps_distribution)
    if ttft_dist is None:
        raise ValueError(f"Provider {config.name!r} requires ttft_distribution")
    if tps_dist is None:
        raise ValueError(f"Provider {config.name!r} requires tps_distribution")
    if len(config.shift_events) > 1:
        raise ValueError(f"Provider {config.name!r} has multiple shift events")

    shift_time = None
    ttft_dist_after = None
    if config.shift_events:
        shift = config.shift_events[0]
        shift_time = shift.time_sec
        ttft_dist_after = distribution(shift.ttft_distribution)

    return TieredProvider(
        name=config.name,
        cost_per_token=config.cost_per_token,
        input_cost_per_token=config.input_cost_per_token,
        output_cost_per_token=config.output_cost_per_token,
        ttft_dist=ttft_dist,
        tps_dist=tps_dist,
        tier=tier,
        quota=_quota_state(config),
        concurrency=_concurrency_state(config),
        service_time_dist=distribution(config.service_time_distribution),
        shift_time=shift_time,
        ttft_dist_after=ttft_dist_after,
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


def _provider_tier(tier: ConfigProviderTier) -> ProviderTier:
    if tier == ConfigProviderTier.API:
        return ProviderTier.S_A
    if tier == ConfigProviderTier.QUOTA:
        return ProviderTier.S_Q
    if tier == ConfigProviderTier.CONCURRENCY:
        return ProviderTier.S_C
    raise ValueError(f"Unsupported provider tier {tier!r}")


def _quota_state(config: ProviderConfig) -> QuotaState | None:
    if config.tier != ConfigProviderTier.QUOTA:
        return None
    if config.quota_size is None:
        raise ValueError(f"Quota provider {config.name!r} requires quota_size")
    return QuotaState(
        size=int(config.quota_size),
        window_sec=float(config.quota_window_sec or 86400.0),
    )


def _concurrency_state(config: ProviderConfig) -> ConcurrencyState | None:
    if config.tier != ConfigProviderTier.CONCURRENCY:
        return None
    if config.concurrency_limit is None:
        raise ValueError(f"Concurrency provider {config.name!r} requires concurrency_limit")
    return ConcurrencyState(limit=int(config.concurrency_limit))


__all__ = ["distribution", "provider", "scenario"]
