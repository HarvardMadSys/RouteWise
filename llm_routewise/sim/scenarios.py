"""Generic scenario config builder.

Concrete paper scenarios belong in ``experiments/*/configs/*.yaml``. This
module only converts generic dictionaries into shared schema objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llm_routewise.schemas import (
    DistributionConfig,
    ProviderConfig,
    ProviderTier,
    ScenarioConfig,
    ShiftEvent,
    WorkloadConfig,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def load_scenario_config(path: str) -> ScenarioConfig:
    """Load a YAML scenario config and build a generic scenario."""
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to load scenario configs.") from exc

    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return build_scenario(payload)


def _distribution(value: DistributionConfig | Mapping[str, Any] | None) -> DistributionConfig | None:
    if value is None or isinstance(value, DistributionConfig):
        return value
    params = dict(value)
    name = str(params.pop("name"))
    return DistributionConfig(name=name, params={key: float(val) for key, val in params.items()})


def _shift_event(value: ShiftEvent | Mapping[str, Any]) -> ShiftEvent:
    if isinstance(value, ShiftEvent):
        return value
    return ShiftEvent(
        time_sec=float(value["time_sec"]),
        ttft_distribution=_distribution(value["ttft_distribution"]),
    )


def _provider(value: ProviderConfig | Mapping[str, Any]) -> ProviderConfig:
    if isinstance(value, ProviderConfig):
        return value
    return ProviderConfig(
        name=str(value["name"]),
        tier=ProviderTier(value["tier"]),
        cost_per_token=float(value.get("cost_per_token", 0.0)),
        ttft_distribution=_distribution(value["ttft_distribution"]),
        tps_distribution=_distribution(value.get("tps_distribution")),
        input_cost_per_token=(
            None
            if value.get("input_cost_per_token") is None
            else float(value["input_cost_per_token"])
        ),
        output_cost_per_token=(
            None
            if value.get("output_cost_per_token") is None
            else float(value["output_cost_per_token"])
        ),
        cached_input_cost_per_token=(
            None
            if value.get("cached_input_cost_per_token") is None
            else float(value["cached_input_cost_per_token"])
        ),
        quota_size=value.get("quota_size"),
        quota_window_sec=value.get("quota_window_sec"),
        concurrency_limit=value.get("concurrency_limit"),
        service_time_distribution=_distribution(value.get("service_time_distribution")),
        shift_events=tuple(_shift_event(item) for item in value.get("shift_events", ())),
        metadata=dict(value.get("metadata", {})),
    )


def _workload(value: WorkloadConfig | Mapping[str, Any]) -> WorkloadConfig:
    if isinstance(value, WorkloadConfig):
        return value
    return WorkloadConfig(
        n_requests=int(value["n_requests"]),
        duration_seconds=float(value["duration_seconds"]),
        arrival_process=str(value.get("arrival_process", "trace")),
        seed=int(value.get("seed", 0)),
        start_time=float(value.get("start_time", 0.0)),
        input_tokens=int(value.get("input_tokens", 100)),
        output_token_distribution=_distribution(value.get("output_token_distribution")),
        source=str(value.get("source", "trace")),
        metadata=dict(value.get("metadata", {})),
    )


def build_scenario(config: ScenarioConfig | Mapping[str, Any]) -> ScenarioConfig:
    """Build a generic scenario from a loaded config dictionary."""
    if isinstance(config, ScenarioConfig):
        return config
    return ScenarioConfig(
        name=str(config["name"]),
        description=str(config.get("description", "")),
        providers=tuple(_provider(item) for item in config.get("providers", ())),
        workload=_workload(config["workload"]),
        primary_slo_ms=float(config["primary_slo_ms"]),
        slo_thresholds_ms=tuple(
            float(value)
            for value in config.get(
                "slo_thresholds_ms",
                (1000.0, 2000.0, 3000.0, 5000.0),
            )
        ),
        metadata=dict(config.get("metadata", {})),
    )


__all__ = ["build_scenario", "load_scenario_config"]
