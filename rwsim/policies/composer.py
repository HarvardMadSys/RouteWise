"""Pipeline alias registry for migrated RouteWise policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StageSpec:
    """Config reference for one policy stage."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyPipelineSpec:
    """Declarative policy pipeline config."""

    value_estimator: StageSpec
    cost_router: StageSpec
    latency_router: StageSpec
    hedger: StageSpec
    notes: str = ""


def _stage(name: str, **params: Any) -> StageSpec:
    return StageSpec(name=name, params=params)


STRATEGY_PIPELINE_ALIASES: dict[str, PolicyPipelineSpec] = {
    "cheapest_fixed": PolicyPipelineSpec(
        value_estimator=_stage("none"),
        cost_router=_stage("fixed", mode="cheapest"),
        latency_router=_stage("none"),
        hedger=_stage("none"),
    ),
    "fastest_fixed": PolicyPipelineSpec(
        value_estimator=_stage("none"),
        cost_router=_stage("fixed", mode="fastest"),
        latency_router=_stage("none"),
        hedger=_stage("none"),
    ),
    "round_robin": PolicyPipelineSpec(
        value_estimator=_stage("none"),
        cost_router=_stage("round_robin"),
        latency_router=_stage("none"),
        hedger=_stage("none"),
    ),
    "oracle_per_window": PolicyPipelineSpec(
        value_estimator=_stage("oracle"),
        cost_router=_stage("oracle_window", window_sec=900.0),
        latency_router=_stage("none"),
        hedger=_stage("none"),
    ),
    "lp_mix": PolicyPipelineSpec(
        value_estimator=_stage("fixed"),
        cost_router=_stage("lp_mix", target_cdf=0.99),
        latency_router=_stage("lp_slo_constraint"),
        hedger=_stage("none"),
    ),
    "lp_hedge": PolicyPipelineSpec(
        value_estimator=_stage("fixed"),
        cost_router=_stage("lp_mix", target_cdf=0.99),
        latency_router=_stage("lp_slo_constraint"),
        hedger=_stage("smart_economic", cost_ratio=0.1),
        notes="Preserves current SmartHedger behavior during migration.",
    ),
    "lp_explorer": PolicyPipelineSpec(
        value_estimator=_stage("fixed", hedge_as_probe=True, dedicated_probe=True),
        cost_router=_stage("lp_explorer", target_cdf=0.99),
        latency_router=_stage("lp_slo_constraint"),
        hedger=_stage("smart_economic", cost_ratio=0.1),
    ),
    "lp_explorer_no_probe": PolicyPipelineSpec(
        value_estimator=_stage("fixed", hedge_as_probe=True, dedicated_probe=False),
        cost_router=_stage("lp_explorer", target_cdf=0.99),
        latency_router=_stage("lp_slo_constraint"),
        hedger=_stage("smart_economic", cost_ratio=0.1),
    ),
    "v2_only": PolicyPipelineSpec(
        value_estimator=_stage("fixed"),
        cost_router=_stage("v2_cost"),
        latency_router=_stage("p50_band", margin=0.10),
        hedger=_stage("none"),
    ),
    "v2_p50_hedge": PolicyPipelineSpec(
        value_estimator=_stage("fixed"),
        cost_router=_stage("v2_cost"),
        latency_router=_stage("p50_band", margin=0.10),
        hedger=_stage("smart_economic", cost_ratio=0.1),
        notes="Preserves current SmartHedger behavior during migration.",
    ),
    "v2_explorer": PolicyPipelineSpec(
        value_estimator=_stage("fixed", hedge_as_probe=True, dedicated_probe=True),
        cost_router=_stage("v2_explorer"),
        latency_router=_stage("p50_band", margin=0.10),
        hedger=_stage("smart_economic", cost_ratio=0.1),
    ),
    "v2_explorer_no_probe": PolicyPipelineSpec(
        value_estimator=_stage("fixed", hedge_as_probe=True, dedicated_probe=False),
        cost_router=_stage("v2_explorer"),
        latency_router=_stage("p50_band", margin=0.10),
        hedger=_stage("smart_economic", cost_ratio=0.1),
    ),
    "two_layer": PolicyPipelineSpec(
        value_estimator=_stage("fixed"),
        cost_router=_stage("two_layer", tier_order=["quota", "concurrency", "api"]),
        latency_router=_stage("tier_gate"),
        hedger=_stage("none"),
    ),
    "joint_nohedge": PolicyPipelineSpec(
        value_estimator=_stage("fixed"),
        cost_router=_stage("cheapest_effective"),
        latency_router=_stage("p95_slo_filter", safety_margin=0.8),
        hedger=_stage("none"),
    ),
    "joint_hedge": PolicyPipelineSpec(
        value_estimator=_stage("fixed"),
        cost_router=_stage("cheapest_effective"),
        latency_router=_stage("p95_slo_filter", safety_margin=0.8),
        hedger=_stage("smart_economic", slo_budget_floor=0.5),
    ),
    "joint_p50band_nohedge": PolicyPipelineSpec(
        value_estimator=_stage("fixed"),
        cost_router=_stage("cheapest_effective"),
        latency_router=_stage("p50_band", margin=0.10),
        hedger=_stage("none"),
    ),
    "joint_p50band_hedge": PolicyPipelineSpec(
        value_estimator=_stage("fixed"),
        cost_router=_stage("cheapest_effective"),
        latency_router=_stage("p50_band", margin=0.10),
        hedger=_stage("smart_economic", slo_budget_floor=0.5),
    ),
}


def available_aliases() -> tuple[str, ...]:
    """Return the known strategy aliases."""
    return tuple(sorted(STRATEGY_PIPELINE_ALIASES))


def get_pipeline_alias(strategy_name: str) -> PolicyPipelineSpec:
    """Return the target pipeline config for a current strategy name."""
    try:
        return STRATEGY_PIPELINE_ALIASES[strategy_name]
    except KeyError as exc:
        known = ", ".join(available_aliases())
        raise ValueError(f"Unknown pipeline alias {strategy_name!r}. Known: {known}") from exc


__all__ = [
    "PolicyPipelineSpec",
    "STRATEGY_PIPELINE_ALIASES",
    "StageSpec",
    "available_aliases",
    "get_pipeline_alias",
]
