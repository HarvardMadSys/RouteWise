"""§3 end-to-end simulator section.

Real-world latency profiles plus real subscription capacity tiers. This section
combines the pieces validated separately by §1 cost-layer and §2 latency /
hedging:

- ``end_to_end_rw3``: one on-demand API provider, one quota provider, and one
  concurrency provider.
- ``end_to_end_3sa_cost_tiers``: three on-demand API providers with the §1
  synthetic cost tiers and dispersed real-world latency profiles, plus the same
  quota and concurrency providers.
- ``end_to_end_rw8``: the selected eight-provider MiniMax M2.5 OpenRouter
  on-demand pool plus the same quota and concurrency providers.

The policy dimension carries the no-hedge / hedging comparison and the ``p``
sweep. Explorer is intentionally disabled in the simulator; drift/profile
freshness belongs to the live-evaluation harness.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from experiments.simulation.common import (
    DEFAULT_CACHED_INPUT_PRICE_FRACTION,
    DEFAULT_SEEDS,
    DEFAULT_WORKLOAD,
    OUTPUT_DIR,
    P_SWEEP,
    WORKLOAD_CHOICES,
    WORKLOAD_COST_ENVELOPE,
    SectionCell,
    SectionCellResult,
    load_workload,
    make_concurrency_provider,
    make_quota_provider,
    make_routewise_presets,
    make_tps_distribution,
    routewise_hedging_policy_name,
    routewise_lp_policy_name,
    run_policy,
    run_section,
    write_json,
)
from experiments.simulation.latency_profiles import (
    DEFAULT_SUBSCRIPTION_PROFILE,
    load_empirical_distribution,
)
from experiments.simulation.provider_profiles import load_provider_pool
from experiments.subscriptions import SubscriptionPlan, load_subscription_plans
from rwsim.world.capacity import ProviderTier
from rwsim.world.providers import TieredProvider
from rwsim.world.scenarios import ScenarioConfig

if TYPE_CHECKING:
    from rwsim.schemas import Request

SECTION_NAME = "end-to-end"
PUBLIC_SCENARIO_TAG = "end_to_end"

RW3_SCENARIO_NAME = "end_to_end_rw3"
COST_TIERED_SCENARIO_NAME = "end_to_end_3sa_cost_tiers"
RW8_SCENARIO_NAME = "end_to_end_rw8"
RW3_POOL_NAME = "rw3"
RW8_POOL_NAME = "rw8"
MINIMAX_M25_RW8_POOL_NAME = "minimax_m25_rw8"

# Conservative default from the §1.3.2 joint q/c sweep:
# outputs/simulation/cost_layer_1_3_2_joint_qc_sweep_q10_20_c8_16_p0_greedy/summary.csv.
# The baseline-optimal point is q=14,c=8; RouteWise p0 remains cheaper there.
DEFAULT_QUOTA_PLAN = "chutes"
DEFAULT_QUOTA_COUNT = 14
DEFAULT_CONCURRENCY_PLAN = "featherless_premium"
DEFAULT_CONCURRENCY_COUNT = 8
DEFAULT_CONCURRENCY_MODEL = "qwen3-235b"
DEFAULT_ROUTEWISE_P_VALUES = P_SWEEP
DEFAULT_SLO_MS = 5000.0

SUBSCRIPTION_LATENCY_PROFILE = DEFAULT_SUBSCRIPTION_PROFILE
_SCENARIO_KWARGS_PRESET_KEY = "__end_to_end_scenario_kwargs__"

_QUOTA_PROFILE_PROVIDER_KEYS = {
    "chutes": "chutes",
    "minimax_subscription_starter": "minimax",
    "minimax_subscription_plus": "minimax",
    "minimax_subscription_max": "minimax",
}
_CONCURRENCY_PROFILE_PROVIDER_KEYS = {
    "featherless_premium": "featherless",
}

# Controlled §3 setting that preserves the §1 cost-layer API prices while using
# real-world Qwen3 latency profiles with clear fast/mid/slow separation.
COST_TIERED_API_SPECS: tuple[tuple[str, str, float, float], ...] = (
    ("api_A_slow_SiliconFlow", "SiliconFlow", 1.0, 5.0),
    ("api_B_mid_Google", "Google", 2.0, 10.0),
    ("api_C_fast_WandB", "WandB", 4.0, 20.0),
)


def list_scenarios() -> tuple[str, ...]:
    """Return §3 scenario names."""
    return (RW3_SCENARIO_NAME, COST_TIERED_SCENARIO_NAME, RW8_SCENARIO_NAME)


def make_scenarios(
    *,
    quota_plan: str = DEFAULT_QUOTA_PLAN,
    quota_count: int = DEFAULT_QUOTA_COUNT,
    concurrency_plan: str = DEFAULT_CONCURRENCY_PLAN,
    concurrency_count: int = DEFAULT_CONCURRENCY_COUNT,
    model: str = DEFAULT_CONCURRENCY_MODEL,
    prefix_cache_enabled: bool = False,
    cached_input_price_fraction: float = DEFAULT_CACHED_INPUT_PRICE_FRACTION,
) -> dict[str, ScenarioConfig]:
    """Build all §3 scenarios keyed by scenario name."""
    return {
        name: make_scenario(
            name,
            quota_plan=quota_plan,
            quota_count=quota_count,
            concurrency_plan=concurrency_plan,
            concurrency_count=concurrency_count,
            model=model,
            prefix_cache_enabled=prefix_cache_enabled,
            cached_input_price_fraction=cached_input_price_fraction,
        )
        for name in list_scenarios()
    }


def make_scenario(
    name: str,
    *,
    quota_plan: str = DEFAULT_QUOTA_PLAN,
    quota_count: int = DEFAULT_QUOTA_COUNT,
    concurrency_plan: str = DEFAULT_CONCURRENCY_PLAN,
    concurrency_count: int = DEFAULT_CONCURRENCY_COUNT,
    model: str = DEFAULT_CONCURRENCY_MODEL,
    prefix_cache_enabled: bool = False,
    cached_input_price_fraction: float = DEFAULT_CACHED_INPUT_PRICE_FRACTION,
) -> ScenarioConfig:
    """Build one §3 scenario by name."""
    if name == RW3_SCENARIO_NAME:
        return _with_prefix_cache_config(
            _make_end_to_end_scenario(
                scenario_name=name,
                pool_name=RW3_POOL_NAME,
                api_provider_limit=1,
                quota_plan_id=quota_plan,
                quota_count=quota_count,
                concurrency_plan_id=concurrency_plan,
                concurrency_count=concurrency_count,
                model=model,
            ),
            enabled=prefix_cache_enabled,
            cached_input_price_fraction=cached_input_price_fraction,
        )
    if name == COST_TIERED_SCENARIO_NAME:
        return _with_prefix_cache_config(
            _make_end_to_end_scenario(
                scenario_name=name,
                pool_name=RW8_POOL_NAME,
                api_provider_limit=None,
                quota_plan_id=quota_plan,
                quota_count=quota_count,
                concurrency_plan_id=concurrency_plan,
                concurrency_count=concurrency_count,
                model=model,
                api_specs=COST_TIERED_API_SPECS,
                api_price_source="cost_layer_synthetic_tiers",
            ),
            enabled=prefix_cache_enabled,
            cached_input_price_fraction=cached_input_price_fraction,
        )
    if name == RW8_SCENARIO_NAME:
        return _with_prefix_cache_config(
            _make_end_to_end_scenario(
                scenario_name=name,
                pool_name=MINIMAX_M25_RW8_POOL_NAME,
                api_provider_limit=None,
                quota_plan_id=quota_plan,
                quota_count=quota_count,
                concurrency_plan_id=concurrency_plan,
                concurrency_count=concurrency_count,
                model=model,
            ),
            enabled=prefix_cache_enabled,
            cached_input_price_fraction=cached_input_price_fraction,
        )
    known = ", ".join(list_scenarios())
    raise ValueError(f"unknown end-to-end scenario {name!r}; known: {known}")


def _with_prefix_cache_config(
    scenario: ScenarioConfig,
    *,
    enabled: bool,
    cached_input_price_fraction: float,
) -> ScenarioConfig:
    """Apply provider-local prefix-cache accounting to one §3 scenario."""
    if cached_input_price_fraction < 0.0:
        raise ValueError(
            "cached input price fraction must be non-negative, got "
            f"{cached_input_price_fraction!r}"
        )
    use_fraction_fallback = (
        scenario.metadata.get("api_price_source") != "metadata_openrouter_price"
    )
    scenario.metadata["prefix_cache_enabled"] = bool(enabled)
    scenario.metadata["cached_input_price_fraction"] = (
        cached_input_price_fraction if use_fraction_fallback else None
    )
    scenario.metadata["cached_input_price_source"] = (
        "fraction_of_input_price"
        if use_fraction_fallback
        else "metadata_openrouter_input_cache_read"
    )
    if not enabled:
        return scenario
    for provider in scenario.providers:
        if provider.tier == ProviderTier.S_A and provider.input_cost_per_token is not None:
            if provider.cached_input_cost_per_token is not None:
                continue
            if not use_fraction_fallback:
                continue
            provider.cached_input_cost_per_token = (
                provider.input_cost_per_token * cached_input_price_fraction
            )
    return scenario


def policies_for_section(
    p_values: tuple[float, ...] = DEFAULT_ROUTEWISE_P_VALUES,
) -> tuple[str, ...]:
    """Return §3 baselines plus LP-only and LP+hedging p sweeps."""
    return (
        "greedy_cost",
        "greedy_latency",
        "random",
        *(routewise_lp_policy_name(value) for value in p_values),
        *(routewise_hedging_policy_name(value) for value in p_values),
    )


def make_policy_presets(
    p_values: tuple[float, ...] = DEFAULT_ROUTEWISE_P_VALUES,
) -> dict[str, dict[str, Any]]:
    """Build section-local presets with configured empirical profiles."""
    presets = make_routewise_presets(
        p_values=p_values,
        include_hedging=True,
        cost_envelope=WORKLOAD_COST_ENVELOPE,
    )
    for preset in presets.values():
        if preset.get("policy") != "RouteWisePolicy":
            continue
        params = dict(preset.get("params", {}))
        params["latency_profile_mode"] = "configured"
        params["explorer"] = False
        params["slo_ms"] = DEFAULT_SLO_MS
        preset["params"] = params
    return presets


def run_end_to_end_policy(
    scenario: ScenarioConfig,
    requests: list[Request],
    policy_name: str,
    *,
    presets: dict[str, dict[str, Any]],
    seed: int,
    retain_records: bool = True,
):
    """Run one §3 policy with the scenario SLO injected into RouteWise."""
    return run_policy(
        scenario,
        requests,
        policy_name,
        presets=_presets_for_scenario_slo(presets, scenario),
        seed=seed,
        retain_records=retain_records,
    )


def run_end_to_end_cell(
    cell: SectionCell,
    presets: dict[str, dict[str, Any]],
    workload_dataset: str,
    duration_sec: float | None,
    max_requests: int | None,
    retain_records: bool,
) -> SectionCellResult:
    """Run one §3 simulation cell in a worker process."""
    scenario = make_scenario(
        cell.scenario_name,
        **_scenario_kwargs_from_presets(presets),
    )
    requests = load_workload(
        dataset=workload_dataset,
        duration_sec=duration_sec,
        max_requests=max_requests,
    )
    run = run_end_to_end_policy(
        scenario,
        requests,
        cell.policy,
        presets=presets,
        seed=cell.seed,
        retain_records=retain_records,
    )
    return SectionCellResult(
        scenario_name=cell.scenario_name,
        policy=cell.policy,
        seed=cell.seed,
        run=run,
    )


def _make_end_to_end_scenario(
    *,
    scenario_name: str,
    pool_name: str,
    api_provider_limit: int | None,
    quota_plan_id: str,
    quota_count: int,
    concurrency_plan_id: str,
    concurrency_count: int,
    model: str,
    api_specs: tuple[tuple[str, str, float, float], ...] | None = None,
    api_price_source: str | None = None,
) -> ScenarioConfig:
    plans = load_subscription_plans()
    quota_plan = _require_plan(plans, quota_plan_id, tier="quota")
    concurrency_plan = _require_plan(plans, concurrency_plan_id, tier="concurrency")
    _validate_plan_count(quota_plan, quota_count, field="quota_count")
    _validate_plan_count(concurrency_plan, concurrency_count, field="concurrency_count")
    _validate_end_to_end_eligibility(quota_plan)
    _validate_end_to_end_eligibility(concurrency_plan)
    resolution = concurrency_plan.resolve_model_class_with_cost(model)
    if resolution is None:
        raise ValueError(
            f"concurrency plan {concurrency_plan.plan_id!r} does not support model {model!r}"
        )

    provider_pool = load_provider_pool(pool_name)
    if api_specs is None:
        api_items = list(provider_pool.providers)
        if api_provider_limit is not None:
            api_items = api_items[:api_provider_limit]
        if not api_items:
            raise ValueError(f"real-world pool {pool_name!r} did not yield API providers")
        api_providers = [
            _make_empirical_api_provider(
                item.name,
                item.ttft_dist,
                input_per_m=item.input_per_m,
                output_per_m=item.output_per_m,
                cached_input_per_m=item.cached_input_per_m,
            )
            for item in api_items
        ]
        api_latency_provider_names = [item.name for item in api_items]
        resolved_api_price_source = api_price_source or provider_pool.price_source
        api_price_tiers_per_m = [
            {
                "provider": provider.name,
                "input_per_m": provider.effective_input_cost_per_token * 1_000_000.0,
                "output_per_m": provider.effective_output_cost_per_token * 1_000_000.0,
                "cached_input_per_m": (
                    None
                    if provider.cached_input_cost_per_token is None
                    else provider.cached_input_cost_per_token * 1_000_000.0
                ),
            }
            for provider in api_providers
        ]
    else:
        api_pool_by_name = provider_pool.by_name()
        api_providers = []
        api_latency_provider_names = []
        api_price_tiers_per_m = []
        for api_name, latency_provider_name, input_per_m, output_per_m in api_specs:
            try:
                ttft_dist = api_pool_by_name[latency_provider_name].ttft_dist
            except KeyError as exc:
                known = ", ".join(sorted(api_pool_by_name))
                raise ValueError(
                    f"cost-tiered API latency provider {latency_provider_name!r} "
                    f"not found in pool {pool_name!r}; known: {known}"
                ) from exc
            api_providers.append(
                _make_empirical_api_provider(
                    latency_provider_name,
                    ttft_dist,
                    provider_name=api_name,
                    input_per_m=input_per_m,
                    output_per_m=output_per_m,
                )
            )
            api_latency_provider_names.append(latency_provider_name)
            api_price_tiers_per_m.append(
                {
                    "provider": api_name,
                    "latency_provider": latency_provider_name,
                    "input_per_m": input_per_m,
                    "output_per_m": output_per_m,
                }
            )
        resolved_api_price_source = api_price_source or "scenario_api_specs"

    quota_provider = make_quota_provider(
        f"{quota_plan.plan_id}_quota",
        plan=quota_plan,
        subscription_count=quota_count,
        latency_family="heavy_tail",
    )
    concurrency_provider = make_concurrency_provider(
        f"{concurrency_plan.plan_id}_concurrency",
        plan=concurrency_plan,
        concurrency_count=concurrency_count,
        model=model,
        latency_family="heavy_tail",
    )
    quota_profile_key = _subscription_latency_profile_key(quota_plan.plan_id)
    concurrency_profile_key = _subscription_latency_profile_key(concurrency_plan.plan_id)
    if quota_profile_key is not None:
        quota_provider.ttft_dist = load_empirical_distribution(
            SUBSCRIPTION_LATENCY_PROFILE,
            quota_profile_key,
        )
    if concurrency_profile_key is not None:
        concurrency_provider.ttft_dist = load_empirical_distribution(
            SUBSCRIPTION_LATENCY_PROFILE,
            concurrency_profile_key,
        )

    providers = [quota_provider, concurrency_provider, *api_providers]

    quota_window_text = ", ".join(
        f"{window.quota_requests * quota_count:g}/{window.quota_window_sec:g}s"
        for window in quota_plan.quota_windows
    )
    capacity_units = int(concurrency_plan.concurrency_allotment or 0) * int(
        concurrency_count
    )
    description = (
        f"§3 end-to-end {pool_name.upper()}: {len(api_providers)} empirical "
        f"OpenRouter API provider(s), {quota_plan.display_name} x{quota_count} "
        f"({quota_window_text}), {concurrency_plan.display_name} "
        f"x{concurrency_count} for model {model!r}; SLO={DEFAULT_SLO_MS:.0f}ms."
    )
    return ScenarioConfig(
        name=scenario_name,
        description=description,
        providers=providers,
        arrival_process="trace",
        primary_slo_ms=DEFAULT_SLO_MS,
        metadata={
            "public_scenario": PUBLIC_SCENARIO_TAG,
            "artifact_label": scenario_name,
            "real_world_pool": pool_name,
            "api_provider_count": len(api_providers),
            "api_provider_names": [provider.name for provider in api_providers],
            "api_latency_provider_names": api_latency_provider_names,
            "api_price_source": resolved_api_price_source,
            "api_price_tiers_per_m": api_price_tiers_per_m,
            "subscription_plan": quota_plan.plan_id,
            "subscription_plan_display_name": quota_plan.display_name,
            "subscription_count": quota_count,
            "concurrency_plan": concurrency_plan.plan_id,
            "concurrency_plan_display_name": concurrency_plan.display_name,
            "concurrency_count": concurrency_count,
            "model": model,
            "model_class": resolution.model_class,
            "model_concurrency_cost": resolution.cost,
            "concurrency_capacity_units": capacity_units,
            "effective_concurrency_limit": (
                concurrency_provider.concurrency.limit
                if concurrency_provider.concurrency is not None
                else 0
            ),
            "latency_profile": SUBSCRIPTION_LATENCY_PROFILE,
            "quota_latency_profile_provider": quota_profile_key,
            "concurrency_latency_profile_provider": concurrency_profile_key,
            "api_latency_family": "real_world",
            "slo_ms": DEFAULT_SLO_MS,
            "quota_windows": [
                {
                    "name": window.name,
                    "quota_requests": window.quota_requests,
                    "quota_window_sec": window.quota_window_sec,
                    "aggregate_quota_requests": window.quota_requests * quota_count,
                }
                for window in quota_plan.quota_windows
            ],
        },
    )


def _make_empirical_api_provider(
    latency_provider_name: str,
    ttft_dist,
    *,
    provider_name: str | None = None,
    input_per_m: float,
    output_per_m: float,
    cached_input_per_m: float | None = None,
) -> TieredProvider:
    return TieredProvider(
        name=provider_name or f"api_{latency_provider_name}",
        cost_per_token=input_per_m / 1_000_000.0,
        input_cost_per_token=input_per_m / 1_000_000.0,
        output_cost_per_token=output_per_m / 1_000_000.0,
        cached_input_cost_per_token=(
            None if cached_input_per_m is None else cached_input_per_m / 1_000_000.0
        ),
        ttft_dist=ttft_dist,
        tps_dist=make_tps_distribution(),
        tier=ProviderTier.S_A,
    )


def _require_plan(
    plans: dict[str, SubscriptionPlan],
    plan_id: str,
    *,
    tier: str,
) -> SubscriptionPlan:
    try:
        plan = plans[plan_id]
    except KeyError as exc:
        known = ", ".join(sorted(plans))
        raise ValueError(f"unknown subscription plan {plan_id!r}; known plans: {known}") from exc
    if plan.tier != tier:
        raise ValueError(f"plan {plan_id!r} must be tier={tier!r}, got {plan.tier!r}")
    return plan


def _validate_plan_count(plan: SubscriptionPlan, count: int, *, field: str) -> None:
    del plan
    if count <= 0:
        raise ValueError(f"{field} must be > 0, got {count}")


def _validate_end_to_end_eligibility(plan: SubscriptionPlan) -> None:
    if "end_to_end" not in plan.eligible_sections:
        raise ValueError(
            f"plan {plan.plan_id!r} is not eligible for end-to-end runs"
        )


def _subscription_latency_profile_key(plan_id: str) -> str | None:
    return _QUOTA_PROFILE_PROVIDER_KEYS.get(
        plan_id,
        _CONCURRENCY_PROFILE_PROVIDER_KEYS.get(plan_id),
    )


def _presets_for_scenario_slo(
    presets: dict[str, dict[str, Any]],
    scenario: ScenarioConfig,
) -> dict[str, dict[str, Any]]:
    patched: dict[str, dict[str, Any]] = {}
    for name, preset in presets.items():
        params = dict(preset.get("params", {}))
        if preset.get("policy") == "RouteWisePolicy":
            params["slo_ms"] = float(scenario.primary_slo_ms)
        patched[name] = {**preset, "params": params}
    return patched


def _make_serial_runner(policy_name: str, presets: dict[str, dict[str, Any]]):
    def runner(
        scenario: ScenarioConfig,
        requests: list[Request],
        seed: int,
        *,
        retain_records: bool = True,
    ):
        return run_end_to_end_policy(
            scenario,
            requests,
            policy_name,
            presets=presets,
            seed=seed,
            retain_records=retain_records,
        )

    return runner


def _enrich_rows_with_end_to_end_metadata(
    rows: list[dict[str, Any]],
    scenarios: dict[str, ScenarioConfig],
    presets: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        scenario = scenarios[row["scenario"]]
        meta = dict(scenario.metadata or {})
        preset = presets.get(row["policy"], {})
        params = dict(preset.get("params", {}))
        merged = dict(row)
        merged.update(
            {
                "real_world_pool": meta.get("real_world_pool"),
                "api_provider_count": meta.get("api_provider_count"),
                "api_provider_names": meta.get("api_provider_names"),
                "api_latency_provider_names": meta.get("api_latency_provider_names"),
                "api_price_source": meta.get("api_price_source"),
                "api_price_tiers_per_m": meta.get("api_price_tiers_per_m"),
                "prefix_cache_enabled": meta.get("prefix_cache_enabled"),
                "cached_input_price_fraction": meta.get("cached_input_price_fraction"),
                "cached_input_price_source": meta.get("cached_input_price_source"),
                "slo_ms": meta.get("slo_ms"),
                "routewise_p": params.get("p"),
                "hedging_enabled": bool(params.get("hedging", False)),
                "explorer_enabled": bool(params.get("explorer", False)),
                "latency_profile_mode": params.get("latency_profile_mode"),
            }
        )
        enriched.append(merged)
    return enriched


_END_TO_END_CSV_FIELDNAMES: tuple[str, ...] = (
    "scenario",
    "public_scenario",
    "artifact_label",
    "real_world_pool",
    "api_provider_count",
    "api_provider_names",
    "api_latency_provider_names",
    "api_price_source",
    "api_price_tiers_per_m",
    "prefix_cache_enabled",
    "cached_input_price_fraction",
    "cached_input_price_source",
    "subscription_plan",
    "subscription_plan_display_name",
    "subscription_count",
    "concurrency_plan",
    "concurrency_plan_display_name",
    "concurrency_count",
    "model",
    "model_class",
    "model_concurrency_cost",
    "concurrency_capacity_units",
    "effective_concurrency_limit",
    "latency_profile",
    "quota_latency_profile_provider",
    "concurrency_latency_profile_provider",
    "api_latency_family",
    "policy",
    "routewise_p",
    "hedging_enabled",
    "explorer_enabled",
    "latency_profile_mode",
    "seeds",
    "n_requests",
    "mean_ttft_ms",
    "p50_ms",
    "p90_ms",
    "p99_ms",
    "slo_ms",
    "slo_violation_rate",
    "hedge_rate",
    "provider_mix",
    "tier_mix",
    "mean_api_cost_usd",
    "mean_total_cost_usd",
    "api_cost_usd",
    "total_cost_usd",
    "subscription_fixed_cost_usd",
    "trace_days",
    "percentile_source",
    "histogram_bins",
)


def _write_end_to_end_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the §3-specific csv view."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_END_TO_END_CSV_FIELDNAMES))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(row[key], sort_keys=True)
                        if isinstance(row.get(key), (dict, list))
                        else row.get(key)
                    )
                    for key in _END_TO_END_CSV_FIELDNAMES
                }
            )


def main(argv: list[str] | None = None) -> int:
    """Run the §3 end-to-end simulator section."""
    parser = argparse.ArgumentParser(
        prog="routewise simulator end-to-end",
        description=__doc__,
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=list(list_scenarios()),
        help="Scenario to run. Repeat to run multiple. Defaults to all §3 scenarios.",
    )
    parser.add_argument(
        "--policy",
        action="append",
        help="Policy to run. Repeat to run multiple. Defaults to baselines + p sweep.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        help=f"Seed to run. Repeat to run multiple. Defaults to {DEFAULT_SEEDS}.",
    )
    parser.add_argument(
        "--p",
        type=float,
        action="append",
        dest="p_values",
        help=f"RouteWise p value. Repeat to sweep. Defaults to {DEFAULT_ROUTEWISE_P_VALUES}.",
    )
    parser.add_argument(
        "--quota-plan",
        default=DEFAULT_QUOTA_PLAN,
        help=f"Quota subscription plan id. Defaults to {DEFAULT_QUOTA_PLAN}.",
    )
    parser.add_argument(
        "--quota-count",
        type=int,
        default=DEFAULT_QUOTA_COUNT,
        help=f"Quota subscription count. Defaults to {DEFAULT_QUOTA_COUNT}.",
    )
    parser.add_argument(
        "--concurrency-plan",
        default=DEFAULT_CONCURRENCY_PLAN,
        help=f"Concurrency subscription plan id. Defaults to {DEFAULT_CONCURRENCY_PLAN}.",
    )
    parser.add_argument(
        "--concurrency-count",
        type=int,
        default=DEFAULT_CONCURRENCY_COUNT,
        help=f"Concurrency subscription count. Defaults to {DEFAULT_CONCURRENCY_COUNT}.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_CONCURRENCY_MODEL,
        help=f"Concurrency model id or trace alias. Defaults to {DEFAULT_CONCURRENCY_MODEL}.",
    )
    parser.add_argument(
        "--workload",
        default=DEFAULT_WORKLOAD,
        choices=WORKLOAD_CHOICES,
        help="Trace workload to replay.",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        help="Optional trace truncation for smoke runs.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        help="Optional request-count truncation for smoke runs.",
    )
    parser.add_argument(
        "--prefix-cache-enabled",
        action="store_true",
        help=(
            "Enable provider-local prefix-cache cost discounts for API providers "
            "when requests carry prefix_id/session metadata."
        ),
    )
    parser.add_argument(
        "--cached-input-price-fraction",
        type=float,
        default=DEFAULT_CACHED_INPUT_PRICE_FRACTION,
        help=(
            "Cached-input price as a fraction of each API provider's normal input "
            f"price. Defaults to {DEFAULT_CACHED_INPUT_PRICE_FRACTION}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "end_to_end",
        help="Directory for metadata.json, summary.json, and summary.csv.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel scenario-policy-seed cells to run. Defaults to 1.",
    )

    args = parser.parse_args(argv)
    p_values = tuple(args.p_values) if args.p_values else DEFAULT_ROUTEWISE_P_VALUES
    selected_scenarios = tuple(args.scenario) if args.scenario else list_scenarios()
    scenarios = {
        name: make_scenario(
            name,
            quota_plan=args.quota_plan,
            quota_count=args.quota_count,
            concurrency_plan=args.concurrency_plan,
            concurrency_count=args.concurrency_count,
            model=args.model,
            prefix_cache_enabled=args.prefix_cache_enabled,
            cached_input_price_fraction=args.cached_input_price_fraction,
        )
        for name in selected_scenarios
    }
    presets = make_policy_presets(p_values)
    policies = tuple(args.policy) if args.policy else policies_for_section(p_values)
    unknown = [policy for policy in policies if policy not in presets]
    if unknown:
        known = ", ".join(sorted(presets))
        raise SystemExit(f"unknown end-to-end policy {unknown[0]!r}; known policies: {known}")
    presets = _with_scenario_kwargs(
        presets,
        {
            "quota_plan": args.quota_plan,
            "quota_count": args.quota_count,
            "concurrency_plan": args.concurrency_plan,
            "concurrency_count": args.concurrency_count,
            "model": args.model,
            "prefix_cache_enabled": args.prefix_cache_enabled,
            "cached_input_price_fraction": args.cached_input_price_fraction,
        },
    )

    rows = run_section(
        section_name=SECTION_NAME,
        scenarios=scenarios,
        policies=policies,
        presets=presets,
        seeds=tuple(args.seed) if args.seed else DEFAULT_SEEDS,
        section_runners={
            policy: _make_serial_runner(policy, presets) for policy in policies
        },
        workload_dataset=args.workload,
        duration_sec=args.duration_sec,
        max_requests=args.max_requests,
        output_dir=args.output_dir,
        jobs=args.jobs,
        parallel_cell_runner=run_end_to_end_cell,
    )
    enriched_rows = _enrich_rows_with_end_to_end_metadata(rows, scenarios, presets)
    write_json(args.output_dir / "summary.json", enriched_rows)
    _write_end_to_end_summary_csv(args.output_dir / "summary.csv", enriched_rows)
    print(
        json.dumps(
            {
                "section": SECTION_NAME,
                "rows": len(enriched_rows),
                "output_dir": str(args.output_dir),
            }
        )
    )
    return 0


def _with_scenario_kwargs(
    presets: dict[str, dict[str, Any]],
    kwargs: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    patched = dict(presets)
    patched[_SCENARIO_KWARGS_PRESET_KEY] = {
        "policy": "SectionMetadata",
        "params": dict(kwargs),
    }
    return patched


def _scenario_kwargs_from_presets(presets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    payload = presets.get(_SCENARIO_KWARGS_PRESET_KEY, {})
    params = payload.get("params", {}) if isinstance(payload, dict) else {}
    return dict(params)


__all__ = [
    "COST_TIERED_API_SPECS",
    "COST_TIERED_SCENARIO_NAME",
    "DEFAULT_CONCURRENCY_COUNT",
    "DEFAULT_CONCURRENCY_MODEL",
    "DEFAULT_CONCURRENCY_PLAN",
    "DEFAULT_QUOTA_COUNT",
    "DEFAULT_QUOTA_PLAN",
    "DEFAULT_ROUTEWISE_P_VALUES",
    "PUBLIC_SCENARIO_TAG",
    "RW3_SCENARIO_NAME",
    "RW8_SCENARIO_NAME",
    "SECTION_NAME",
    "list_scenarios",
    "main",
    "make_policy_presets",
    "make_scenario",
    "make_scenarios",
    "policies_for_section",
    "run_end_to_end_cell",
    "run_end_to_end_policy",
]


if __name__ == "__main__":
    raise SystemExit(main())
