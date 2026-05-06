"""Cost-layer simulator section."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from experiments.simulation.common import (
    COST_LAYER_P50_MS,
    COST_RATIO_PER_MILLION,
    DEFAULT_SEEDS,
    DEFAULT_WORKLOAD,
    OUTPUT_COST_MULTIPLIER,
    OUTPUT_DIR,
    P_SWEEP,
    SectionCell,
    SectionCellResult,
    load_workload,
    make_api_provider,
    make_concurrency_provider,
    make_quota_provider,
    make_routewise_presets,
    make_tps_distribution,
    routewise_lp_policy_name,
    run_policy,
    run_section,
)
from experiments.simulation.latency_profiles import load_pooled_distribution
from experiments.subscriptions import SubscriptionPlan, load_subscription_plans
from rwsim.metrics import PerRequestRecord, Run, RunAggregator, Status
from rwsim.world.capacity import ProviderTier
from rwsim.world.providers import TieredProvider
from rwsim.world.scenarios import ScenarioConfig

if TYPE_CHECKING:
    from rwsim.schemas import Request

SECTION_NAME = "cost-layer"
OFFLINE_POLICY = "offline"
REAL_WORLD_SCENARIO = "cost_layer_real_world"
QUOTA_SCENARIO = "quota"

_SYNTHETIC_FAMILIES = ("uniform", "normal", "heavy_tail")
_CONCURRENCY_LIMIT_PER_PROVIDER = 8
_SCENARIO_NAMES = (
    "cost_layer_uniform",
    "cost_layer_normal",
    "cost_layer_heavy_tail",
    REAL_WORLD_SCENARIO,
    QUOTA_SCENARIO,
    "cost_layer_concurrency_c1",
    "cost_layer_concurrency_c2",
    "cost_layer_concurrency_c3",
    "cost_layer_concurrency_c4",
)


def list_scenarios() -> tuple[str, ...]:
    """Return all cost-layer scenario names."""
    return _SCENARIO_NAMES


def make_scenarios(
    *,
    subscription_plans: tuple[str, ...] = ("chutes",),
    subscription_counts: tuple[int, ...] | None = None,
) -> dict[str, ScenarioConfig]:
    """Build runnable cost-layer scenarios keyed by artifact label."""
    scenarios: dict[str, ScenarioConfig] = {}
    for name in _SCENARIO_NAMES:
        if name == QUOTA_SCENARIO:
            scenarios.update(
                _make_quota_plan_scenarios(
                    subscription_plans=subscription_plans,
                    subscription_counts=subscription_counts,
                )
            )
        else:
            scenario = make_scenario(name)
            scenarios[scenario.name] = scenario
    return scenarios


def make_scenario(
    name: str,
    *,
    subscription_plan: str = "chutes",
    subscription_count: int = 1,
) -> ScenarioConfig:
    """Build one cost-layer scenario by name."""
    if name.startswith("cost_layer_") and name.removeprefix("cost_layer_") in _SYNTHETIC_FAMILIES:
        return _make_api_cost_scenario(name.removeprefix("cost_layer_"))
    if name == REAL_WORLD_SCENARIO:
        return _make_real_world_api_cost_scenario()
    if name == QUOTA_SCENARIO:
        return _make_quota_scenario_for_plan(subscription_plan, subscription_count)
    parsed_quota = _parse_quota_artifact_label(name)
    if parsed_quota is not None:
        plan_id, count = parsed_quota
        return _make_quota_scenario_for_plan(plan_id, count)
    if name.startswith("cost_layer_concurrency_c"):
        return _make_concurrency_scenario(
            _parse_positive_suffix(name, "cost_layer_concurrency_c")
        )
    known = ", ".join(_SCENARIO_NAMES)
    raise ValueError(f"unknown cost-layer scenario {name!r}; known scenarios: {known}")


def policies_for_section(
    p_values: tuple[float, ...] = P_SWEEP,
) -> tuple[str, ...]:
    """Return policies relevant to cost-layer experiments."""
    return (
        "greedy_cost",
        "random",
        OFFLINE_POLICY,
        *(routewise_lp_policy_name(value) for value in p_values),
    )


def _parse_positive_suffix(name: str, prefix: str) -> int:
    suffix = name.removeprefix(prefix)
    try:
        value = int(suffix)
    except ValueError as exc:
        raise ValueError(f"invalid cost-layer scenario name {name!r}") from exc
    if value not in range(1, 5):
        raise ValueError(f"cost-layer scenario count must be in [1, 4], got {value}")
    return value


def quota_artifact_label(plan_id: str, subscription_count: int) -> str:
    """Return the stable artifact label for one quota plan/count."""
    return f"quota__plan={plan_id}__n={subscription_count}"


def _parse_quota_artifact_label(name: str) -> tuple[str, int] | None:
    prefix = "quota__plan="
    marker = "__n="
    if not name.startswith(prefix) or marker not in name:
        return None
    plan_part, count_part = name.removeprefix(prefix).rsplit(marker, 1)
    try:
        count = int(count_part)
    except ValueError as exc:
        raise ValueError(f"invalid quota artifact label {name!r}") from exc
    return plan_part, count


def _make_quota_plan_scenarios(
    *,
    subscription_plans: tuple[str, ...],
    subscription_counts: tuple[int, ...] | None,
) -> dict[str, ScenarioConfig]:
    plans = load_subscription_plans()
    scenarios: dict[str, ScenarioConfig] = {}
    for plan_id in subscription_plans:
        try:
            plan = plans[plan_id]
        except KeyError as exc:
            known = ", ".join(sorted(plans))
            raise ValueError(
                f"unknown subscription plan {plan_id!r}; known plans: {known}"
            ) from exc
        counts = subscription_counts or plan.subscription_counts
        _validate_subscription_counts(plan, counts)
        for count in counts:
            scenario = _make_quota_scenario_for_plan(plan.plan_id, count)
            scenarios[scenario.name] = scenario
    return scenarios


def _validate_subscription_counts(
    plan: SubscriptionPlan,
    counts: tuple[int, ...],
) -> None:
    allowed = set(plan.subscription_counts)
    invalid = [count for count in counts if count not in allowed]
    if invalid:
        allowed_text = ", ".join(str(count) for count in plan.subscription_counts)
        raise ValueError(
            f"subscription count {invalid[0]} is not allowed for plan "
            f"{plan.plan_id!r}; allowed counts: {allowed_text}"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the cost-layer simulator section."""
    parser = argparse.ArgumentParser(
        prog="routewise simulator cost-layer",
        description=__doc__,
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=list_scenarios(),
        help="Scenario to run. Repeat to run multiple. Defaults to all.",
    )
    parser.add_argument(
        "--policy",
        action="append",
        help="Policy to run. Repeat to run multiple. Defaults to this section's policy set.",
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
        help=f"RouteWise p value. Repeat to sweep. Defaults to {P_SWEEP}.",
    )
    parser.add_argument(
        "--subscription-plan",
        help="Quota scenario subscription plan id, e.g. chutes.",
    )
    parser.add_argument(
        "--subscription-plans",
        help="Comma-separated quota scenario plan ids, e.g. chutes,minimax_subscription_plus.",
    )
    parser.add_argument(
        "--subscription-count",
        type=int,
        help="Quota scenario subscription count.",
    )
    parser.add_argument(
        "--subscription-counts",
        help="Comma-separated quota scenario subscription counts, e.g. 1,2,3,4.",
    )
    parser.add_argument(
        "--workload",
        default=DEFAULT_WORKLOAD,
        choices=("sharegpt_burstgpt", "burstgpt"),
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
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "cost_layer",
        help="Directory for metadata.json, summary.json, and summary.csv.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel scenario-policy-seed cells to run. Defaults to 1.",
    )

    args = parser.parse_args(argv)
    p_values = tuple(args.p_values) if args.p_values else P_SWEEP
    selected_public_scenarios = tuple(args.scenario) if args.scenario else _SCENARIO_NAMES
    has_quota = QUOTA_SCENARIO in selected_public_scenarios
    subscription_plan_ids = _parse_subscription_plan_args(
        args.subscription_plan,
        args.subscription_plans,
        has_quota=has_quota,
    )
    subscription_counts = _parse_subscription_count_args(
        args.subscription_count,
        args.subscription_counts,
        has_quota=has_quota,
    )
    scenarios = make_scenarios(
        subscription_plans=subscription_plan_ids,
        subscription_counts=subscription_counts,
    )
    if args.scenario:
        selected: dict[str, ScenarioConfig] = {}
        for name in selected_public_scenarios:
            if name == QUOTA_SCENARIO:
                selected.update(
                    {
                        scenario_name: scenario
                        for scenario_name, scenario in scenarios.items()
                        if scenario.metadata.get("public_scenario") == QUOTA_SCENARIO
                    }
                )
            else:
                selected[name] = scenarios[name]
        scenarios = selected

    presets = make_routewise_presets(p_values=p_values, include_hedging=False)
    policies = tuple(args.policy) if args.policy else policies_for_section(p_values)
    section_runners = {OFFLINE_POLICY: run_offline_policy}
    known_policies = set(presets) | set(section_runners)
    unknown = [policy for policy in policies if policy not in known_policies]
    if unknown:
        known = ", ".join(sorted(known_policies))
        raise SystemExit(f"unknown cost-layer policy {unknown[0]!r}; known policies: {known}")

    rows = run_section(
        section_name=SECTION_NAME,
        scenarios=scenarios,
        policies=policies,
        presets=presets,
        seeds=tuple(args.seed) if args.seed else DEFAULT_SEEDS,
        section_runners=section_runners,
        workload_dataset=args.workload,
        duration_sec=args.duration_sec,
        max_requests=args.max_requests,
        output_dir=args.output_dir,
        jobs=args.jobs,
        parallel_cell_runner=run_cost_layer_cell,
    )
    print(json.dumps({"section": SECTION_NAME, "rows": len(rows), "output_dir": str(args.output_dir)}))
    return 0


def _parse_subscription_plan_args(
    single: str | None,
    multiple: str | None,
    *,
    has_quota: bool,
) -> tuple[str, ...]:
    if single and multiple:
        raise SystemExit("use either --subscription-plan or --subscription-plans, not both")
    if not has_quota:
        if single or multiple:
            raise SystemExit("subscription plan flags require --scenario quota")
        return ("chutes",)
    values = _split_csv(multiple) if multiple else ((single,) if single else ("chutes",))
    plans = load_subscription_plans()
    unknown = [value for value in values if value not in plans]
    if unknown:
        known = ", ".join(sorted(plans))
        raise SystemExit(f"unknown subscription plan {unknown[0]!r}; known plans: {known}")
    return values


def _parse_subscription_count_args(
    single: int | None,
    multiple: str | None,
    *,
    has_quota: bool,
) -> tuple[int, ...] | None:
    if single is not None and multiple:
        raise SystemExit("use either --subscription-count or --subscription-counts, not both")
    if not has_quota:
        if single is not None or multiple:
            raise SystemExit("subscription count flags require --scenario quota")
        return None
    if multiple:
        return tuple(int(value) for value in _split_csv(multiple))
    if single is not None:
        return (int(single),)
    return None


def _split_csv(value: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    if not values:
        raise SystemExit("comma-separated argument cannot be empty")
    return values


def run_cost_layer_cell(
    cell: SectionCell,
    presets: dict[str, dict[str, object]],
    workload_dataset: str,
    duration_sec: float | None,
    max_requests: int | None,
    retain_records: bool,
) -> SectionCellResult:
    """Run one cost-layer simulation cell in a worker process."""
    scenario = make_scenario(cell.scenario_name)
    requests = load_workload(
        dataset=workload_dataset,
        duration_sec=duration_sec,
        max_requests=max_requests,
    )
    if cell.policy == OFFLINE_POLICY:
        run = run_offline_policy(
            scenario,
            requests,
            cell.seed,
            retain_records=retain_records,
        )
    else:
        run = run_policy(
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


def run_offline_policy(
    scenario: ScenarioConfig,
    requests: list[Request],
    seed: int,
    retain_records: bool = True,
) -> Run:
    """Run the cost-layer offline baseline with full trace knowledge."""
    del seed
    assignments = _offline_assignments(scenario, requests)
    aggregator = RunAggregator(
        policy=OFFLINE_POLICY,
        scenario_name=scenario.name,
        source="simulation",
        retain_records=retain_records,
    )
    for request in requests:
        aggregator.observe(
            _offline_record(
                scenario=scenario,
                request=request,
                provider=assignments[request.id],
            )
        )
    return aggregator.finalize()


def _make_api_cost_scenario(family: str) -> ScenarioConfig:
    providers = [
        make_api_provider(
            "api_cheap",
            cost_per_million_tokens=COST_RATIO_PER_MILLION[0],
            latency_family=family,
        ),
        make_api_provider(
            "api_mid",
            cost_per_million_tokens=COST_RATIO_PER_MILLION[1],
            latency_family=family,
        ),
        make_api_provider(
            "api_expensive",
            cost_per_million_tokens=COST_RATIO_PER_MILLION[2],
            latency_family=family,
        ),
    ]
    return ScenarioConfig(
        name=f"cost_layer_{family}",
        description=(
            "Cost-layer on-demand API scenario: identical TTFT distribution "
            f"({family}, P50={COST_LAYER_P50_MS:.0f}ms), input cost ratio $1/$2/$4 "
            "and output cost ratio $5/$10/$20 per million tokens."
        ),
        providers=providers,
        arrival_process="trace",
        primary_slo_ms=2000.0,
    )


def _make_real_world_api_cost_scenario() -> ScenarioConfig:
    ttft_dist = load_pooled_distribution("rw8_pooled")
    providers = [
        TieredProvider(
            name="api_cheap",
            cost_per_token=COST_RATIO_PER_MILLION[0] / 1_000_000.0,
            input_cost_per_token=COST_RATIO_PER_MILLION[0] / 1_000_000.0,
            output_cost_per_token=(
                COST_RATIO_PER_MILLION[0] * OUTPUT_COST_MULTIPLIER / 1_000_000.0
            ),
            ttft_dist=ttft_dist,
            tps_dist=make_tps_distribution(),
            tier=ProviderTier.S_A,
        ),
        TieredProvider(
            name="api_mid",
            cost_per_token=COST_RATIO_PER_MILLION[1] / 1_000_000.0,
            input_cost_per_token=COST_RATIO_PER_MILLION[1] / 1_000_000.0,
            output_cost_per_token=(
                COST_RATIO_PER_MILLION[1] * OUTPUT_COST_MULTIPLIER / 1_000_000.0
            ),
            ttft_dist=ttft_dist,
            tps_dist=make_tps_distribution(),
            tier=ProviderTier.S_A,
        ),
        TieredProvider(
            name="api_expensive",
            cost_per_token=COST_RATIO_PER_MILLION[2] / 1_000_000.0,
            input_cost_per_token=COST_RATIO_PER_MILLION[2] / 1_000_000.0,
            output_cost_per_token=(
                COST_RATIO_PER_MILLION[2] * OUTPUT_COST_MULTIPLIER / 1_000_000.0
            ),
            ttft_dist=ttft_dist,
            tps_dist=make_tps_distribution(),
            tier=ProviderTier.S_A,
        ),
    ]
    return ScenarioConfig(
        name=REAL_WORLD_SCENARIO,
        description=(
            "Cost-layer real-world API scenario: identical pooled Qwen3/OpenRouter "
            "TTFT distribution (rw8_pooled), input cost ratio $1/$2/$4 and "
            "output cost ratio $5/$10/$20 per million tokens."
        ),
        providers=providers,
        arrival_process="trace",
        primary_slo_ms=2000.0,
    )


def _make_quota_scenario_for_plan(
    plan_id: str,
    subscription_count: int,
) -> ScenarioConfig:
    plans = load_subscription_plans()
    try:
        plan = plans[plan_id]
    except KeyError as exc:
        known = ", ".join(sorted(plans))
        raise ValueError(f"unknown subscription plan {plan_id!r}; known plans: {known}") from exc
    _validate_subscription_counts(plan, (subscription_count,))
    if "cost_layer_quota" not in plan.eligible_sections:
        raise ValueError(
            f"subscription plan {plan.plan_id!r} is not eligible for cost-layer quota runs"
        )

    label = quota_artifact_label(plan.plan_id, subscription_count)
    providers = [
        make_quota_provider(
            f"{plan.plan_id}_quota",
            plan=plan,
            subscription_count=subscription_count,
        ),
        make_api_provider(
            "api_cheap",
            cost_per_million_tokens=COST_RATIO_PER_MILLION[0],
            latency_family="heavy_tail",
        ),
        make_api_provider(
            "api_mid",
            cost_per_million_tokens=COST_RATIO_PER_MILLION[1],
            latency_family="heavy_tail",
        ),
        make_api_provider(
            "api_expensive",
            cost_per_million_tokens=COST_RATIO_PER_MILLION[2],
            latency_family="heavy_tail",
        ),
    ]
    quota_window_text = ", ".join(
        f"{window.quota_requests * subscription_count:g}/{window.quota_window_sec:g}s"
        for window in plan.quota_windows
    )
    return ScenarioConfig(
        name=label,
        description=(
            f"Cost-layer quota scenario: {plan.display_name} x{subscription_count} "
            f"({quota_window_text}) as one aggregate quota provider, fixed "
            "cheap/mid/expensive on-demand fallback providers, "
            f"LogNormal P50={COST_LAYER_P50_MS:.0f}ms."
        ),
        providers=providers,
        arrival_process="trace",
        primary_slo_ms=2000.0,
        metadata={
            "public_scenario": QUOTA_SCENARIO,
            "artifact_label": label,
            "subscription_plan": plan.plan_id,
            "subscription_plan_display_name": plan.display_name,
            "subscription_count": subscription_count,
            "quota_windows": [
                {
                    "name": window.name,
                    "quota_requests": window.quota_requests,
                    "quota_window_sec": window.quota_window_sec,
                    "aggregate_quota_requests": (
                        window.quota_requests * subscription_count
                    ),
                }
                for window in plan.quota_windows
            ],
        },
    )


def _make_concurrency_scenario(count: int) -> ScenarioConfig:
    providers = [
        make_concurrency_provider(
            f"concurrency_{idx + 1}",
            concurrency_limit=_CONCURRENCY_LIMIT_PER_PROVIDER,
        )
        for idx in range(count)
    ]
    api_count = 2 if count == 1 else 1
    providers.extend(
        make_api_provider(
            f"api_fallback_{idx + 1}",
            cost_per_million_tokens=COST_RATIO_PER_MILLION[-1],
            latency_family="heavy_tail",
        )
        for idx in range(api_count)
    )
    return ScenarioConfig(
        name=f"cost_layer_concurrency_c{count}",
        description=(
            f"Cost-layer concurrency scenario: {count} concurrency provider(s), "
            f"{api_count} on-demand fallback provider(s), LogNormal P50={COST_LAYER_P50_MS:.0f}ms."
        ),
        providers=providers,
        arrival_process="trace",
        primary_slo_ms=2000.0,
    )


def _offline_assignments(
    scenario: ScenarioConfig,
    requests: list[Request],
) -> dict[int, TieredProvider]:
    providers = list(scenario.providers)
    api_provider = _cheapest_api_provider(providers)
    if api_provider is None:
        raise ValueError(f"offline baseline requires at least one API provider: {scenario.name}")

    quota_providers = [provider for provider in providers if provider.tier == ProviderTier.S_Q]
    concurrency_providers = [
        provider for provider in providers if provider.tier == ProviderTier.S_C
    ]

    assignments = {request.id: api_provider for request in requests}
    if quota_providers:
        assignments.update(_quota_assignments(requests, quota_providers, api_provider))
    if concurrency_providers:
        assignments.update(
            _concurrency_assignments(requests, concurrency_providers, api_provider)
        )
    return assignments


def _quota_assignments(
    requests: list[Request],
    quota_providers: list[TieredProvider],
    api_provider: TieredProvider,
) -> dict[int, TieredProvider]:
    assignments: dict[int, TieredProvider] = {}
    trace_start = float(requests[0].timestamp) if requests else 0.0
    quota_usage: dict[str, dict[tuple[int, int], int]] = {
        provider.name: {} for provider in quota_providers
    }
    ranked = sorted(
        requests,
        key=lambda request: (
            _api_cost(api_provider, request),
            -(request.timestamp),
            -request.id,
        ),
        reverse=True,
    )
    for request in ranked:
        for provider in quota_providers:
            quota_windows = _provider_quota_windows(provider)
            if _quota_windows_have_capacity(
                quota_usage[provider.name],
                quota_windows,
                request,
                trace_start,
            ):
                _charge_quota_windows(
                    quota_usage[provider.name],
                    quota_windows,
                    request,
                    trace_start,
                )
                assignments[request.id] = provider
                break
    return assignments


def _provider_quota_windows(provider: TieredProvider) -> tuple[tuple[int, float], ...]:
    quota = provider.quota
    if quota is None:
        return ()
    if hasattr(quota, "windows"):
        return tuple((window.size, window.window_sec) for window in quota.windows)
    return ((quota.size, quota.window_sec),)


def _quota_windows_have_capacity(
    usage: dict[tuple[int, int], int],
    quota_windows: tuple[tuple[int, float], ...],
    request: Request,
    trace_start: float,
) -> bool:
    return all(
        usage.get((index, _quota_window_id(request, window_sec, trace_start)), 0)
        < size
        for index, (size, window_sec) in enumerate(quota_windows)
    )


def _charge_quota_windows(
    usage: dict[tuple[int, int], int],
    quota_windows: tuple[tuple[int, float], ...],
    request: Request,
    trace_start: float,
) -> None:
    for index, (_, window_sec) in enumerate(quota_windows):
        key = (index, _quota_window_id(request, window_sec, trace_start))
        usage[key] = usage.get(key, 0) + 1


def _quota_window_id(request: Request, window_sec: float, trace_start: float) -> int:
    return int((float(request.timestamp) - trace_start) // float(window_sec))


def _concurrency_assignments(
    requests: list[Request],
    concurrency_providers: list[TieredProvider],
    api_provider: TieredProvider,
) -> dict[int, TieredProvider]:
    slots: list[tuple[TieredProvider, list[tuple[float, float]]]] = [
        (provider, [])
        for provider in concurrency_providers
        for _ in range(provider.concurrency.limit if provider.concurrency is not None else 0)
    ]
    assignments: dict[int, TieredProvider] = {}
    ranked = sorted(
        requests,
        key=lambda request: (
            _api_cost(api_provider, request),
            -(request.timestamp),
            -request.id,
        ),
        reverse=True,
    )
    for request in ranked:
        start = float(request.timestamp)
        end = start + _offline_service_time_sec(concurrency_providers[0], request)
        for provider, intervals in slots:
            if all(end <= lo or start >= hi for lo, hi in intervals):
                intervals.append((start, end))
                assignments[request.id] = provider
                break
    return assignments


def _offline_record(
    *,
    scenario: ScenarioConfig,
    request: Request,
    provider: TieredProvider,
) -> PerRequestRecord:
    ttft_ms = provider.true_p50_ms(float(request.timestamp))
    cost_usd = provider.marginal_cost_for_request(request, float(request.timestamp))
    return PerRequestRecord(
        request_id=str(request.id),
        elapsed_sec=float(request.timestamp),
        policy=OFFLINE_POLICY,
        prompt_tokens=int(request.request_tokens),
        completion_tokens_budget=(
            request.estimated_response_tokens
            if request.estimated_response_tokens is not None
            else request.response_tokens
        ),
        completion_tokens_actual=request.response_tokens,
        primary_provider=provider.name,
        primary_tier=provider.tier.value,
        final_provider=provider.name,
        final_tier=provider.tier.value,
        ttft_ms=ttft_ms,
        primary_local_ttft_ms=ttft_ms,
        slo_ms=scenario.primary_slo_ms,
        slo_violated=ttft_ms > scenario.primary_slo_ms,
        total_cost_usd=cost_usd,
        primary_cost_usd=cost_usd,
        status=Status.SUCCESS,
        metadata={"offline_cost_baseline": True},
    )


def _cheapest_api_provider(providers: list[TieredProvider]) -> TieredProvider | None:
    api_providers = [provider for provider in providers if provider.tier == ProviderTier.S_A]
    if not api_providers:
        return None
    return min(
        api_providers,
        key=lambda provider: (
            provider.effective_input_cost_per_token,
            provider.effective_output_cost_per_token,
            provider.name,
        ),
    )


def _api_cost(provider: TieredProvider, request: Request) -> float:
    return provider.marginal_cost_for_request(request, float(request.timestamp))


def _offline_service_time_sec(provider: TieredProvider, request: Request) -> float:
    ttft_ms = provider.true_p50_ms(float(request.timestamp))
    tps = max(provider.tps_dist.p50(), 1.0)
    response_tokens = request.response_tokens or 1
    return (ttft_ms + (response_tokens / tps) * 1000.0) / 1000.0


__all__ = [
    "OFFLINE_POLICY",
    "QUOTA_SCENARIO",
    "SECTION_NAME",
    "list_scenarios",
    "make_scenario",
    "make_scenarios",
    "policies_for_section",
    "quota_artifact_label",
    "run_offline_policy",
]


if __name__ == "__main__":
    raise SystemExit(main())
