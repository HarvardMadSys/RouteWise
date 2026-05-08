"""Cost-layer simulator section."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import quote, unquote

from experiments.simulation.common import (
    COST_LAYER_LATENCY_ANCHOR_MS,
    COST_RATIO_PER_MILLION,
    DEFAULT_SEEDS,
    DEFAULT_WORKLOAD,
    OUTPUT_COST_MULTIPLIER,
    OUTPUT_DIR,
    P_SWEEP,
    WORKLOAD_CHOICES,
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
from experiments.simulation.latency_factory import (
    SyntheticLatencySpec,
    describe_synthetic_latency,
    synthetic_latency_metadata,
)
from experiments.simulation.latency_profiles import (
    DEFAULT_SUBSCRIPTION_PROFILE,
    load_empirical_distribution,
    load_pooled_distribution,
)
from experiments.simulation.offline_oracle import (
    OFFLINE_POLICY,
    run_offline_oracle_policy as run_offline_policy,
)
from experiments.subscriptions import SubscriptionPlan, load_subscription_plans
from rwsim.world.capacity import ProviderTier
from rwsim.world.providers import TieredProvider
from rwsim.world.scenarios import ScenarioConfig

SECTION_NAME = "cost-layer"
REAL_WORLD_SCENARIO = "cost_layer_real_world"
QUOTA_SCENARIO = "quota"
CONCURRENCY_SCENARIO = "concurrency"
JOINT_SCENARIO = "joint"
DEFAULT_CONCURRENCY_PLAN = "featherless_premium"
DEFAULT_CONCURRENCY_MODEL = "sharegpt"
SUBSCRIPTION_LATENCY_PROFILE = DEFAULT_SUBSCRIPTION_PROFILE

_SYNTHETIC_FAMILIES = ("uniform", "normal", "heavy_tail")
_QUOTA_LATENCY_FAMILIES = (*_SYNTHETIC_FAMILIES, "real_world")
_QUOTA_PROFILE_PROVIDER_KEYS = {
    "chutes": "chutes",
    "minimax_subscription_starter": "minimax",
    "minimax_subscription_plus": "minimax",
    "minimax_subscription_max": "minimax",
}
_CONCURRENCY_PROFILE_PROVIDER_KEYS = {
    "featherless_premium": "featherless",
}
_DEFAULT_SCENARIO_NAMES = (
    "cost_layer_uniform",
    "cost_layer_normal",
    "cost_layer_heavy_tail",
    REAL_WORLD_SCENARIO,
    QUOTA_SCENARIO,
    CONCURRENCY_SCENARIO,
)


def _synthetic_latency_spec(family: str = "heavy_tail") -> SyntheticLatencySpec:
    return SyntheticLatencySpec(
        family=family,
        anchor_ms=COST_LAYER_LATENCY_ANCHOR_MS,
    )


def _api_latency_metadata(family: str = "heavy_tail") -> dict[str, object]:
    metadata = synthetic_latency_metadata(_synthetic_latency_spec(family))
    return {
        "api_latency_generation_version": metadata["latency_generation_version"],
        "api_latency_family": metadata["latency_family"],
        "api_latency_anchor_kind": metadata["latency_anchor_kind"],
        "api_latency_anchor_ms": metadata["latency_anchor_ms"],
        "api_latency_distribution_mean_ms": metadata["latency_distribution_mean_ms"],
        "api_latency_distribution_p50_ms": metadata["latency_distribution_p50_ms"],
        "api_latency_shape": metadata["latency_shape"],
    }


def _synthetic_latency_text(family: str = "heavy_tail") -> str:
    return describe_synthetic_latency(_synthetic_latency_spec(family))


_SCENARIO_NAMES = (*_DEFAULT_SCENARIO_NAMES, JOINT_SCENARIO)


def list_scenarios() -> tuple[str, ...]:
    """Return all cost-layer scenario names."""
    return _SCENARIO_NAMES


def make_scenarios(
    *,
    subscription_plans: tuple[str, ...] = ("chutes",),
    subscription_counts: tuple[int, ...] | None = None,
    concurrency_plans: tuple[str, ...] = (DEFAULT_CONCURRENCY_PLAN,),
    concurrency_counts: tuple[int, ...] | None = None,
    concurrency_model: str = DEFAULT_CONCURRENCY_MODEL,
    quota_latency_family: str = "heavy_tail",
    include_joint: bool = False,
) -> dict[str, ScenarioConfig]:
    """Build runnable cost-layer scenarios keyed by artifact label."""
    scenarios: dict[str, ScenarioConfig] = {}
    scenario_names = (
        (*_DEFAULT_SCENARIO_NAMES, JOINT_SCENARIO)
        if include_joint
        else _DEFAULT_SCENARIO_NAMES
    )
    for name in scenario_names:
        if name == QUOTA_SCENARIO:
            scenarios.update(
                _make_quota_plan_scenarios(
                    subscription_plans=subscription_plans,
                    subscription_counts=subscription_counts,
                    quota_latency_family=quota_latency_family,
                )
            )
        elif name == CONCURRENCY_SCENARIO:
            scenarios.update(
                _make_concurrency_plan_scenarios(
                    concurrency_plans=concurrency_plans,
                    concurrency_counts=concurrency_counts,
                    model=concurrency_model,
                )
            )
        elif name == JOINT_SCENARIO:
            scenarios.update(
                _make_joint_plan_scenarios(
                    subscription_plans=subscription_plans,
                    subscription_counts=subscription_counts,
                    concurrency_plans=concurrency_plans,
                    concurrency_counts=concurrency_counts,
                    model=concurrency_model,
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
    concurrency_plan: str = DEFAULT_CONCURRENCY_PLAN,
    concurrency_count: int = 1,
    concurrency_model: str = DEFAULT_CONCURRENCY_MODEL,
    quota_latency_family: str = "heavy_tail",
) -> ScenarioConfig:
    """Build one cost-layer scenario by name."""
    if name.startswith("cost_layer_") and name.removeprefix("cost_layer_") in _SYNTHETIC_FAMILIES:
        return _make_api_cost_scenario(name.removeprefix("cost_layer_"))
    if name == REAL_WORLD_SCENARIO:
        return _make_real_world_api_cost_scenario()
    if name == QUOTA_SCENARIO:
        return _make_quota_scenario_for_plan(
            subscription_plan,
            subscription_count,
            latency_family=quota_latency_family,
        )
    if name == CONCURRENCY_SCENARIO:
        return _make_concurrency_scenario_for_plan(
            concurrency_plan,
            concurrency_count,
            model=concurrency_model,
        )
    if name == JOINT_SCENARIO:
        return _make_joint_scenario_for_plans(
            subscription_plan,
            subscription_count,
            concurrency_plan,
            concurrency_count,
            model=concurrency_model,
        )
    parsed_quota = _parse_quota_artifact_label(name)
    if parsed_quota is not None:
        plan_id, count, latency_family = parsed_quota
        return _make_quota_scenario_for_plan(
            plan_id,
            count,
            latency_family=latency_family,
        )
    parsed_concurrency = _parse_concurrency_artifact_label(name)
    if parsed_concurrency is not None:
        plan_id, count, model = parsed_concurrency
        return _make_concurrency_scenario_for_plan(
            plan_id,
            count,
            model=model,
        )
    parsed_joint = _parse_joint_artifact_label(name)
    if parsed_joint is not None:
        quota_plan_id, quota_count, concurrency_plan_id, concurrency_count, model = parsed_joint
        return _make_joint_scenario_for_plans(
            quota_plan_id,
            quota_count,
            concurrency_plan_id,
            concurrency_count,
            model=model,
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


def quota_artifact_label(
    plan_id: str,
    subscription_count: int,
    *,
    latency_family: str = "heavy_tail",
) -> str:
    """Return the stable artifact label for one quota plan/count."""
    label = f"quota__plan={plan_id}__n={subscription_count}"
    if latency_family != "heavy_tail":
        label += f"__latency={latency_family}"
    return label


def _parse_quota_artifact_label(name: str) -> tuple[str, int, str] | None:
    prefix = "quota__plan="
    marker = "__n="
    if not name.startswith(prefix) or marker not in name:
        return None
    plan_part, rest = name.removeprefix(prefix).rsplit(marker, 1)
    count_part, latency_part = _split_quota_count_and_latency(rest)
    try:
        count = int(count_part)
    except ValueError as exc:
        raise ValueError(f"invalid quota artifact label {name!r}") from exc
    return plan_part, count, latency_part


def _split_quota_count_and_latency(value: str) -> tuple[str, str]:
    marker = "__latency="
    if marker not in value:
        return value, "heavy_tail"
    count_part, latency_family = value.split(marker, 1)
    _validate_quota_latency_family(latency_family)
    return count_part, latency_family


def concurrency_artifact_label(
    plan_id: str,
    concurrency_count: int,
    *,
    model: str,
) -> str:
    """Return the stable artifact label for one concurrency plan/count/model."""
    return (
        f"concurrency__plan={plan_id}__n={concurrency_count}"
        f"__model={_encode_label_value(model)}"
    )


def _parse_concurrency_artifact_label(name: str) -> tuple[str, int, str] | None:
    prefix = "concurrency__plan="
    marker = "__n="
    model_marker = "__model="
    if not name.startswith(prefix) or marker not in name or model_marker not in name:
        return None
    plan_part, rest = name.removeprefix(prefix).rsplit(marker, 1)
    count_part, model_part = rest.split(model_marker, 1)
    try:
        count = int(count_part)
    except ValueError as exc:
        raise ValueError(f"invalid concurrency artifact label {name!r}") from exc
    if count <= 0:
        raise ValueError(f"invalid concurrency artifact label {name!r}: n must be > 0")
    model = unquote(model_part)
    if not model:
        raise ValueError(f"invalid concurrency artifact label {name!r}: missing model")
    return plan_part, count, model


def joint_artifact_label(
    quota_plan_id: str,
    quota_count: int,
    concurrency_plan_id: str,
    concurrency_count: int,
    *,
    model: str,
) -> str:
    """Return the stable artifact label for one joint quota/concurrency cell."""
    return (
        f"joint__quota_plan={quota_plan_id}__q={quota_count}"
        f"__concurrency_plan={concurrency_plan_id}__c={concurrency_count}"
        f"__model={_encode_label_value(model)}"
    )


def _parse_joint_artifact_label(name: str) -> tuple[str, int, str, int, str] | None:
    prefix = "joint__quota_plan="
    q_marker = "__q="
    c_plan_marker = "__concurrency_plan="
    c_marker = "__c="
    model_marker = "__model="
    if (
        not name.startswith(prefix)
        or q_marker not in name
        or c_plan_marker not in name
        or c_marker not in name
        or model_marker not in name
    ):
        return None
    quota_plan_part, rest = name.removeprefix(prefix).split(q_marker, 1)
    quota_count_part, rest = rest.split(c_plan_marker, 1)
    concurrency_plan_part, rest = rest.split(c_marker, 1)
    concurrency_count_part, model_part = rest.split(model_marker, 1)
    try:
        quota_count = int(quota_count_part)
        concurrency_count = int(concurrency_count_part)
    except ValueError as exc:
        raise ValueError(f"invalid joint artifact label {name!r}") from exc
    if quota_count <= 0 or concurrency_count <= 0:
        raise ValueError(f"invalid joint artifact label {name!r}: q and c must be > 0")
    model = unquote(model_part)
    if not model:
        raise ValueError(f"invalid joint artifact label {name!r}: missing model")
    return (
        quota_plan_part,
        quota_count,
        concurrency_plan_part,
        concurrency_count,
        model,
    )


def _encode_label_value(value: str) -> str:
    return quote(str(value), safe="-._~")


def _make_quota_plan_scenarios(
    *,
    subscription_plans: tuple[str, ...],
    subscription_counts: tuple[int, ...] | None,
    quota_latency_family: str,
) -> dict[str, ScenarioConfig]:
    _validate_quota_latency_family(quota_latency_family)
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
            scenario = _make_quota_scenario_for_plan(
                plan.plan_id,
                count,
                latency_family=quota_latency_family,
            )
            scenarios[scenario.name] = scenario
    return scenarios


def _make_concurrency_plan_scenarios(
    *,
    concurrency_plans: tuple[str, ...],
    concurrency_counts: tuple[int, ...] | None,
    model: str,
) -> dict[str, ScenarioConfig]:
    plans = load_subscription_plans()
    scenarios: dict[str, ScenarioConfig] = {}
    for plan_id in concurrency_plans:
        try:
            plan = plans[plan_id]
        except KeyError as exc:
            known = ", ".join(sorted(plans))
            raise ValueError(
                f"unknown concurrency plan {plan_id!r}; known plans: {known}"
            ) from exc
        counts = concurrency_counts or plan.subscription_counts
        _validate_subscription_counts(plan, counts)
        for count in counts:
            scenario = _make_concurrency_scenario_for_plan(
                plan.plan_id,
                count,
                model=model,
            )
            scenarios[scenario.name] = scenario
    return scenarios


def _make_joint_plan_scenarios(
    *,
    subscription_plans: tuple[str, ...],
    subscription_counts: tuple[int, ...] | None,
    concurrency_plans: tuple[str, ...],
    concurrency_counts: tuple[int, ...] | None,
    model: str,
) -> dict[str, ScenarioConfig]:
    plans = load_subscription_plans()
    scenarios: dict[str, ScenarioConfig] = {}
    for quota_plan_id in subscription_plans:
        try:
            quota_plan = plans[quota_plan_id]
        except KeyError as exc:
            known = ", ".join(sorted(plans))
            raise ValueError(
                f"unknown subscription plan {quota_plan_id!r}; known plans: {known}"
            ) from exc
        quota_counts = subscription_counts or quota_plan.subscription_counts
        _validate_subscription_counts(quota_plan, quota_counts)
        for concurrency_plan_id in concurrency_plans:
            try:
                concurrency_plan = plans[concurrency_plan_id]
            except KeyError as exc:
                known = ", ".join(sorted(plans))
                raise ValueError(
                    f"unknown concurrency plan {concurrency_plan_id!r}; known plans: {known}"
                ) from exc
            concurrency_counts_resolved = (
                concurrency_counts or concurrency_plan.subscription_counts
            )
            _validate_subscription_counts(
                concurrency_plan,
                concurrency_counts_resolved,
            )
            for quota_count in quota_counts:
                for concurrency_count in concurrency_counts_resolved:
                    scenario = _make_joint_scenario_for_plans(
                        quota_plan.plan_id,
                        quota_count,
                        concurrency_plan.plan_id,
                        concurrency_count,
                        model=model,
                    )
                    scenarios[scenario.name] = scenario
    return scenarios


def _validate_subscription_counts(
    plan: SubscriptionPlan,
    counts: tuple[int, ...],
) -> None:
    del plan
    invalid = [count for count in counts if count <= 0]
    if invalid:
        raise ValueError(f"subscription counts must be > 0, got {invalid[0]}")


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
        "--concurrency-plan",
        help="Concurrency scenario subscription plan id, e.g. featherless_premium.",
    )
    parser.add_argument(
        "--concurrency-plans",
        help="Comma-separated concurrency scenario plan ids, e.g. featherless_premium.",
    )
    parser.add_argument(
        "--concurrency-count",
        type=int,
        help="Concurrency scenario subscription/account count.",
    )
    parser.add_argument(
        "--concurrency-counts",
        help="Comma-separated concurrency scenario counts, e.g. 1,2,3,4.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_CONCURRENCY_MODEL,
        help=(
            "Concurrency scenario model id or trace alias. Defaults to "
            f"{DEFAULT_CONCURRENCY_MODEL}."
        ),
    )
    parser.add_argument(
        "--latency-family",
        default="heavy_tail",
        choices=_QUOTA_LATENCY_FAMILIES,
        help=(
            "Quota scenario TTFT family. Defaults to heavy_tail (LogNormal). "
            "Use real_world for the pooled rw8_pooled latency distribution."
        ),
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
    selected_public_scenarios = (
        tuple(args.scenario) if args.scenario else _DEFAULT_SCENARIO_NAMES
    )
    has_joint = JOINT_SCENARIO in selected_public_scenarios
    has_quota = QUOTA_SCENARIO in selected_public_scenarios or has_joint
    has_concurrency = CONCURRENCY_SCENARIO in selected_public_scenarios or has_joint
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
    concurrency_plan_ids = _parse_concurrency_plan_args(
        args.concurrency_plan,
        args.concurrency_plans,
        has_concurrency=has_concurrency,
    )
    concurrency_counts = _parse_concurrency_count_args(
        args.concurrency_count,
        args.concurrency_counts,
        has_concurrency=has_concurrency,
    )
    scenarios = make_scenarios(
        subscription_plans=subscription_plan_ids,
        subscription_counts=subscription_counts,
        concurrency_plans=concurrency_plan_ids,
        concurrency_counts=concurrency_counts,
        concurrency_model=args.model,
        quota_latency_family=args.latency_family,
        include_joint=has_joint,
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
            elif name == CONCURRENCY_SCENARIO:
                selected.update(
                    {
                        scenario_name: scenario
                        for scenario_name, scenario in scenarios.items()
                        if scenario.metadata.get("public_scenario") == CONCURRENCY_SCENARIO
                    }
                )
            elif name == JOINT_SCENARIO:
                selected.update(
                    {
                        scenario_name: scenario
                        for scenario_name, scenario in scenarios.items()
                        if scenario.metadata.get("public_scenario") == JOINT_SCENARIO
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


def _parse_concurrency_plan_args(
    single: str | None,
    multiple: str | None,
    *,
    has_concurrency: bool,
) -> tuple[str, ...]:
    if single and multiple:
        raise SystemExit("use either --concurrency-plan or --concurrency-plans, not both")
    if not has_concurrency:
        if single or multiple:
            raise SystemExit("concurrency plan flags require --scenario concurrency")
        return (DEFAULT_CONCURRENCY_PLAN,)
    values = (
        _split_csv(multiple)
        if multiple
        else ((single,) if single else (DEFAULT_CONCURRENCY_PLAN,))
    )
    plans = load_subscription_plans()
    unknown = [value for value in values if value not in plans]
    if unknown:
        known = ", ".join(sorted(plans))
        raise SystemExit(f"unknown concurrency plan {unknown[0]!r}; known plans: {known}")
    return values


def _parse_concurrency_count_args(
    single: int | None,
    multiple: str | None,
    *,
    has_concurrency: bool,
) -> tuple[int, ...] | None:
    if single is not None and multiple:
        raise SystemExit("use either --concurrency-count or --concurrency-counts, not both")
    if not has_concurrency:
        if single is not None or multiple:
            raise SystemExit("concurrency count flags require --scenario concurrency")
        return None
    if multiple:
        return tuple(int(value) for value in _split_csv(multiple))
    if single is not None:
        return (int(single),)
    return None


def _validate_quota_latency_family(latency_family: str) -> None:
    if latency_family not in _QUOTA_LATENCY_FAMILIES:
        known = ", ".join(_QUOTA_LATENCY_FAMILIES)
        raise ValueError(f"unknown quota latency family {latency_family!r}; known: {known}")


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


def _make_api_cost_scenario(family: str) -> ScenarioConfig:
    latency_spec = _synthetic_latency_spec(family)
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
            f"({_synthetic_latency_text(family)}), input cost ratio $1/$2/$4 "
            "and output cost ratio $5/$10/$20 per million tokens."
        ),
        providers=providers,
        arrival_process="trace",
        primary_slo_ms=2000.0,
        metadata={
            "public_scenario": f"cost_layer_{family}",
            "artifact_label": f"cost_layer_{family}",
            **synthetic_latency_metadata(latency_spec),
        },
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
    *,
    latency_family: str = "heavy_tail",
) -> ScenarioConfig:
    _validate_quota_latency_family(latency_family)
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

    label = quota_artifact_label(
        plan.plan_id,
        subscription_count,
        latency_family=latency_family,
    )
    ttft_dist = load_pooled_distribution("rw8_pooled") if latency_family == "real_world" else None
    providers = [
        make_quota_provider(
            f"{plan.plan_id}_quota",
            plan=plan,
            subscription_count=subscription_count,
            latency_family="heavy_tail" if ttft_dist is not None else latency_family,
        ),
        make_api_provider(
            "api_cheap",
            cost_per_million_tokens=COST_RATIO_PER_MILLION[0],
            latency_family="heavy_tail" if ttft_dist is not None else latency_family,
        ),
        make_api_provider(
            "api_mid",
            cost_per_million_tokens=COST_RATIO_PER_MILLION[1],
            latency_family="heavy_tail" if ttft_dist is not None else latency_family,
        ),
        make_api_provider(
            "api_expensive",
            cost_per_million_tokens=COST_RATIO_PER_MILLION[2],
            latency_family="heavy_tail" if ttft_dist is not None else latency_family,
        ),
    ]
    if ttft_dist is not None:
        for provider in providers:
            provider.ttft_dist = ttft_dist
    quota_window_text = ", ".join(
        f"{window.quota_requests * subscription_count:g}/{window.quota_window_sec:g}s"
        for window in plan.quota_windows
    )
    latency_text = (
        "real-world pooled rw8_pooled"
        if latency_family == "real_world"
        else _synthetic_latency_text(latency_family)
    )
    latency_metadata = (
        {}
        if latency_family == "real_world"
        else synthetic_latency_metadata(_synthetic_latency_spec(latency_family))
    )
    return ScenarioConfig(
        name=label,
        description=(
            f"Cost-layer quota scenario: {plan.display_name} x{subscription_count} "
            f"({quota_window_text}) as one aggregate quota provider, fixed "
            "cheap/mid/expensive on-demand fallback providers, "
            f"TTFT={latency_text}."
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
            "latency_family": latency_family,
            **latency_metadata,
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


def _make_concurrency_scenario_for_plan(
    plan_id: str,
    concurrency_count: int,
    *,
    model: str,
) -> ScenarioConfig:
    plans = load_subscription_plans()
    try:
        plan = plans[plan_id]
    except KeyError as exc:
        known = ", ".join(sorted(plans))
        raise ValueError(f"unknown concurrency plan {plan_id!r}; known plans: {known}") from exc
    _validate_subscription_counts(plan, (concurrency_count,))
    if "cost_layer_concurrency" not in plan.eligible_sections:
        raise ValueError(
            f"subscription plan {plan.plan_id!r} is not eligible for cost-layer concurrency runs"
        )
    if plan.concurrency_allotment is None:
        raise ValueError(f"concurrency plan {plan.plan_id!r} lacks concurrency_allotment")
    resolution = plan.resolve_model_class_with_cost(model)
    if resolution is None:
        raise ValueError(
            f"concurrency plan {plan.plan_id!r} does not support model {model!r}"
        )
    label = concurrency_artifact_label(
        plan.plan_id,
        concurrency_count,
        model=model,
    )
    latency_metadata = synthetic_latency_metadata(_synthetic_latency_spec("heavy_tail"))
    concurrency_provider = make_concurrency_provider(
        f"{plan.plan_id}_concurrency",
        plan=plan,
        concurrency_count=concurrency_count,
        model=model,
    )
    providers = [
        concurrency_provider,
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
    capacity_units = int(plan.concurrency_allotment) * int(concurrency_count)
    return ScenarioConfig(
        name=label,
        description=(
            f"Cost-layer concurrency scenario: {plan.display_name} x{concurrency_count} "
            f"for model {model!r} ({resolution.model_class}, cost={resolution.cost}) "
            f"with weighted capacity {capacity_units}, fixed cheap/mid/expensive "
            f"on-demand fallback providers, TTFT={_synthetic_latency_text()}."
        ),
        providers=providers,
        arrival_process="trace",
        primary_slo_ms=2000.0,
        metadata={
            "public_scenario": CONCURRENCY_SCENARIO,
            "artifact_label": label,
            "concurrency_plan": plan.plan_id,
            "concurrency_plan_display_name": plan.display_name,
            "concurrency_count": concurrency_count,
            "model": model,
            "model_class": resolution.model_class,
            "model_concurrency_cost": resolution.cost,
            "concurrency_capacity_units": capacity_units,
            "effective_concurrency_limit": concurrency_provider.concurrency.limit
            if concurrency_provider.concurrency is not None
            else 0,
            "model_concurrency_costs_by_class": dict(
                plan.model_concurrency_costs_by_class
            ),
            **latency_metadata,
        },
    )


def _make_joint_scenario_for_plans(
    quota_plan_id: str,
    subscription_count: int,
    concurrency_plan_id: str,
    concurrency_count: int,
    *,
    model: str,
) -> ScenarioConfig:
    plans = load_subscription_plans()
    try:
        quota_plan = plans[quota_plan_id]
    except KeyError as exc:
        known = ", ".join(sorted(plans))
        raise ValueError(f"unknown subscription plan {quota_plan_id!r}; known plans: {known}") from exc
    try:
        concurrency_plan = plans[concurrency_plan_id]
    except KeyError as exc:
        known = ", ".join(sorted(plans))
        raise ValueError(
            f"unknown concurrency plan {concurrency_plan_id!r}; known plans: {known}"
        ) from exc

    _validate_subscription_counts(quota_plan, (subscription_count,))
    _validate_subscription_counts(concurrency_plan, (concurrency_count,))
    if "cost_layer_joint" not in quota_plan.eligible_sections:
        raise ValueError(
            f"subscription plan {quota_plan.plan_id!r} is not eligible for cost-layer joint runs"
        )
    if "cost_layer_joint" not in concurrency_plan.eligible_sections:
        raise ValueError(
            f"concurrency plan {concurrency_plan.plan_id!r} is not eligible for cost-layer joint runs"
        )
    if concurrency_plan.concurrency_allotment is None:
        raise ValueError(
            f"concurrency plan {concurrency_plan.plan_id!r} lacks concurrency_allotment"
        )
    resolution = concurrency_plan.resolve_model_class_with_cost(model)
    if resolution is None:
        raise ValueError(
            f"concurrency plan {concurrency_plan.plan_id!r} does not support model {model!r}"
        )

    label = joint_artifact_label(
        quota_plan.plan_id,
        subscription_count,
        concurrency_plan.plan_id,
        concurrency_count,
        model=model,
    )
    quota_provider = make_quota_provider(
        f"{quota_plan.plan_id}_quota",
        plan=quota_plan,
        subscription_count=subscription_count,
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
    providers = [
        quota_provider,
        concurrency_provider,
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
        for window in quota_plan.quota_windows
    )
    capacity_units = int(concurrency_plan.concurrency_allotment) * int(concurrency_count)
    api_latency_metadata = _api_latency_metadata("heavy_tail")
    return ScenarioConfig(
        name=label,
        description=(
            f"Cost-layer joint scenario: {quota_plan.display_name} x{subscription_count} "
            f"({quota_window_text}) plus {concurrency_plan.display_name} "
            f"x{concurrency_count} for model {model!r} "
            f"({resolution.model_class}, cost={resolution.cost}), fixed "
            "cheap/mid/expensive on-demand fallback providers, "
            f"subscription TTFT={SUBSCRIPTION_LATENCY_PROFILE} and "
            f"fallback API TTFT={_synthetic_latency_text()}."
        ),
        providers=providers,
        arrival_process="trace",
        primary_slo_ms=2000.0,
        metadata={
            "public_scenario": JOINT_SCENARIO,
            "artifact_label": label,
            "subscription_plan": quota_plan.plan_id,
            "subscription_plan_display_name": quota_plan.display_name,
            "subscription_count": subscription_count,
            "concurrency_plan": concurrency_plan.plan_id,
            "concurrency_plan_display_name": concurrency_plan.display_name,
            "concurrency_count": concurrency_count,
            "model": model,
            "model_class": resolution.model_class,
            "model_concurrency_cost": resolution.cost,
            "concurrency_capacity_units": capacity_units,
            "effective_concurrency_limit": concurrency_provider.concurrency.limit
            if concurrency_provider.concurrency is not None
            else 0,
            "model_concurrency_costs_by_class": dict(
                concurrency_plan.model_concurrency_costs_by_class
            ),
            "latency_profile": SUBSCRIPTION_LATENCY_PROFILE,
            "quota_latency_profile_provider": quota_profile_key,
            "concurrency_latency_profile_provider": concurrency_profile_key,
            **api_latency_metadata,
            "quota_windows": [
                {
                    "name": window.name,
                    "quota_requests": window.quota_requests,
                    "quota_window_sec": window.quota_window_sec,
                    "aggregate_quota_requests": window.quota_requests * subscription_count,
                }
                for window in quota_plan.quota_windows
            ],
        },
    )


def _subscription_latency_profile_key(plan_id: str) -> str | None:
    return _QUOTA_PROFILE_PROVIDER_KEYS.get(
        plan_id,
        _CONCURRENCY_PROFILE_PROVIDER_KEYS.get(plan_id),
    )


__all__ = [
    "CONCURRENCY_SCENARIO",
    "JOINT_SCENARIO",
    "OFFLINE_POLICY",
    "QUOTA_SCENARIO",
    "SECTION_NAME",
    "concurrency_artifact_label",
    "joint_artifact_label",
    "list_scenarios",
    "make_scenario",
    "make_scenarios",
    "policies_for_section",
    "quota_artifact_label",
    "run_offline_policy",
]


if __name__ == "__main__":
    raise SystemExit(main())
