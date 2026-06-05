"""Method A harness for effective-cost formula ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from experiments.ablations.effective_cost.policy import LPOnlyAblationPolicy
from experiments.ablations.effective_cost.presets import (
    DEFAULT_CONCURRENCY_CURVES,
    DEFAULT_P_VALUES,
    DEFAULT_QUOTA_CURVES,
    make_ablation_presets,
    make_concurrency_ablation_presets,
)
from experiments.simulation import common, cost_layer
from experiments.simulation.latency_factory import (
    SyntheticLatencySpec,
    describe_synthetic_latency,
    synthetic_latency_metadata,
)
from experiments.simulation.latency_profiles import load_pooled_distribution
from experiments.subscriptions import SubscriptionPlan, load_subscription_plans
from rwsim.engine.simulator import Simulator
from rwsim.world.capacity import MultiWindowQuotaState, QuotaState
from rwsim.world.scenarios import ScenarioConfig

if TYPE_CHECKING:
    from rwsim.core.cost import ScarcityCurve
    from rwsim.metrics import Run
    from rwsim.schemas import Request

SECTION_NAME = "effective-cost-ablation"
PHASE_QUOTA = "quota"
PHASE_CONCURRENCY = "concurrency"
PHASE_JOINT = "joint"
PHASES = (PHASE_QUOTA, PHASE_CONCURRENCY, PHASE_JOINT)
DEFAULT_SUBSCRIPTION_PLAN = "chutes"
DEFAULT_QSTAR = 16
DEFAULT_CONCURRENCY_PLAN = cost_layer.DEFAULT_CONCURRENCY_PLAN
DEFAULT_CONCURRENCY_MODEL = cost_layer.DEFAULT_CONCURRENCY_MODEL
DEFAULT_CONCURRENCY_COUNTS = (6, 8, 10, 11, 12, 13, 14, 16)
DEFAULT_LATENCY_FAMILY = "heavy_tail"
DEFAULT_WORKLOAD = "burstgpt"
DEFAULT_OUTPUT_DIR = common.OUTPUT_DIR / "ablations" / "effective_cost"


def list_scenarios() -> tuple[str, ...]:
    """Return default Phase A scenario labels."""
    return tuple(make_scenarios().keys())


def make_scenarios(
    *,
    phase: str = PHASE_QUOTA,
    subscription_plan: str = DEFAULT_SUBSCRIPTION_PLAN,
    quota_limit: int | tuple[int, ...] | None = None,
    qstar: int | tuple[int, ...] | None = None,
    latency_family: str = DEFAULT_LATENCY_FAMILY,
    concurrency_plan: str = DEFAULT_CONCURRENCY_PLAN,
    concurrency_count: int | tuple[int, ...] = DEFAULT_CONCURRENCY_COUNTS,
    concurrency_model: str = DEFAULT_CONCURRENCY_MODEL,
) -> dict[str, ScenarioConfig]:
    """Build scenarios for the current Method A/B implementation."""
    _require_supported_phase(phase)
    scenarios: dict[str, ScenarioConfig] = {}
    if phase == PHASE_QUOTA:
        for value in _quota_limit_values(
            subscription_plan=subscription_plan,
            quota_limit=quota_limit,
            qstar=qstar,
        ):
            scenario = _make_quota_limit_scenario_for_plan(
                subscription_plan,
                value,
                latency_family=latency_family,
            )
            scenarios[scenario.name] = scenario
    elif phase == PHASE_CONCURRENCY:
        for value in _positive_unique_values("concurrency_count", concurrency_count):
            label = cost_layer.concurrency_artifact_label(
                concurrency_plan,
                value,
                model=concurrency_model,
            )
            scenario = cost_layer.make_scenario(label)
            scenarios[scenario.name] = scenario
    return scenarios


def make_scenario(name: str) -> ScenarioConfig:
    """Rebuild an ablation scenario from its stable artifact label."""
    parsed_quota_limit = _parse_quota_limit_artifact_label(name)
    if parsed_quota_limit is not None:
        plan_id, quota_limit, latency_family = parsed_quota_limit
        return _make_quota_limit_scenario_for_plan(
            plan_id,
            quota_limit,
            latency_family=latency_family,
        )
    return cost_layer.make_scenario(name)


def policies_for_phase(
    *,
    phase: str = PHASE_QUOTA,
    curves: tuple[ScarcityCurve, ...] = DEFAULT_QUOTA_CURVES,
    concurrency_curves: tuple[ScarcityCurve, ...] = DEFAULT_CONCURRENCY_CURVES,
    alpha_values: tuple[float, ...] = DEFAULT_P_VALUES,
) -> tuple[str, ...]:
    """Return default policies for one phase."""
    _require_supported_phase(phase)
    if phase == PHASE_QUOTA:
        return tuple(make_ablation_presets(curves=curves, alpha_values=alpha_values).keys())
    return tuple(
        make_concurrency_ablation_presets(
            concurrency_curves=concurrency_curves,
            alpha_values=alpha_values,
        ).keys()
    )


def run_ablation_policy(
    scenario: ScenarioConfig,
    requests: list[Request],
    policy_name: str,
    *,
    presets: dict[str, dict[str, Any]],
    seed: int,
    retain_records: bool = True,
) -> Run:
    """Run one ablation-local LP-only policy."""
    policy = build_ablation_policy(
        policy_name,
        presets=presets,
        scenario=scenario,
        requests=requests,
        seed=seed,
    )
    simulator = Simulator(scenario=scenario, seed=seed, retain_records=retain_records)
    return simulator.run(requests, policy, policy_name=policy_name)


def build_ablation_policy(
    policy_name: str,
    *,
    presets: dict[str, dict[str, Any]],
    scenario: ScenarioConfig,
    requests: list[Request],
    seed: int,
) -> LPOnlyAblationPolicy:
    """Instantiate LPOnlyAblationPolicy through the ablation-local builder."""
    try:
        preset = presets[policy_name]
    except KeyError as exc:
        known = ", ".join(sorted(presets))
        raise ValueError(f"unknown effective-cost policy {policy_name!r}; known: {known}") from exc
    if preset.get("policy") != "LPOnlyAblationPolicy":
        raise ValueError(f"unsupported effective-cost preset {policy_name!r}: {preset!r}")

    params = dict(preset.get("params", {}))
    if params.get("cost_envelope") == common.WORKLOAD_COST_ENVELOPE:
        params["cost_envelope"] = common.workload_cost_envelope(
            scenario.providers,
            requests,
        )
    return LPOnlyAblationPolicy(seed=seed, **params)


def run_effective_cost_cell(
    cell: common.SectionCell,
    presets: dict[str, dict[str, Any]],
    workload_dataset: str,
    duration_sec: float | None,
    max_requests: int | None,
    retain_records: bool,
) -> common.SectionCellResult:
    """Run one effective-cost ablation cell in a worker process."""
    scenario = make_scenario(cell.scenario_name)
    requests = common.load_workload(
        dataset=workload_dataset,
        duration_sec=duration_sec,
        max_requests=max_requests,
    )
    run = run_ablation_policy(
        scenario,
        requests,
        cell.policy,
        presets=presets,
        seed=cell.seed,
        retain_records=retain_records,
    )
    return common.SectionCellResult(
        scenario_name=cell.scenario_name,
        policy=cell.policy,
        seed=cell.seed,
        run=run,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the effective-cost ablation harness."""
    parser = argparse.ArgumentParser(
        prog="routewise ablation effective-cost",
        description=__doc__,
    )
    parser.add_argument(
        "--phase",
        default=PHASE_QUOTA,
        choices=PHASES,
        help="Ablation phase. Quota and concurrency are implemented; joint is deferred.",
    )
    parser.add_argument(
        "--curve",
        action="append",
        choices=DEFAULT_QUOTA_CURVES,
        help="Quota curve to run. Repeat to compare curves. Defaults to all Phase A curves.",
    )
    parser.add_argument(
        "--alpha",
        "--p",
        type=float,
        action="append",
        dest="alpha_values",
        help=f"LP budget alpha value. Repeat to sweep. Defaults to {DEFAULT_P_VALUES}.",
    )
    parser.add_argument(
        "--concurrency-curve",
        action="append",
        choices=DEFAULT_CONCURRENCY_CURVES,
        dest="concurrency_curves",
        help=(
            "Concurrency curve to run for Phase B. Repeat to compare curves. "
            f"Defaults to {DEFAULT_CONCURRENCY_CURVES}."
        ),
    )
    parser.add_argument(
        "--subscription-plan",
        default=DEFAULT_SUBSCRIPTION_PLAN,
        help=f"Quota subscription plan id. Defaults to {DEFAULT_SUBSCRIPTION_PLAN}.",
    )
    parser.add_argument(
        "--quota-limit",
        type=int,
        action="append",
        dest="quota_limits",
        help=(
            "Aggregate quota request limit to assign to the fixed one-plan quota "
            "provider. Repeat to sweep. Defaults to the old q*=16 equivalent."
        ),
    )
    parser.add_argument(
        "--qstar",
        type=int,
        action="append",
        dest="qstar_values",
        help=(
            "Deprecated: quota subscription multiplier used only to derive "
            f"quota-limit = qstar * plan quota. Defaults to {DEFAULT_QSTAR}."
        ),
    )
    parser.add_argument(
        "--concurrency-plan",
        default=DEFAULT_CONCURRENCY_PLAN,
        help=f"Concurrency subscription plan id. Defaults to {DEFAULT_CONCURRENCY_PLAN}.",
    )
    parser.add_argument(
        "--concurrency-count",
        type=int,
        action="append",
        dest="concurrency_counts",
        help=(
            "Concurrency subscription/account count. Repeat to sweep. "
            f"Defaults to {DEFAULT_CONCURRENCY_COUNTS}."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_CONCURRENCY_MODEL,
        help=f"Concurrency scenario model id or trace alias. Defaults to {DEFAULT_CONCURRENCY_MODEL}.",
    )
    parser.add_argument(
        "--latency-family",
        default=DEFAULT_LATENCY_FAMILY,
        choices=("uniform", "normal", "heavy_tail", "real_world"),
        help=f"Quota scenario TTFT family. Defaults to {DEFAULT_LATENCY_FAMILY}.",
    )
    parser.add_argument(
        "--workload",
        default=DEFAULT_WORKLOAD,
        choices=("burstgpt", "sharegpt_burstgpt"),
        help=f"Trace workload to replay. Defaults to {DEFAULT_WORKLOAD}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        help=f"Seed to run. Repeat to run multiple. Defaults to {common.DEFAULT_SEEDS}.",
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
        help="Directory for metadata.json, summary.json, and summary.csv.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel scenario-policy-seed cells to run. Defaults to 1.",
    )

    args = parser.parse_args(argv)
    _require_supported_phase(args.phase)
    alpha_values = tuple(args.alpha_values) if args.alpha_values else DEFAULT_P_VALUES
    quota_limits = tuple(args.quota_limits) if args.quota_limits else None
    qstar_values = tuple(args.qstar_values) if args.qstar_values else None
    concurrency_counts = (
        tuple(args.concurrency_counts) if args.concurrency_counts else DEFAULT_CONCURRENCY_COUNTS
    )
    scenarios = make_scenarios(
        phase=args.phase,
        subscription_plan=args.subscription_plan,
        quota_limit=quota_limits,
        qstar=qstar_values,
        latency_family=args.latency_family,
        concurrency_plan=args.concurrency_plan,
        concurrency_count=concurrency_counts,
        concurrency_model=args.model,
    )
    if args.phase == PHASE_QUOTA:
        curves = tuple(args.curve) if args.curve else DEFAULT_QUOTA_CURVES
        presets = make_ablation_presets(curves=curves, alpha_values=alpha_values)
    elif args.phase == PHASE_CONCURRENCY:
        concurrency_curves = (
            tuple(args.concurrency_curves)
            if args.concurrency_curves
            else DEFAULT_CONCURRENCY_CURVES
        )
        presets = make_concurrency_ablation_presets(
            concurrency_curves=concurrency_curves,
            alpha_values=alpha_values,
        )
    else:
        _require_supported_phase(args.phase)
    policies = tuple(presets)
    output_dir = args.output_dir or (DEFAULT_OUTPUT_DIR / args.phase)

    rows = common.run_section(
        section_name=SECTION_NAME,
        scenarios=scenarios,
        policies=policies,
        presets=presets,
        seeds=tuple(args.seed) if args.seed else common.DEFAULT_SEEDS,
        section_runners=_section_runners(policies, presets),
        parallel_cell_runner=run_effective_cost_cell,
        workload_dataset=args.workload,
        duration_sec=args.duration_sec,
        max_requests=args.max_requests,
        output_dir=output_dir,
        retain_records=False,
        jobs=args.jobs,
    )
    print(
        json.dumps(
            {
                "section": SECTION_NAME,
                "phase": args.phase,
                "rows": len(rows),
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


def _section_runners(
    policies: tuple[str, ...],
    presets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    runners = {}
    for policy_name in policies:
        runners[policy_name] = _runner_for_policy(policy_name, presets)
    return runners


def _runner_for_policy(policy_name: str, presets: dict[str, dict[str, Any]]):
    def run(
        scenario: ScenarioConfig,
        requests: list[Request],
        seed: int,
        retain_records: bool = True,
    ) -> Run:
        return run_ablation_policy(
            scenario,
            requests,
            policy_name,
            presets=presets,
            seed=seed,
            retain_records=retain_records,
        )

    return run


def _require_supported_phase(phase: str) -> None:
    if phase == PHASE_JOINT:
        raise ValueError(
            f"effective-cost ablation phase {phase!r} is deferred until Phase A "
            "and Phase B have stable configurations"
        )
    if phase not in (PHASE_QUOTA, PHASE_CONCURRENCY):
        known = ", ".join(PHASES)
        raise ValueError(f"unknown effective-cost ablation phase {phase!r}; known: {known}")


def quota_limit_artifact_label(
    plan_id: str,
    quota_limit: int,
    *,
    latency_family: str = DEFAULT_LATENCY_FAMILY,
) -> str:
    """Return the stable artifact label for one fixed-plan quota-limit cell."""
    label = f"quota_limit__plan={plan_id}__quota={quota_limit}"
    if latency_family != DEFAULT_LATENCY_FAMILY:
        label += f"__latency={latency_family}"
    return label


def _parse_quota_limit_artifact_label(name: str) -> tuple[str, int, str] | None:
    prefix = "quota_limit__plan="
    marker = "__quota="
    if not name.startswith(prefix) or marker not in name:
        return None
    plan_part, rest = name.removeprefix(prefix).rsplit(marker, 1)
    quota_part, latency_family = _split_quota_limit_and_latency(rest)
    try:
        quota_limit = int(quota_part)
    except ValueError as exc:
        raise ValueError(f"invalid quota-limit artifact label {name!r}") from exc
    if quota_limit <= 0:
        raise ValueError(f"invalid quota-limit artifact label {name!r}: quota must be > 0")
    return plan_part, quota_limit, latency_family


def _split_quota_limit_and_latency(value: str) -> tuple[str, str]:
    marker = "__latency="
    if marker not in value:
        return value, DEFAULT_LATENCY_FAMILY
    quota_part, latency_family = value.split(marker, 1)
    if latency_family not in ("uniform", "normal", "heavy_tail", "real_world"):
        raise ValueError(f"unknown quota latency family {latency_family!r}")
    return quota_part, latency_family


def _quota_limit_values(
    *,
    subscription_plan: str,
    quota_limit: int | tuple[int, ...] | None,
    qstar: int | tuple[int, ...] | None,
) -> tuple[int, ...]:
    if quota_limit is not None and qstar is not None:
        raise ValueError("provide either quota_limit or qstar, not both")
    if quota_limit is not None:
        return _positive_unique_values("quota_limit", quota_limit)

    qstar_values = _positive_unique_values(
        "qstar",
        DEFAULT_QSTAR if qstar is None else qstar,
    )
    base_quota = _primary_quota_requests(_load_quota_plan(subscription_plan))
    return tuple(base_quota * value for value in qstar_values)


def _make_quota_limit_scenario_for_plan(
    plan_id: str,
    quota_limit: int,
    *,
    latency_family: str = DEFAULT_LATENCY_FAMILY,
) -> ScenarioConfig:
    if quota_limit <= 0:
        raise ValueError(f"quota_limit must be > 0, got {quota_limit}")
    if latency_family not in ("uniform", "normal", "heavy_tail", "real_world"):
        raise ValueError(f"unknown quota latency family {latency_family!r}")
    plan = _load_quota_plan(plan_id)
    if "cost_layer_quota" not in plan.eligible_sections:
        raise ValueError(
            f"subscription plan {plan.plan_id!r} is not eligible for effective-cost quota runs"
        )

    base_quota = _primary_quota_requests(plan)
    quota_multiplier = quota_limit / float(base_quota)
    ttft_dist = load_pooled_distribution("rw8_pooled") if latency_family == "real_world" else None
    synthetic_family = DEFAULT_LATENCY_FAMILY if ttft_dist is not None else latency_family
    providers = [
        common.make_quota_provider(
            f"{plan.plan_id}_quota",
            plan=plan,
            subscription_count=1,
            latency_family=synthetic_family,
        ),
        common.make_api_provider(
            "api_cheap",
            cost_per_million_tokens=common.COST_RATIO_PER_MILLION[0],
            latency_family=synthetic_family,
        ),
        common.make_api_provider(
            "api_mid",
            cost_per_million_tokens=common.COST_RATIO_PER_MILLION[1],
            latency_family=synthetic_family,
        ),
        common.make_api_provider(
            "api_expensive",
            cost_per_million_tokens=common.COST_RATIO_PER_MILLION[2],
            latency_family=synthetic_family,
        ),
    ]
    providers[0].quota = _quota_state_for_limit(
        plan,
        quota_limit=quota_limit,
        quota_multiplier=quota_multiplier,
    )
    if ttft_dist is not None:
        for provider in providers:
            provider.ttft_dist = ttft_dist

    label = quota_limit_artifact_label(
        plan.plan_id,
        quota_limit,
        latency_family=latency_family,
    )
    quota_windows = _quota_window_metadata(
        plan,
        quota_limit=quota_limit,
        quota_multiplier=quota_multiplier,
    )
    quota_window_text = ", ".join(
        f"{window['aggregate_quota_requests']:g}/{window['quota_window_sec']:g}s"
        for window in quota_windows
    )
    latency_text = (
        "real-world pooled rw8_pooled"
        if latency_family == "real_world"
        else describe_synthetic_latency(
            SyntheticLatencySpec(
                family=latency_family,
                anchor_ms=common.COST_LAYER_LATENCY_ANCHOR_MS,
            )
        )
    )
    latency_metadata = (
        {}
        if latency_family == "real_world"
        else synthetic_latency_metadata(
            SyntheticLatencySpec(
                family=latency_family,
                anchor_ms=common.COST_LAYER_LATENCY_ANCHOR_MS,
            )
        )
    )
    return ScenarioConfig(
        name=label,
        description=(
            f"Effective-cost quota-limit ablation: {plan.display_name} with one "
            f"subscription fixed cost and direct quota limit {quota_window_text}, "
            "fixed cheap/mid/expensive on-demand fallback providers, "
            f"TTFT={latency_text}."
        ),
        providers=providers,
        arrival_process="trace",
        primary_slo_ms=cost_layer.DEFAULT_PRIMARY_SLO_MS,
        metadata={
            "public_scenario": PHASE_QUOTA,
            "artifact_label": label,
            "subscription_plan": plan.plan_id,
            "subscription_plan_display_name": plan.display_name,
            "subscription_count": 1,
            "quota_limit": quota_limit,
            "quota_multiplier": quota_multiplier,
            "quota_base_requests_per_subscription": base_quota,
            "latency_family": latency_family,
            **latency_metadata,
            "quota_windows": quota_windows,
        },
    )


def _load_quota_plan(plan_id: str) -> SubscriptionPlan:
    plans = load_subscription_plans()
    try:
        plan = plans[plan_id]
    except KeyError as exc:
        known = ", ".join(sorted(plans))
        raise ValueError(f"unknown subscription plan {plan_id!r}; known plans: {known}") from exc
    if plan.tier != "quota":
        raise ValueError(f"subscription plan {plan.plan_id!r} is not a quota plan")
    return plan


def _primary_quota_requests(plan: SubscriptionPlan) -> int:
    if not plan.quota_windows:
        raise ValueError(f"subscription plan {plan.plan_id!r} has no quota windows")
    return int(plan.quota_windows[0].quota_requests)


def _quota_state_for_limit(
    plan: SubscriptionPlan,
    *,
    quota_limit: int,
    quota_multiplier: float,
) -> QuotaState | MultiWindowQuotaState:
    windows = tuple(
        QuotaState(
            size=(
                int(quota_limit)
                if index == 0
                else _scaled_quota_size(window.quota_requests, quota_multiplier)
            ),
            window_sec=window.quota_window_sec,
        )
        for index, window in enumerate(plan.quota_windows)
    )
    return windows[0] if len(windows) == 1 else MultiWindowQuotaState(windows)


def _quota_window_metadata(
    plan: SubscriptionPlan,
    *,
    quota_limit: int,
    quota_multiplier: float,
) -> list[dict[str, float | int | str]]:
    return [
        {
            "name": window.name,
            "quota_requests": window.quota_requests,
            "quota_window_sec": window.quota_window_sec,
            "aggregate_quota_requests": (
                int(quota_limit)
                if index == 0
                else _scaled_quota_size(
                    window.quota_requests,
                    quota_multiplier,
                )
            ),
        }
        for index, window in enumerate(plan.quota_windows)
    ]


def _scaled_quota_size(quota_requests: int, quota_multiplier: float) -> int:
    return max(1, round(int(quota_requests) * float(quota_multiplier)))


def _positive_unique_values(name: str, values: int | tuple[int, ...]) -> tuple[int, ...]:
    values = (values,) if isinstance(values, int) else values
    if not values:
        raise ValueError(f"{name} sweep must contain at least one value")
    for value in values:
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} sweep values must be unique, got {values}")
    return values


__all__ = [
    "DEFAULT_CONCURRENCY_COUNTS",
    "DEFAULT_CONCURRENCY_MODEL",
    "DEFAULT_CONCURRENCY_PLAN",
    "DEFAULT_LATENCY_FAMILY",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_QSTAR",
    "DEFAULT_SUBSCRIPTION_PLAN",
    "DEFAULT_WORKLOAD",
    "PHASE_CONCURRENCY",
    "PHASE_JOINT",
    "PHASE_QUOTA",
    "SECTION_NAME",
    "build_ablation_policy",
    "list_scenarios",
    "make_scenario",
    "make_scenarios",
    "policies_for_phase",
    "quota_limit_artifact_label",
    "run_ablation_policy",
    "run_effective_cost_cell",
]
