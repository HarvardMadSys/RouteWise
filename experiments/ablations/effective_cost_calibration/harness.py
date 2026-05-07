"""Effective-cost L/U envelope calibration ablation harness."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from experiments.ablations.effective_cost import harness as curve_harness
from experiments.ablations.effective_cost.policy import LPOnlyAblationPolicy
from experiments.ablations.effective_cost.presets import DEFAULT_CONCURRENCY_CURVE
from experiments.ablations.effective_cost_calibration.envelope import (
    API_REFERENCES,
    DEFAULT_API_REFERENCE,
    DEFAULT_PERCENTILE_ENVELOPE,
    PERCENTILE_ENVELOPES,
    ApiReference,
    EnvelopeSpec,
    PercentileEnvelope,
    workload_cost_envelope,
)
from experiments.simulation import common
from experiments.simulation.latency_profiles import load_pooled_distribution
from experiments.subscriptions import load_subscription_plans
from rwsim.engine.simulator import Simulator
from rwsim.world.scenarios import ScenarioConfig

if TYPE_CHECKING:
    from experiments.ablations.effective_cost.curves import ScarcityCurve
    from rwsim.metrics import Run
    from rwsim.schemas import Request

SECTION_NAME = "effective-cost-calibration"
DEFAULT_QUOTA_CURVE: ScarcityCurve = "exp_lu"
DEFAULT_P_VALUES = (0.5,)
DEFAULT_SWEEP = "percentile"
SWEEPS = ("percentile", "reference", "all", "cross-product")
Sweep = Literal["percentile", "reference", "all", "cross-product"]
DEFAULT_OUTPUT_DIR = common.OUTPUT_DIR / "ablations" / "effective_cost_calibration"
API_SURFACE = "quota_plus_api_cheap"
_CALIBRATION_COLUMNS = (
    "api_reference",
    "percentile_envelope",
    "envelope_L",
    "envelope_U",
)


def list_scenarios() -> tuple[str, ...]:
    """Return the locked default calibration scenario labels."""
    return tuple(make_scenarios().keys())


def make_scenarios(
    *,
    subscription_plan: str = curve_harness.DEFAULT_SUBSCRIPTION_PLAN,
    qstar: int | tuple[int, ...] = curve_harness.DEFAULT_QSTAR,
    latency_family: str = curve_harness.DEFAULT_LATENCY_FAMILY,
) -> dict[str, ScenarioConfig]:
    """Build clean quota + cheap-API scenarios for envelope calibration."""
    scenarios: dict[str, ScenarioConfig] = {}
    for value in _positive_unique_values("qstar", qstar):
        scenario = _make_clean_quota_scenario_for_plan(
            subscription_plan,
            value,
            latency_family=latency_family,
        )
        scenarios[scenario.name] = scenario
    return scenarios


def make_scenario(name: str) -> ScenarioConfig:
    """Rebuild a calibration scenario from its stable artifact label."""
    parsed = _parse_clean_quota_artifact_label(name)
    if parsed is None:
        raise ValueError(
            "unknown effective-cost calibration scenario "
            f"{name!r}; expected label like "
            "'quota_clean__plan=chutes__n=16'"
        )
    plan_id, subscription_count, latency_family = parsed
    return _make_clean_quota_scenario_for_plan(
        plan_id,
        subscription_count,
        latency_family=latency_family,
    )


def calibration_specs(
    *,
    sweep: Sweep = DEFAULT_SWEEP,
    api_references: tuple[ApiReference, ...] = API_REFERENCES,
    percentile_envelopes: tuple[PercentileEnvelope, ...] = PERCENTILE_ENVELOPES,
) -> tuple[EnvelopeSpec, ...]:
    """Return ordered calibration specs for one sweep mode."""
    _validate_unique("api reference", api_references)
    _validate_unique("percentile envelope", percentile_envelopes)
    base_reference = (
        DEFAULT_API_REFERENCE
        if DEFAULT_API_REFERENCE in api_references
        else api_references[0]
    )
    base_percentile = (
        DEFAULT_PERCENTILE_ENVELOPE
        if DEFAULT_PERCENTILE_ENVELOPE in percentile_envelopes
        else percentile_envelopes[0]
    )
    if sweep == "percentile":
        return tuple(
            EnvelopeSpec(
                api_reference=base_reference,
                percentile_envelope=percentile,
            )
            for percentile in percentile_envelopes
        )
    if sweep == "reference":
        return tuple(
            EnvelopeSpec(
                api_reference=reference,
                percentile_envelope=base_percentile,
            )
            for reference in api_references
        )
    if sweep == "all":
        specs = [
            EnvelopeSpec(
                api_reference=base_reference,
                percentile_envelope=percentile,
            )
            for percentile in percentile_envelopes
        ]
        specs.extend(
            EnvelopeSpec(
                api_reference=reference,
                percentile_envelope=base_percentile,
            )
            for reference in api_references
        )
        return _dedupe_specs(specs)
    if sweep == "cross-product":
        return tuple(
            EnvelopeSpec(api_reference=reference, percentile_envelope=percentile)
            for reference in api_references
            for percentile in percentile_envelopes
        )
    known = ", ".join(SWEEPS)
    raise ValueError(f"unknown calibration sweep {sweep!r}; known: {known}")


def calibration_policy_name(
    spec: EnvelopeSpec,
    *,
    quota_curve: ScarcityCurve = DEFAULT_QUOTA_CURVE,
    p: float = DEFAULT_P_VALUES[0],
) -> str:
    """Return a stable policy name for one calibration spec."""
    return (
        "effective_cost_calibration__"
        f"{spec.label}__q={quota_curve}__{common.p_label(p)}"
    )


def make_calibration_presets(
    *,
    specs: tuple[EnvelopeSpec, ...] | None = None,
    quota_curve: ScarcityCurve = DEFAULT_QUOTA_CURVE,
    p_values: tuple[float, ...] = DEFAULT_P_VALUES,
    concurrency_curve: ScarcityCurve = DEFAULT_CONCURRENCY_CURVE,
) -> dict[str, dict[str, Any]]:
    """Build ablation-local preset metadata for envelope calibration sweeps."""
    if specs is None:
        specs = calibration_specs()
    presets: dict[str, dict[str, Any]] = {}
    for p in p_values:
        for spec in specs:
            name = calibration_policy_name(spec, quota_curve=quota_curve, p=p)
            presets[name] = {
                "policy": "LPOnlyAblationPolicy",
                "params": {
                    "quota_curve": quota_curve,
                    "concurrency_curve": concurrency_curve,
                    "p": float(p),
                    "cost_envelope_spec": spec,
                },
            }
    return presets


def policies_for_section() -> tuple[str, ...]:
    """Return the default calibration policy list."""
    return tuple(make_calibration_presets())


def build_calibration_policy(
    policy_name: str,
    *,
    presets: dict[str, dict[str, Any]],
    scenario: ScenarioConfig,
    requests: list[Request],
    seed: int,
) -> LPOnlyAblationPolicy:
    """Instantiate LPOnlyAblationPolicy with a materialized calibration envelope."""
    try:
        preset = presets[policy_name]
    except KeyError as exc:
        known = ", ".join(sorted(presets))
        raise ValueError(
            f"unknown effective-cost calibration policy {policy_name!r}; known: {known}"
        ) from exc
    if preset.get("policy") != "LPOnlyAblationPolicy":
        raise ValueError(
            f"unsupported effective-cost calibration preset {policy_name!r}: {preset!r}"
        )

    params = dict(preset.get("params", {}))
    spec = params.pop("cost_envelope_spec", None)
    if not isinstance(spec, EnvelopeSpec):
        raise ValueError(
            f"effective-cost calibration preset {policy_name!r} lacks EnvelopeSpec"
        )
    params["cost_envelope"] = workload_cost_envelope(
        scenario.providers,
        requests,
        spec=spec,
    )
    return LPOnlyAblationPolicy(seed=seed, **params)


def run_calibration_policy(
    scenario: ScenarioConfig,
    requests: list[Request],
    policy_name: str,
    *,
    presets: dict[str, dict[str, Any]],
    seed: int,
    retain_records: bool = True,
) -> Run:
    """Run one envelope-calibration LP-only policy."""
    policy = build_calibration_policy(
        policy_name,
        presets=presets,
        scenario=scenario,
        requests=requests,
        seed=seed,
    )
    simulator = Simulator(
        scenario=scenario,
        seed=seed,
        retain_records=retain_records,
    )
    return simulator.run(requests, policy, policy_name=policy_name)


def run_calibration_cell(
    cell: common.SectionCell,
    presets: dict[str, dict[str, Any]],
    workload_dataset: str,
    duration_sec: float | None,
    max_requests: int | None,
    retain_records: bool,
) -> common.SectionCellResult:
    """Run one effective-cost calibration cell in a worker process."""
    scenario = make_scenario(cell.scenario_name)
    requests = common.load_workload(
        dataset=workload_dataset,
        duration_sec=duration_sec,
        max_requests=max_requests,
    )
    run = run_calibration_policy(
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


def clean_quota_artifact_label(
    plan_id: str,
    subscription_count: int,
    *,
    latency_family: str = curve_harness.DEFAULT_LATENCY_FAMILY,
) -> str:
    """Return the stable label for the clean quota + cheap-API surface."""
    label = f"quota_clean__plan={plan_id}__n={subscription_count}"
    if latency_family != curve_harness.DEFAULT_LATENCY_FAMILY:
        label += f"__latency={latency_family}"
    return label


def _make_clean_quota_scenario_for_plan(
    plan_id: str,
    subscription_count: int,
    *,
    latency_family: str = curve_harness.DEFAULT_LATENCY_FAMILY,
) -> ScenarioConfig:
    _validate_latency_family(latency_family)
    plans = load_subscription_plans()
    try:
        plan = plans[plan_id]
    except KeyError as exc:
        known = ", ".join(sorted(plans))
        raise ValueError(f"unknown subscription plan {plan_id!r}; known plans: {known}") from exc
    if "cost_layer_quota" not in plan.eligible_sections:
        raise ValueError(
            f"subscription plan {plan.plan_id!r} is not eligible for quota calibration"
        )

    ttft_dist = load_pooled_distribution("rw8_pooled") if latency_family == "real_world" else None
    providers = [
        common.make_quota_provider(
            f"{plan.plan_id}_quota",
            plan=plan,
            subscription_count=subscription_count,
            latency_family=(
                curve_harness.DEFAULT_LATENCY_FAMILY
                if ttft_dist is not None
                else latency_family
            ),
        ),
        common.make_api_provider(
            "api_cheap",
            cost_per_million_tokens=common.COST_RATIO_PER_MILLION[0],
            latency_family=(
                curve_harness.DEFAULT_LATENCY_FAMILY
                if ttft_dist is not None
                else latency_family
            ),
        ),
    ]
    if ttft_dist is not None:
        for provider in providers:
            provider.ttft_dist = ttft_dist

    label = clean_quota_artifact_label(
        plan.plan_id,
        subscription_count,
        latency_family=latency_family,
    )
    quota_window_text = ", ".join(
        f"{window.quota_requests * subscription_count:g}/{window.quota_window_sec:g}s"
        for window in plan.quota_windows
    )
    latency_text = (
        "real-world pooled rw8_pooled"
        if latency_family == "real_world"
        else f"{latency_family}, P50={common.COST_LAYER_P50_MS:.0f}ms"
    )
    return ScenarioConfig(
        name=label,
        description=(
            "Effective-cost calibration scenario: "
            f"{plan.display_name} x{subscription_count} ({quota_window_text}) "
            "as one aggregate quota provider plus exactly one cheap on-demand "
            f"API fallback, TTFT={latency_text}."
        ),
        providers=providers,
        arrival_process="trace",
        primary_slo_ms=2000.0,
        metadata={
            "public_scenario": "quota_clean",
            "artifact_label": label,
            "api_surface": API_SURFACE,
            "subscription_plan": plan.plan_id,
            "subscription_plan_display_name": plan.display_name,
            "subscription_count": subscription_count,
            "latency_family": latency_family,
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


def calibration_envelope_records(
    *,
    scenarios: dict[str, ScenarioConfig],
    policies: tuple[str, ...],
    presets: dict[str, dict[str, Any]],
    requests: list[Request],
) -> list[dict[str, Any]]:
    """Return materialized L/U metadata for each scenario-policy pair."""
    records: list[dict[str, Any]] = []
    for scenario in scenarios.values():
        for policy in policies:
            spec = _preset_envelope_spec(policy, presets=presets)
            L, U = workload_cost_envelope(scenario.providers, requests, spec=spec)
            records.append(
                {
                    "scenario": scenario.name,
                    "policy": policy,
                    "api_reference": spec.api_reference,
                    "percentile_envelope": spec.percentile_envelope,
                    "envelope_L": L,
                    "envelope_U": U,
                }
            )
    return records


def enrich_calibration_rows(
    rows: list[dict[str, Any]],
    *,
    scenarios: dict[str, ScenarioConfig],
    policies: tuple[str, ...],
    presets: dict[str, dict[str, Any]],
    requests: list[Request],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Attach numeric L/U envelope columns to summary rows."""
    records = calibration_envelope_records(
        scenarios=scenarios,
        policies=policies,
        presets=presets,
        requests=requests,
    )
    by_key = {
        (record["scenario"], record["policy"]): record
        for record in records
    }
    enriched: list[dict[str, Any]] = []
    for row in rows:
        key = (row.get("scenario"), row.get("policy"))
        try:
            record = by_key[key]
        except KeyError as exc:
            raise ValueError(
                "cannot attach calibration envelope to summary row for "
                f"scenario={key[0]!r}, policy={key[1]!r}"
            ) from exc
        enriched.append(
            {
                **row,
                **{column: record[column] for column in _CALIBRATION_COLUMNS},
            }
        )
    return enriched, records


def main(argv: list[str] | None = None) -> int:
    """Run the envelope calibration ablation harness."""
    parser = argparse.ArgumentParser(
        prog="routewise ablation effective-cost-calibration",
        description=__doc__,
    )
    parser.add_argument(
        "--sweep",
        default=DEFAULT_SWEEP,
        choices=SWEEPS,
        help=(
            "Calibration sweep shape. 'all' runs the two one-dimensional sweeps "
            "and de-duplicates the default point."
        ),
    )
    parser.add_argument(
        "--api-reference",
        action="append",
        choices=API_REFERENCES,
        dest="api_references",
        help="API reference to include. Repeat to select a subset.",
    )
    parser.add_argument(
        "--percentile-envelope",
        action="append",
        choices=PERCENTILE_ENVELOPES,
        dest="percentile_envelopes",
        help="Percentile envelope to include. Repeat to select a subset.",
    )
    parser.add_argument(
        "--curve",
        default=DEFAULT_QUOTA_CURVE,
        choices=("exp_lu", "linear_lu", "constant_l", "constant_u"),
        help=f"Quota scarcity curve to hold fixed. Defaults to {DEFAULT_QUOTA_CURVE}.",
    )
    parser.add_argument(
        "--p",
        type=float,
        action="append",
        dest="p_values",
        help=f"LP budget p value. Repeat to sweep. Defaults to {DEFAULT_P_VALUES}.",
    )
    parser.add_argument(
        "--subscription-plan",
        default=curve_harness.DEFAULT_SUBSCRIPTION_PLAN,
        help=(
            "Quota subscription plan id. Defaults to "
            f"{curve_harness.DEFAULT_SUBSCRIPTION_PLAN}."
        ),
    )
    parser.add_argument(
        "--qstar",
        type=int,
        action="append",
        dest="qstar_values",
        help=f"Quota subscription count q*. Repeat to sweep. Defaults to {curve_harness.DEFAULT_QSTAR}.",
    )
    parser.add_argument(
        "--latency-family",
        default=curve_harness.DEFAULT_LATENCY_FAMILY,
        choices=("uniform", "normal", "heavy_tail", "real_world"),
        help=f"Quota scenario TTFT family. Defaults to {curve_harness.DEFAULT_LATENCY_FAMILY}.",
    )
    parser.add_argument(
        "--workload",
        default=curve_harness.DEFAULT_WORKLOAD,
        choices=("burstgpt", "sharegpt_burstgpt"),
        help=(
            "Trace workload to replay. Defaults to the workload used by the "
            f"existing curve-shape result: {curve_harness.DEFAULT_WORKLOAD}."
        ),
    )
    parser.add_argument("--duration-sec", type=float, help="Optional trace duration cap.")
    parser.add_argument("--max-requests", type=int, help="Optional request cap for smoke runs.")
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        help=f"Seed to run. Repeat to run multiple seeds. Defaults to {common.DEFAULT_SEEDS}.",
    )
    parser.add_argument("--jobs", type=int, default=1, help="Parallel worker count.")
    parser.add_argument("--output-dir", type=Path, help="Directory for artifacts.")

    args = parser.parse_args(argv)
    api_references = tuple(args.api_references) if args.api_references else API_REFERENCES
    percentile_envelopes = (
        tuple(args.percentile_envelopes)
        if args.percentile_envelopes
        else PERCENTILE_ENVELOPES
    )
    specs = calibration_specs(
        sweep=args.sweep,
        api_references=api_references,
        percentile_envelopes=percentile_envelopes,
    )
    p_values = tuple(args.p_values) if args.p_values else DEFAULT_P_VALUES
    qstar_values = (
        tuple(args.qstar_values) if args.qstar_values else (curve_harness.DEFAULT_QSTAR,)
    )
    scenarios = make_scenarios(
        subscription_plan=args.subscription_plan,
        qstar=qstar_values,
        latency_family=args.latency_family,
    )
    presets = make_calibration_presets(
        specs=specs,
        quota_curve=args.curve,
        p_values=p_values,
    )
    policies = tuple(presets)
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR

    rows = common.run_section(
        section_name=SECTION_NAME,
        scenarios=scenarios,
        policies=policies,
        presets=presets,
        seeds=tuple(args.seed) if args.seed else common.DEFAULT_SEEDS,
        section_runners=_section_runners(policies, presets),
        parallel_cell_runner=run_calibration_cell,
        workload_dataset=args.workload,
        duration_sec=args.duration_sec,
        max_requests=args.max_requests,
        output_dir=output_dir,
        retain_records=False,
        jobs=args.jobs,
    )
    if rows:
        rows = _write_calibration_outputs(
            root=output_dir,
            rows=rows,
            scenarios=scenarios,
            policies=policies,
            presets=presets,
            workload_dataset=args.workload,
            duration_sec=args.duration_sec,
            max_requests=args.max_requests,
        )
    print(
        json.dumps(
            {
                "section": SECTION_NAME,
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
    return {policy: _runner_for_policy(policy, presets) for policy in policies}


def _runner_for_policy(policy_name: str, presets: dict[str, dict[str, Any]]):
    def run(
        scenario: ScenarioConfig,
        requests: list[Request],
        seed: int,
        retain_records: bool = True,
    ) -> Run:
        return run_calibration_policy(
            scenario,
            requests,
            policy_name,
            presets=presets,
            seed=seed,
            retain_records=retain_records,
        )

    return run


def _parse_clean_quota_artifact_label(name: str) -> tuple[str, int, str] | None:
    prefix = "quota_clean__plan="
    marker = "__n="
    if not name.startswith(prefix) or marker not in name:
        return None
    plan_part, rest = name.removeprefix(prefix).rsplit(marker, 1)
    count_part = rest
    latency_family = curve_harness.DEFAULT_LATENCY_FAMILY
    latency_marker = "__latency="
    if latency_marker in rest:
        count_part, latency_family = rest.split(latency_marker, 1)
    try:
        count = int(count_part)
    except ValueError as exc:
        raise ValueError(f"invalid clean quota scenario label {name!r}") from exc
    return plan_part, count, latency_family


def _positive_unique_values(label: str, values: int | tuple[int, ...]) -> tuple[int, ...]:
    raw_values = (values,) if isinstance(values, int) else tuple(values)
    if not raw_values:
        raise ValueError(f"{label} sweep must contain at least one value")
    unique_values = tuple(dict.fromkeys(int(value) for value in raw_values))
    invalid = [value for value in unique_values if value <= 0]
    if invalid:
        raise ValueError(f"{label} values must be > 0, got {invalid[0]}")
    return unique_values


def _validate_latency_family(latency_family: str) -> None:
    known = ("uniform", "normal", "heavy_tail", "real_world")
    if latency_family not in known:
        raise ValueError(
            f"unknown calibration latency family {latency_family!r}; "
            f"known: {', '.join(known)}"
        )


def _write_calibration_outputs(
    *,
    root: Path,
    rows: list[dict[str, Any]],
    scenarios: dict[str, ScenarioConfig],
    policies: tuple[str, ...],
    presets: dict[str, dict[str, Any]],
    workload_dataset: str,
    duration_sec: float | None,
    max_requests: int | None,
) -> list[dict[str, Any]]:
    """Rewrite section artifacts with calibration-specific L/U fields."""
    requests = common.load_workload(
        dataset=workload_dataset,
        duration_sec=duration_sec,
        max_requests=max_requests,
    )
    enriched_rows, envelope_records = enrich_calibration_rows(
        rows,
        scenarios=scenarios,
        policies=policies,
        presets=presets,
        requests=requests,
    )

    metadata_path = root / "metadata.json"
    metadata = _read_json_object(metadata_path)
    metadata["calibration_envelopes"] = envelope_records
    common.write_json(metadata_path, metadata)
    common.write_json(root / "summary.json", enriched_rows)
    summary_csv_path = root / "summary.csv"
    _write_calibration_summary_csv(
        summary_csv_path,
        enriched_rows,
        preferred_fieldnames=_read_csv_header(summary_csv_path),
    )
    return enriched_rows


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}, got {type(payload).__name__}")
    return payload


def _write_calibration_summary_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    preferred_fieldnames: tuple[str, ...] = (),
) -> None:
    """Write summary rows while preserving calibration-only columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = list(preferred_fieldnames)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _read_csv_header(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        try:
            return tuple(next(csv.reader(handle)))
        except StopIteration:
            return ()


def _preset_envelope_spec(
    policy_name: str,
    *,
    presets: dict[str, dict[str, Any]],
) -> EnvelopeSpec:
    try:
        preset = presets[policy_name]
    except KeyError as exc:
        known = ", ".join(sorted(presets))
        raise ValueError(
            f"unknown effective-cost calibration policy {policy_name!r}; known: {known}"
        ) from exc
    params = preset.get("params", {})
    spec = params.get("cost_envelope_spec")
    if not isinstance(spec, EnvelopeSpec):
        raise ValueError(
            f"effective-cost calibration preset {policy_name!r} lacks EnvelopeSpec"
        )
    return spec


def _dedupe_specs(specs: list[EnvelopeSpec]) -> tuple[EnvelopeSpec, ...]:
    return tuple(dict.fromkeys(specs))


def _validate_unique(label: str, values: tuple[str, ...]) -> None:
    if not values:
        raise ValueError(f"{label} sweep must contain at least one value")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} sweep values must be unique, got {values}")


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_P_VALUES",
    "DEFAULT_QUOTA_CURVE",
    "DEFAULT_SWEEP",
    "SECTION_NAME",
    "SWEEPS",
    "build_calibration_policy",
    "calibration_envelope_records",
    "calibration_policy_name",
    "calibration_specs",
    "enrich_calibration_rows",
    "list_scenarios",
    "make_calibration_presets",
    "make_scenario",
    "make_scenarios",
    "policies_for_section",
    "run_calibration_cell",
    "run_calibration_policy",
]
