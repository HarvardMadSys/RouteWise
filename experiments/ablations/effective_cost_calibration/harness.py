"""Effective-cost L/U envelope calibration ablation harness."""

from __future__ import annotations

import argparse
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
from rwsim.engine.simulator import Simulator

if TYPE_CHECKING:
    from experiments.ablations.effective_cost.curves import ScarcityCurve
    from rwsim.metrics import Run
    from rwsim.schemas import Request
    from rwsim.world.scenarios import ScenarioConfig

SECTION_NAME = "effective-cost-calibration"
DEFAULT_QUOTA_CURVE: ScarcityCurve = "exp_lu"
DEFAULT_P_VALUES = (0.5,)
DEFAULT_SWEEP = "all"
SWEEPS = ("percentile", "reference", "all", "cross-product")
Sweep = Literal["percentile", "reference", "all", "cross-product"]
DEFAULT_OUTPUT_DIR = common.OUTPUT_DIR / "ablations" / "effective_cost_calibration"


def list_scenarios() -> tuple[str, ...]:
    """Return the locked default calibration scenario labels."""
    return tuple(make_scenarios().keys())


def make_scenarios(
    *,
    subscription_plan: str = curve_harness.DEFAULT_SUBSCRIPTION_PLAN,
    qstar: int | tuple[int, ...] = curve_harness.DEFAULT_QSTAR,
    latency_family: str = curve_harness.DEFAULT_LATENCY_FAMILY,
) -> dict[str, ScenarioConfig]:
    """Build the quota-only scenario set used by envelope calibration."""
    return curve_harness.make_scenarios(
        phase=curve_harness.PHASE_QUOTA,
        subscription_plan=subscription_plan,
        qstar=qstar,
        latency_family=latency_family,
    )


def make_scenario(name: str) -> ScenarioConfig:
    """Rebuild a calibration scenario from its stable artifact label."""
    return curve_harness.make_scenario(name)


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
    "calibration_policy_name",
    "calibration_specs",
    "list_scenarios",
    "make_calibration_presets",
    "make_scenario",
    "make_scenarios",
    "policies_for_section",
    "run_calibration_cell",
    "run_calibration_policy",
]
