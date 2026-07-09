"""Harness for the online latency-profile window-length ablation.

Sweeps the RouteWise rolling-profile window length against environments whose
provider TTFT distributions change periodically (staggered square wave), on
top of the §3 end-to-end RW8 scenario. The ``configured`` latency-profile
mode provides an oracle reference that reacts to every change instantly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from experiments.ablations.profile_window.presets import (
    DEFAULT_ALPHA_VALUES,
    DEFAULT_WINDOW_MINUTES,
    format_minutes,
    make_profile_window_presets,
    parse_minutes,
    parse_profile_window_policy_name,
)
from experiments.simulation import common, end_to_end
from routewise.sim.engine.simulator import Simulator
from routewise.sim.policies import build_policy
from routewise.sim.world.distributions import ScaledDistribution

if TYPE_CHECKING:
    from routewise.metrics import Run
    from routewise.schemas import Request
    from routewise.sim.world.providers import Provider
    from routewise.sim.world.scenarios import ScenarioConfig

SECTION_NAME = "profile-window-ablation"
PUBLIC_SCENARIO_TAG = "profile_window"
# Environment change period in minutes; 0 denotes the static environment.
DEFAULT_PERIOD_MINUTES: tuple[float, ...] = (0.0, 60.0, 30.0, 10.0, 5.0)
DEFAULT_MAGNITUDE = 3.0
STATIC_PERIOD_TOKEN = "static"
DEFAULT_WORKLOAD = end_to_end.DEFAULT_WORKLOAD
DEFAULT_OUTPUT_DIR = common.OUTPUT_DIR / "ablations" / "profile_window"

_LABEL_PREFIX = "rw8_shift__period="
_MAG_MARKER = "__mag="


def shift_artifact_label(period_min: float, magnitude: float) -> str:
    """Return the stable artifact label for one environment cell."""
    period_token = (
        STATIC_PERIOD_TOKEN if period_min <= 0.0 else f"{format_minutes(period_min)}m"
    )
    return f"{_LABEL_PREFIX}{period_token}{_MAG_MARKER}{format_minutes(magnitude)}"


def parse_shift_artifact_label(name: str) -> tuple[float, float]:
    """Parse ``(period_min, magnitude)`` back from an artifact label."""
    if not name.startswith(_LABEL_PREFIX) or _MAG_MARKER not in name:
        raise ValueError(f"unknown profile-window artifact label {name!r}")
    period_token, magnitude_token = name.removeprefix(_LABEL_PREFIX).split(_MAG_MARKER, 1)
    period_min = (
        0.0
        if period_token == STATIC_PERIOD_TOKEN
        else parse_minutes(period_token.removesuffix("m"))
    )
    return period_min, parse_minutes(magnitude_token)


def build_square_wave_schedule(
    base_dist: Any,
    *,
    anchor_sec: float,
    end_sec: float,
    period_sec: float,
    offset_sec: float,
    magnitude: float,
):
    """Return alternating degraded/baseline shift entries for one provider.

    The provider runs its baseline distribution before ``anchor_sec +
    offset_sec``, then toggles degraded/baseline every ``period_sec`` until
    ``end_sec``.
    """
    if period_sec <= 0.0:
        raise ValueError(f"period_sec must be > 0, got {period_sec}")
    degraded = ScaledDistribution(base=base_dist, scale=magnitude)
    entries: list[tuple[float, Any]] = []
    time_sec = float(anchor_sec) + float(offset_sec)
    degraded_phase = True
    while time_sec <= end_sec:
        entries.append((time_sec, degraded if degraded_phase else base_dist))
        degraded_phase = not degraded_phase
        time_sec += float(period_sec)
    return tuple(entries)


def apply_square_wave(
    providers: list[Provider],
    *,
    anchor_sec: float,
    end_sec: float,
    period_sec: float,
    magnitude: float,
) -> None:
    """Attach staggered square-wave TTFT schedules to every provider.

    Provider ``i`` of ``N`` gets phase offset ``i * (2 * period) / N`` so the
    degrade onsets spread across one full cycle and the identity of the
    lowest-latency provider genuinely changes over time.
    """
    count = len(providers)
    for index, provider in enumerate(providers):
        offset_sec = index * (2.0 * period_sec) / count
        provider.ttft_shift_schedule = build_square_wave_schedule(
            provider.ttft_dist,
            anchor_sec=anchor_sec,
            end_sec=end_sec,
            period_sec=period_sec,
            offset_sec=offset_sec,
            magnitude=magnitude,
        )


def make_scenario(
    name: str,
    *,
    requests: list[Request],
    slo_ms: float = end_to_end.DEFAULT_SLO_MS,
) -> ScenarioConfig:
    """Rebuild one environment scenario from its stable artifact label.

    ``requests`` anchor the square wave at the first trace timestamp; trace
    timestamps are not zero-based.
    """
    period_min, magnitude = parse_shift_artifact_label(name)
    scenario = end_to_end.make_scenario(end_to_end.RW8_SCENARIO_NAME, slo_ms=slo_ms)
    if period_min > 0.0:
        if not requests:
            raise ValueError("square-wave scenarios need a non-empty workload to anchor")
        apply_square_wave(
            scenario.providers,
            anchor_sec=float(requests[0].timestamp),
            end_sec=float(requests[-1].timestamp),
            period_sec=float(period_min) * 60.0,
            magnitude=magnitude,
        )
    scenario.name = name
    scenario.metadata = {
        **scenario.metadata,
        "public_scenario": PUBLIC_SCENARIO_TAG,
        "artifact_label": name,
        "shift_period_min": period_min,
        "shift_period_sec": period_min * 60.0,
        "shift_magnitude": magnitude,
        "shift_stagger": "uniform_full_cycle",
    }
    return scenario


def make_scenarios(
    *,
    period_minutes: tuple[float, ...] = DEFAULT_PERIOD_MINUTES,
    magnitude: float = DEFAULT_MAGNITUDE,
    requests: list[Request],
    slo_ms: float = end_to_end.DEFAULT_SLO_MS,
) -> dict[str, ScenarioConfig]:
    """Build all environment scenarios keyed by artifact label."""
    if magnitude <= 1.0:
        raise ValueError(f"magnitude must be > 1 to degrade latency, got {magnitude}")
    if len(set(period_minutes)) != len(period_minutes):
        raise ValueError(f"period sweep values must be unique, got {period_minutes}")
    scenarios: dict[str, ScenarioConfig] = {}
    for period_min in period_minutes:
        label = shift_artifact_label(period_min, magnitude)
        scenarios[label] = make_scenario(label, requests=requests, slo_ms=slo_ms)
    return scenarios


def run_profile_window_policy(
    scenario: ScenarioConfig,
    requests: list[Request],
    policy_name: str,
    *,
    presets: dict[str, dict[str, Any]],
    seed: int,
    retain_records: bool = True,
) -> Run:
    """Run one preset and attach profile diagnostics to the returned Run."""
    materialized = common.materialize_policy_presets(
        presets,
        policy_name=policy_name,
        scenario=scenario,
        requests=requests,
    )
    policy = build_policy(policy_name, presets=materialized, seed=seed)
    simulator = Simulator(scenario=scenario, seed=seed, retain_records=retain_records)
    run = simulator.run(requests, policy, policy_name=policy_name)
    router = getattr(policy, "router", None)
    if router is not None and router.beliefs.mode == "observed":
        run.extra_metrics = {"profile_fallback_rate": router.beliefs.fallback_rate()}
    return run


def run_profile_window_cell(
    cell: common.SectionCell,
    presets: dict[str, dict[str, Any]],
    workload_dataset: str,
    duration_sec: float | None,
    max_requests: int | None,
    retain_records: bool,
) -> common.SectionCellResult:
    """Run one scenario-policy-seed cell in a worker process."""
    requests = common.load_workload(
        dataset=workload_dataset,
        duration_sec=duration_sec,
        max_requests=max_requests,
    )
    scenario = make_scenario(cell.scenario_name, requests=requests)
    run = run_profile_window_policy(
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


def enrich_rows(
    rows: list[dict[str, Any]],
    scenarios: dict[str, ScenarioConfig],
    presets: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add sweep coordinates (window, period, magnitude) to summary rows."""
    enriched: list[dict[str, Any]] = []
    for row in rows:
        scenario = scenarios[row["scenario"]]
        parsed = parse_profile_window_policy_name(row["policy"])
        params = presets[row["policy"]].get("params", {})
        window_min = parsed["window_min"]
        enriched.append(
            {
                **row,
                "policy_family": parsed["family"],
                "alpha": parsed["alpha"],
                "window_min": window_min,
                "window_sec": None if window_min is None else window_min * 60.0,
                "latency_profile_mode": params.get("latency_profile_mode"),
                "shift_period_min": scenario.metadata.get("shift_period_min"),
                "shift_period_sec": scenario.metadata.get("shift_period_sec"),
                "shift_magnitude": scenario.metadata.get("shift_magnitude"),
            }
        )
    return enriched


def write_enriched_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write enriched rows with union fieldnames and JSON-encoded maps."""
    import csv

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(row[key], sort_keys=True)
                        if isinstance(row.get(key), (dict, list))
                        else row.get(key)
                    )
                    for key in fieldnames
                }
            )


def main(argv: list[str] | None = None) -> int:
    """Run the profile-window ablation grid."""
    parser = argparse.ArgumentParser(
        prog="routewise ablation profile-window",
        description=__doc__,
    )
    parser.add_argument(
        "--window-min",
        type=float,
        action="append",
        dest="window_minutes",
        help=(
            "Rolling-profile window length in minutes. Repeat to sweep. "
            f"Defaults to {DEFAULT_WINDOW_MINUTES}."
        ),
    )
    parser.add_argument(
        "--period-min",
        type=float,
        action="append",
        dest="period_minutes",
        help=(
            "Environment change period in minutes; 0 means static. Repeat to "
            f"sweep. Defaults to {DEFAULT_PERIOD_MINUTES}."
        ),
    )
    parser.add_argument(
        "--magnitude",
        type=float,
        default=DEFAULT_MAGNITUDE,
        help=f"Degraded-phase TTFT scale factor. Defaults to {DEFAULT_MAGNITUDE}.",
    )
    parser.add_argument(
        "--alpha",
        "--p",
        type=float,
        action="append",
        dest="alpha_values",
        help=f"RouteWise alpha value. Repeat to sweep. Defaults to {DEFAULT_ALPHA_VALUES}.",
    )
    parser.add_argument(
        "--no-oracle",
        action="store_true",
        help="Skip the configured-mode oracle reference policies.",
    )
    parser.add_argument(
        "--predictor",
        default=common.DEFAULT_OUTPUT_PREDICTOR,
        help=f"Output-length predictor. Defaults to {common.DEFAULT_OUTPUT_PREDICTOR}.",
    )
    parser.add_argument(
        "--slo-ms",
        type=float,
        default=end_to_end.DEFAULT_SLO_MS,
        help=f"Single-request SLO threshold in ms. Defaults to {end_to_end.DEFAULT_SLO_MS}.",
    )
    parser.add_argument(
        "--workload",
        default=DEFAULT_WORKLOAD,
        choices=end_to_end.WORKLOAD_CHOICES,
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
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for metadata.json, summary.json, and summary.csv.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel scenario-policy-seed cells to run. Defaults to 1.",
    )

    args = parser.parse_args(argv)
    window_minutes = tuple(args.window_minutes or DEFAULT_WINDOW_MINUTES)
    period_minutes = tuple(
        args.period_minutes if args.period_minutes is not None else DEFAULT_PERIOD_MINUTES
    )
    alpha_values = tuple(args.alpha_values or DEFAULT_ALPHA_VALUES)
    if len(set(window_minutes)) != len(window_minutes):
        raise SystemExit(f"window sweep values must be unique, got {window_minutes}")
    if any(value <= 0 for value in window_minutes):
        raise SystemExit(f"window minutes must be > 0, got {window_minutes}")

    requests = common.load_workload(
        dataset=args.workload,
        duration_sec=args.duration_sec,
        max_requests=args.max_requests,
    )
    scenarios = make_scenarios(
        period_minutes=period_minutes,
        magnitude=args.magnitude,
        requests=requests,
        slo_ms=args.slo_ms,
    )
    presets = make_profile_window_presets(
        window_minutes=window_minutes,
        alpha_values=alpha_values,
        include_oracle=not args.no_oracle,
        output_predictor=args.predictor,
    )
    policies = tuple(presets)

    rows = common.run_section(
        section_name=SECTION_NAME,
        scenarios=scenarios,
        policies=policies,
        presets=presets,
        seeds=tuple(args.seed) if args.seed else common.DEFAULT_SEEDS,
        section_runners={
            policy: _make_serial_runner(policy, presets) for policy in policies
        },
        parallel_cell_runner=run_profile_window_cell,
        workload_dataset=args.workload,
        duration_sec=args.duration_sec,
        max_requests=args.max_requests,
        output_dir=args.output_dir,
        retain_records=False,
        jobs=args.jobs,
    )
    enriched = enrich_rows(rows, scenarios, presets)
    common.write_json(args.output_dir / "summary.json", enriched)
    write_enriched_csv(args.output_dir / "summary.csv", enriched)
    print(
        json.dumps(
            {
                "section": SECTION_NAME,
                "rows": len(enriched),
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


def _make_serial_runner(policy_name: str, presets: dict[str, dict[str, Any]]):
    def run(
        scenario: ScenarioConfig,
        requests: list[Request],
        seed: int,
        retain_records: bool = True,
    ) -> Run:
        return run_profile_window_policy(
            scenario,
            requests,
            policy_name,
            presets=presets,
            seed=seed,
            retain_records=retain_records,
        )

    return run


__all__ = [
    "DEFAULT_MAGNITUDE",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PERIOD_MINUTES",
    "DEFAULT_WORKLOAD",
    "SECTION_NAME",
    "apply_square_wave",
    "build_square_wave_schedule",
    "enrich_rows",
    "main",
    "make_scenario",
    "make_scenarios",
    "parse_shift_artifact_label",
    "run_profile_window_cell",
    "run_profile_window_policy",
    "shift_artifact_label",
    "write_enriched_csv",
]

if __name__ == "__main__":
    raise SystemExit(main())
