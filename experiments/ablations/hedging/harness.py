"""Harness for RouteWise hedging timing and backup-selection ablations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from experiments.ablations.hedging.policy import HedgingAblationPolicy
from experiments.ablations.hedging.presets import (
    DEFAULT_P_VALUES,
    make_ablation_presets,
    parse_ablation_policy_name,
    production_baseline_policy_name,
)
from experiments.simulation import common, hedging as hedging_section
from llm_routewise.sim.engine.simulator import Simulator

if TYPE_CHECKING:
    from llm_routewise.metrics import Run
    from llm_routewise.schemas import Request
    from llm_routewise.sim.world.scenarios import ScenarioConfig

SECTION_NAME = "hedging-ablation"
DEFAULT_OUTPUT_DIR = common.OUTPUT_DIR.parent / "ablations" / "hedging"
DEFAULT_WORKLOAD = hedging_section.DEFAULT_WORKLOAD
_COST_MULTIPLIER_BASIS = "mean_total_cost_usd"


def list_scenarios() -> tuple[str, ...]:
    """Return the default §2.2 hedging ablation scenarios."""
    return hedging_section.list_scenarios()


def make_scenarios() -> dict[str, ScenarioConfig]:
    """Build all hedging ablation scenarios."""
    return hedging_section.make_scenarios()


def make_scenario(name: str) -> ScenarioConfig:
    """Build one hedging ablation scenario by name."""
    return hedging_section.make_scenario(name)


def policies_for_ablation(
    *,
    alpha_values: tuple[float, ...] = DEFAULT_P_VALUES,
) -> tuple[str, ...]:
    """Return default hedging ablation policies."""
    return tuple(make_ablation_presets(alpha_values=alpha_values))


def build_ablation_policy(
    policy_name: str,
    *,
    presets: dict[str, dict[str, Any]],
    scenario: ScenarioConfig,
    requests: list[Request],
    seed: int,
) -> HedgingAblationPolicy:
    """Instantiate one HedgingAblationPolicy from an ablation preset."""
    try:
        preset = presets[policy_name]
    except KeyError as exc:
        known = ", ".join(sorted(presets))
        raise ValueError(f"unknown hedging ablation policy {policy_name!r}; known: {known}") from exc
    if preset.get("policy") != "HedgingAblationPolicy":
        raise ValueError(f"unsupported hedging ablation preset {policy_name!r}: {preset!r}")

    params = dict(preset.get("params", {}))
    if params.get("cost_envelope") == common.WORKLOAD_COST_ENVELOPE:
        params["cost_envelope"] = common.workload_cost_envelope(
            scenario.providers,
            requests,
        )
    params.setdefault("slo_ms", float(scenario.primary_slo_ms))
    return HedgingAblationPolicy(seed=seed, **params)


def run_ablation_policy(
    scenario: ScenarioConfig,
    requests: list[Request],
    policy_name: str,
    *,
    presets: dict[str, dict[str, Any]],
    seed: int,
    retain_records: bool = True,
) -> Run:
    """Run one hedging ablation policy."""
    policy = build_ablation_policy(
        policy_name,
        presets=presets,
        scenario=scenario,
        requests=requests,
        seed=seed,
    )
    simulator = Simulator(scenario=scenario, seed=seed, retain_records=retain_records)
    return simulator.run(requests, policy, policy_name=policy_name)


def run_hedging_ablation_cell(
    cell: common.SectionCell,
    presets: dict[str, dict[str, Any]],
    workload_dataset: str,
    duration_sec: float | None,
    max_requests: int | None,
    retain_records: bool,
) -> common.SectionCellResult:
    """Run one hedging ablation cell in a worker process."""
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
    """Run the hedging ablation harness."""
    parser = argparse.ArgumentParser(
        prog="routewise ablation hedging",
        description=__doc__,
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=list(list_scenarios()),
        help="Scenario to run. Repeat to run multiple. Defaults to all hedging scenarios.",
    )
    parser.add_argument(
        "--policy",
        action="append",
        help="Policy to run. Repeat to run multiple. Defaults to the core ablation grid.",
    )
    parser.add_argument(
        "--alpha",
        "--p",
        type=float,
        action="append",
        dest="alpha_values",
        help=f"RouteWise alpha value. Repeat to sweep. Defaults to {DEFAULT_P_VALUES}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        help=f"Seed to run. Repeat to run multiple. Defaults to {common.DEFAULT_SEEDS}.",
    )
    parser.add_argument(
        "--workload",
        default=DEFAULT_WORKLOAD,
        choices=common.WORKLOAD_CHOICES,
        help=f"Trace workload to replay. Defaults to {DEFAULT_WORKLOAD}.",
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
    alpha_values = tuple(args.alpha_values) if args.alpha_values else DEFAULT_P_VALUES
    scenarios = {
        name: make_scenario(name)
        for name in (tuple(args.scenario) if args.scenario else list_scenarios())
    }
    presets = make_ablation_presets(alpha_values=alpha_values)
    policies = tuple(args.policy) if args.policy else tuple(presets)
    unknown = [policy for policy in policies if policy not in presets]
    if unknown:
        known = ", ".join(sorted(presets))
        raise SystemExit(f"unknown hedging ablation policy {unknown[0]!r}; known: {known}")

    rows = common.run_section(
        section_name=SECTION_NAME,
        scenarios=scenarios,
        policies=policies,
        presets=presets,
        seeds=tuple(args.seed) if args.seed else common.DEFAULT_SEEDS,
        section_runners=_section_runners(policies, presets),
        parallel_cell_runner=run_hedging_ablation_cell,
        workload_dataset=args.workload,
        duration_sec=args.duration_sec,
        max_requests=args.max_requests,
        output_dir=args.output_dir,
        retain_records=False,
        jobs=args.jobs,
    )
    enriched_rows = _enrich_rows(
        rows,
        scenarios=scenarios,
        presets=presets,
    )
    common.write_json(args.output_dir / "summary.json", enriched_rows)
    _write_summary_csv(args.output_dir / "summary.csv", enriched_rows)
    print(
        json.dumps(
            {
                "section": SECTION_NAME,
                "rows": len(enriched_rows),
                "output_dir": str(args.output_dir),
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
        *,
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


def _enrich_rows(
    rows: list[dict[str, Any]],
    *,
    scenarios: dict[str, ScenarioConfig],
    presets: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        scenario = scenarios[row["scenario"]]
        meta = dict(scenario.metadata or {})
        preset = presets[row["policy"]]
        params = dict(preset["params"])
        dispatch_timing, backup_selection, alpha_value = parse_ablation_policy_name(row["policy"])
        merged = dict(row)
        merged.update(
            {
                "hedging_scenario": meta.get("hedging_scenario"),
                "source_latency_scenario": meta.get("source_latency_scenario"),
                "real_world_pool": meta.get("real_world_pool"),
                "latency_family": meta.get("latency_family"),
                "overlap_label": meta.get("overlap_label"),
                "slo_ms": meta.get("slo_ms"),
                "target_success_probability": meta.get("target_success_probability"),
                "routewise_alpha": float(params.get("alpha", alpha_value)),
                "dispatch_timing": dispatch_timing,
                "backup_selection": backup_selection,
                "backup_selection_semantics": (
                    "random_among_feasible_non_primary"
                    if backup_selection == "random_feasible"
                    else "probability_target_non_primary"
                ),
                "explorer_enabled": False,
                "learns_from_backup": False,
                "latency_profile_mode": params.get("latency_profile_mode", "observed"),
                "production_baseline_policy": production_baseline_policy_name(alpha=alpha_value),
                "cost_multiplier_basis": _COST_MULTIPLIER_BASIS,
            }
        )
        enriched.append(merged)

    baselines = {
        (row["scenario"], parse_ablation_policy_name(row["policy"])[2]): row
        for row in enriched
        if row["policy"] == production_baseline_policy_name(
            alpha=parse_ablation_policy_name(row["policy"])[2]
        )
    }
    for row in enriched:
        alpha_value = parse_ablation_policy_name(row["policy"])[2]
        baseline = baselines.get((row["scenario"], alpha_value))
        _add_production_deltas(row, baseline)
    return enriched


def _add_production_deltas(
    row: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> None:
    if baseline is None:
        row["p99_delta_vs_production_ms"] = None
        row["mean_ttft_delta_vs_production_ms"] = None
        row["p50_delta_vs_production_ms"] = None
        row["hedge_rate_delta_vs_production"] = None
        row["cost_multiplier_vs_production"] = None
        return
    row["p99_delta_vs_production_ms"] = row["p99_ms"] - baseline["p99_ms"]
    row["mean_ttft_delta_vs_production_ms"] = (
        row["mean_ttft_ms"] - baseline["mean_ttft_ms"]
    )
    row["p50_delta_vs_production_ms"] = row["p50_ms"] - baseline["p50_ms"]
    row["hedge_rate_delta_vs_production"] = row["hedge_rate"] - baseline["hedge_rate"]
    base_cost = _row_cost_for_multiplier(baseline)
    row_cost = _row_cost_for_multiplier(row)
    row["cost_multiplier_vs_production"] = (
        row_cost / base_cost if base_cost and base_cost > 0.0 else None
    )


def _row_cost_for_multiplier(row: dict[str, Any]) -> float:
    value = row.get(_COST_MULTIPLIER_BASIS)
    if value is None:
        raise ValueError(
            f"hedging ablation row missing {_COST_MULTIPLIER_BASIS!r}: "
            f"scenario={row.get('scenario')!r}, policy={row.get('policy')!r}"
        )
    return float(value)


_CSV_FIELDNAMES: tuple[str, ...] = (
    "scenario",
    "policy",
    "hedging_scenario",
    "source_latency_scenario",
    "real_world_pool",
    "latency_family",
    "overlap_label",
    "dispatch_timing",
    "backup_selection",
    "backup_selection_semantics",
    "production_baseline_policy",
    "routewise_alpha",
    "slo_ms",
    "target_success_probability",
    "explorer_enabled",
    "learns_from_backup",
    "latency_profile_mode",
    "seeds",
    "n_requests",
    "mean_ttft_ms",
    "p50_ms",
    "p90_ms",
    "p99_ms",
    "slo_violation_rate",
    "hedge_rate",
    "provider_mix",
    "mean_api_cost_usd",
    "mean_total_cost_usd",
    "api_cost_usd",
    "total_cost_usd",
    "p99_delta_vs_production_ms",
    "mean_ttft_delta_vs_production_ms",
    "p50_delta_vs_production_ms",
    "hedge_rate_delta_vs_production",
    "cost_multiplier_vs_production",
    "cost_multiplier_basis",
    "percentile_source",
    "histogram_bins",
)


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_CSV_FIELDNAMES))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(row[key], sort_keys=True)
                        if isinstance(row.get(key), (dict, list))
                        else row.get(key)
                    )
                    for key in _CSV_FIELDNAMES
                }
            )


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_WORKLOAD",
    "SECTION_NAME",
    "build_ablation_policy",
    "list_scenarios",
    "main",
    "make_scenario",
    "make_scenarios",
    "policies_for_ablation",
    "run_ablation_policy",
    "run_hedging_ablation_cell",
]
