"""§2.2 hedging simulator section.

Same-cost latency scenarios from §2.1, now comparing LP-only RouteWise against
RouteWise with probability-target hedging. Explorer/probing is intentionally
disabled in this simulator section.

This module is experiment glue only: the hedging trigger, backup selection, and
profile learning behavior live in :mod:`rwsim.policies.routewise`.

Outputs:
- ``summary.json`` — full rows enriched with hedging metadata and LP-only deltas
- ``summary.csv`` — compact §2.2 view for hedge-rate / P99 / mean-P50 plots
- ``metadata.json``, ``ttft_histograms.json`` — same as other simulator sections
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from experiments.simulation import latency_layer
from experiments.simulation.common import (
    DEFAULT_SEEDS,
    DEFAULT_WORKLOAD,
    OUTPUT_DIR,
    WORKLOAD_CHOICES,
    WORKLOAD_COST_ENVELOPE,
    SectionCell,
    SectionCellResult,
    load_workload,
    routewise_hedging_policy_name,
    routewise_lp_policy_name,
    run_policy,
    run_section,
    write_json,
)

if TYPE_CHECKING:
    from rwsim.schemas import Request
    from rwsim.world.scenarios import ScenarioConfig

SECTION_NAME = "hedging"

PUBLIC_SCENARIO_TAG: str = "hedging"
DEFAULT_ROUTEWISE_P: float = 0.75
TARGET_SUCCESS_PROBABILITY: float = 0.99

HEAVY_TAIL_SCENARIO_NAME: str = "hedging_heavy_tail"
REAL_WORLD_SCENARIO_NAME: str = "hedging_real_world_rw3"

_SOURCE_LATENCY_SCENARIOS: dict[str, str] = {
    HEAVY_TAIL_SCENARIO_NAME: "latency_layer_heavy_tail_half_overlap",
    REAL_WORLD_SCENARIO_NAME: latency_layer.REAL_WORLD_SCENARIO_NAME,
}

# Heavy-tail synthetic uses a tighter SLO to make hedge decisions observable.
# RW3 keeps the simulator default, matching the real-world latency scale.
_SCENARIO_SLO_MS: dict[str, float] = {
    HEAVY_TAIL_SCENARIO_NAME: 500.0,
    REAL_WORLD_SCENARIO_NAME: 2000.0,
}


def list_scenarios() -> tuple[str, ...]:
    """Return §2.2 scenario names."""
    return (
        HEAVY_TAIL_SCENARIO_NAME,
        REAL_WORLD_SCENARIO_NAME,
    )


def make_scenarios() -> dict[str, ScenarioConfig]:
    """Build all §2.2 scenarios keyed by scenario name."""
    return {name: make_scenario(name) for name in list_scenarios()}


def make_scenario(name: str) -> ScenarioConfig:
    """Build one §2.2 scenario by name."""
    try:
        source_name = _SOURCE_LATENCY_SCENARIOS[name]
    except KeyError as exc:
        known = ", ".join(list_scenarios())
        raise ValueError(f"unknown hedging scenario {name!r}; known: {known}") from exc

    base = latency_layer.make_scenario(source_name)
    slo_ms = _SCENARIO_SLO_MS[name]
    base_meta = dict(base.metadata or {})
    metadata = {
        **base_meta,
        "public_scenario": PUBLIC_SCENARIO_TAG,
        "artifact_label": name,
        "hedging_scenario": name.removeprefix("hedging_"),
        "source_latency_scenario": source_name,
        "source_public_scenario": base_meta.get("public_scenario"),
        "slo_ms": slo_ms,
        "target_success_probability": TARGET_SUCCESS_PROBABILITY,
        "hedging_trigger": "probability_target",
        "backup_selection": "probability_target_non_primary",
        "latency_profile_source": "configured_distribution",
        "explorer_profile_learning": "disabled",
        "same_cost_providers": True,
    }
    description = (
        f"§2.2 hedging: source={source_name}, SLO={slo_ms:.0f} ms, "
        "same provider costs, compare LP-only against probability-target hedging."
    )
    return replace(
        base,
        name=name,
        description=description,
        primary_slo_ms=slo_ms,
        metadata=metadata,
    )


def policies_for_section(p_value: float = DEFAULT_ROUTEWISE_P) -> tuple[str, ...]:
    """Return the §2.2 policy pair: LP-only baseline and hedging-only RouteWise."""
    return (
        routewise_lp_policy_name(p_value),
        routewise_hedging_policy_name(p_value),
    )


def make_policy_presets(p_value: float = DEFAULT_ROUTEWISE_P) -> dict[str, dict[str, Any]]:
    """Build the section-local policy presets for §2.2."""
    p = float(p_value)
    lp_name = routewise_lp_policy_name(p)
    hedging_name = routewise_hedging_policy_name(p)
    return {
        lp_name: {
            "policy": "RouteWisePolicy",
            "params": {
                "hedging": False,
                "explorer": False,
                "p": p,
                "cost_envelope": WORKLOAD_COST_ENVELOPE,
                "latency_profile_mode": "configured",
            },
        },
        hedging_name: {
            "policy": "RouteWisePolicy",
            "params": {
                "hedging": "probability_target",
                "explorer": False,
                "p": p,
                "cost_envelope": WORKLOAD_COST_ENVELOPE,
                "latency_profile_mode": "configured",
            },
        },
    }


def run_hedging_policy(
    scenario: ScenarioConfig,
    requests: list[Request],
    policy_name: str,
    *,
    presets: dict[str, dict[str, Any]],
    seed: int,
    retain_records: bool = True,
):
    """Run one §2.2 policy with the scenario SLO injected into RouteWise."""
    return run_policy(
        scenario,
        requests,
        policy_name,
        presets=_presets_for_scenario_slo(presets, scenario),
        seed=seed,
        retain_records=retain_records,
    )


def run_hedging_cell(
    cell: SectionCell,
    presets: dict[str, dict[str, Any]],
    workload_dataset: str,
    duration_sec: float | None,
    max_requests: int | None,
    retain_records: bool,
) -> SectionCellResult:
    """Run one §2.2 simulation cell in a worker process."""
    scenario = make_scenario(cell.scenario_name)
    requests = load_workload(
        dataset=workload_dataset,
        duration_sec=duration_sec,
        max_requests=max_requests,
    )
    run = run_hedging_policy(
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


def _presets_for_scenario_slo(
    presets: dict[str, dict[str, Any]],
    scenario: ScenarioConfig,
) -> dict[str, dict[str, Any]]:
    """Attach the scenario primary SLO to RouteWise policy params."""
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
        return run_hedging_policy(
            scenario,
            requests,
            policy_name,
            presets=presets,
            seed=seed,
            retain_records=retain_records,
        )

    return runner


def _enrich_rows_with_hedging_metadata(
    rows: list[dict[str, Any]],
    scenarios: dict[str, ScenarioConfig],
    presets: dict[str, dict[str, Any]],
    *,
    p_value: float,
) -> list[dict[str, Any]]:
    """Fold scenario/policy metadata and LP-only deltas into summary rows."""
    enriched: list[dict[str, Any]] = []
    for row in rows:
        scenario = scenarios[row["scenario"]]
        meta = dict(scenario.metadata or {})
        preset = presets[row["policy"]]
        params = dict(preset.get("params", {}))
        merged = dict(row)
        merged.update(
            {
                "hedging_scenario": meta.get("hedging_scenario"),
                "source_latency_scenario": meta.get("source_latency_scenario"),
                "latency_family": meta.get("latency_family"),
                "overlap_label": meta.get("overlap_label"),
                "slo_ms": meta.get("slo_ms"),
                "target_success_probability": meta.get("target_success_probability"),
                "routewise_p": float(params.get("p", p_value)),
                "hedging_enabled": bool(params.get("hedging")),
                "explorer_enabled": bool(params.get("explorer")),
                "hedging_policy_mode": _hedging_policy_mode(row["policy"], params),
                "backup_selection": _backup_selection(params),
                "learns_from_backup": bool(params.get("explorer")),
                "latency_profile_mode": params.get("latency_profile_mode", "observed"),
            }
        )
        for key in (
            "target_band_coverage_fast_medium",
            "realised_band_coverage_fast_medium",
            "realised_band_coverage_medium_slow",
            "realised_band_coverage_fast_slow",
            "overlap_metric_source",
        ):
            merged[key] = meta.get(key)
        enriched.append(merged)

    baseline_policy = routewise_lp_policy_name(p_value)
    baselines = {
        row["scenario"]: row
        for row in enriched
        if row["policy"] == baseline_policy
    }
    for row in enriched:
        baseline = baselines.get(row["scenario"])
        _add_baseline_deltas(row, baseline)
    return enriched


def _hedging_policy_mode(policy_name: str, params: dict[str, Any]) -> str:
    if not params.get("hedging"):
        return "lp_only"
    if params.get("explorer"):
        return "routewise_hedging_explorer"
    return "routewise_hedging"


def _backup_selection(params: dict[str, Any]) -> str:
    if not params.get("hedging"):
        return "none"
    if params.get("explorer"):
        return "random_non_primary"
    return "probability_target_non_primary"


def _add_baseline_deltas(
    row: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> None:
    if baseline is None:
        row["p99_delta_vs_lp_only_ms"] = None
        row["p99_reduction_vs_lp_only_pct"] = None
        row["mean_ttft_delta_vs_lp_only_ms"] = None
        row["p50_delta_vs_lp_only_ms"] = None
        row["hedge_rate_delta_vs_lp_only"] = None
        row["cost_multiplier_vs_lp_only"] = None
        return

    row["p99_delta_vs_lp_only_ms"] = row["p99_ms"] - baseline["p99_ms"]
    row["p99_reduction_vs_lp_only_pct"] = _relative_reduction(
        before=baseline["p99_ms"],
        after=row["p99_ms"],
    )
    row["mean_ttft_delta_vs_lp_only_ms"] = (
        row["mean_ttft_ms"] - baseline["mean_ttft_ms"]
    )
    row["p50_delta_vs_lp_only_ms"] = row["p50_ms"] - baseline["p50_ms"]
    row["hedge_rate_delta_vs_lp_only"] = row["hedge_rate"] - baseline["hedge_rate"]
    base_cost = _row_cost_for_multiplier(baseline)
    row_cost = _row_cost_for_multiplier(row)
    row["cost_multiplier_vs_lp_only"] = (
        row_cost / base_cost if base_cost and base_cost > 0.0 else None
    )


def _relative_reduction(*, before: float, after: float) -> float | None:
    if before <= 0.0:
        return None
    return (before - after) / before


def _row_cost_for_multiplier(row: dict[str, Any]) -> float | None:
    for key in ("mean_total_cost_usd", "mean_api_cost_usd"):
        value = row.get(key)
        if value is not None:
            return float(value)
    return None


_HEDGING_CSV_FIELDNAMES: tuple[str, ...] = (
    "scenario",
    "public_scenario",
    "artifact_label",
    "hedging_scenario",
    "source_latency_scenario",
    "latency_family",
    "overlap_label",
    "policy",
    "hedging_policy_mode",
    "hedging_enabled",
    "explorer_enabled",
    "backup_selection",
    "learns_from_backup",
    "latency_profile_mode",
    "routewise_p",
    "slo_ms",
    "target_success_probability",
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
    "p99_delta_vs_lp_only_ms",
    "p99_reduction_vs_lp_only_pct",
    "mean_ttft_delta_vs_lp_only_ms",
    "p50_delta_vs_lp_only_ms",
    "hedge_rate_delta_vs_lp_only",
    "cost_multiplier_vs_lp_only",
    "target_band_coverage_fast_medium",
    "realised_band_coverage_fast_medium",
    "realised_band_coverage_medium_slow",
    "realised_band_coverage_fast_slow",
    "overlap_metric_source",
    "percentile_source",
    "histogram_bins",
)


def _write_hedging_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the §2.2-specific csv view."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_HEDGING_CSV_FIELDNAMES))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(row[key], sort_keys=True)
                        if isinstance(row.get(key), (dict, list))
                        else row.get(key)
                    )
                    for key in _HEDGING_CSV_FIELDNAMES
                }
            )


def main(argv: list[str] | None = None) -> int:
    """Run the §2.2 hedging simulator section."""
    parser = argparse.ArgumentParser(
        prog="routewise simulator hedging",
        description=__doc__,
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=list(list_scenarios()),
        help="Scenario to run. Repeat to run multiple. Defaults to all §2.2 scenarios.",
    )
    parser.add_argument(
        "--policy",
        action="append",
        help="Policy to run. Repeat to run multiple. Defaults to LP-only + hedging.",
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
        default=DEFAULT_ROUTEWISE_P,
        dest="p_value",
        help=f"Single RouteWise p value for §2.2 policies. Defaults to {DEFAULT_ROUTEWISE_P}.",
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
        default=OUTPUT_DIR / "hedging",
        help="Directory for metadata.json, summary.json, and summary.csv.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel scenario-policy-seed cells to run. Defaults to 1.",
    )

    args = parser.parse_args(argv)
    p_value = float(args.p_value)
    selected_scenarios = tuple(args.scenario) if args.scenario else list_scenarios()
    scenarios = {name: make_scenario(name) for name in selected_scenarios}
    presets = make_policy_presets(p_value)
    policies = tuple(args.policy) if args.policy else policies_for_section(p_value)
    unknown = [policy for policy in policies if policy not in presets]
    if unknown:
        known = ", ".join(sorted(presets))
        raise SystemExit(f"unknown hedging policy {unknown[0]!r}; known policies: {known}")

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
        parallel_cell_runner=run_hedging_cell,
    )
    enriched_rows = _enrich_rows_with_hedging_metadata(
        rows,
        scenarios,
        presets,
        p_value=p_value,
    )
    write_json(args.output_dir / "summary.json", enriched_rows)
    _write_hedging_summary_csv(args.output_dir / "summary.csv", enriched_rows)
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


__all__ = [
    "DEFAULT_ROUTEWISE_P",
    "HEAVY_TAIL_SCENARIO_NAME",
    "PUBLIC_SCENARIO_TAG",
    "REAL_WORLD_SCENARIO_NAME",
    "SECTION_NAME",
    "list_scenarios",
    "main",
    "make_policy_presets",
    "make_scenario",
    "make_scenarios",
    "policies_for_section",
    "run_hedging_cell",
    "run_hedging_policy",
]
