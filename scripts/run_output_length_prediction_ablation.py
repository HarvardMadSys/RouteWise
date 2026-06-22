"""One-command pipeline for output-length prediction error ablations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from experiments.simulation import end_to_end
from plots.ablations import plot_output_length_prediction_ablation

DEFAULT_OUTPUT_ROOT = Path("outputs/ablations/output_length_prediction")
DEFAULT_ERROR_PCTS = (-50, -25, 0, 25, 50, 100, 200)
DEFAULT_ALPHA_VALUES = (0.0, 0.25, 0.5)
DEFAULT_BASE_PREDICTOR = "oracle"
DEFAULT_SCENARIO = end_to_end.RW8_SCENARIO_NAME


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root directory for ablation outputs. Defaults to {DEFAULT_OUTPUT_ROOT}.",
    )
    parser.add_argument(
        "--error-pct",
        type=int,
        action="append",
        dest="error_pcts",
        help=f"Signed output-token prediction error percent. Repeat to sweep. Defaults to {DEFAULT_ERROR_PCTS}.",
    )
    parser.add_argument(
        "--base-predictor",
        default=DEFAULT_BASE_PREDICTOR,
        help=(
            "Predictor to scale for the ablation. Defaults to oracle, so the sweep "
            "isolates systematic output-length bias."
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=list(end_to_end.list_scenarios()),
        default=DEFAULT_SCENARIO,
        help=f"Single end-to-end scenario to run. Defaults to {DEFAULT_SCENARIO}.",
    )
    parser.add_argument(
        "--policy",
        action="append",
        help="Policy to run. Repeat to customize. Defaults to LP and LP+hedging for each alpha.",
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
        "--seed",
        type=int,
        action="append",
        help=f"Seed to run. Repeat to run multiple. Defaults to {end_to_end.DEFAULT_SEEDS}.",
    )
    parser.add_argument(
        "--workload",
        default=end_to_end.DEFAULT_WORKLOAD,
        choices=end_to_end.WORKLOAD_CHOICES,
        help=f"Trace workload to replay. Defaults to {end_to_end.DEFAULT_WORKLOAD}.",
    )
    parser.add_argument(
        "--duration-sec", type=float, help="Optional trace truncation for smoke runs."
    )
    parser.add_argument(
        "--max-requests", type=int, help="Optional request-count truncation for smoke runs."
    )
    parser.add_argument(
        "--quota-plan",
        default=end_to_end.DEFAULT_QUOTA_PLAN,
        help=f"Quota subscription plan id. Defaults to {end_to_end.DEFAULT_QUOTA_PLAN}.",
    )
    parser.add_argument(
        "--quota-count",
        type=int,
        default=end_to_end.DEFAULT_QUOTA_COUNT,
        help=f"Quota subscription count. Defaults to {end_to_end.DEFAULT_QUOTA_COUNT}.",
    )
    parser.add_argument(
        "--concurrency-plan",
        default=end_to_end.DEFAULT_CONCURRENCY_PLAN,
        help=f"Concurrency subscription plan id. Defaults to {end_to_end.DEFAULT_CONCURRENCY_PLAN}.",
    )
    parser.add_argument(
        "--concurrency-count",
        type=int,
        default=end_to_end.DEFAULT_CONCURRENCY_COUNT,
        help=f"Concurrency subscription count. Defaults to {end_to_end.DEFAULT_CONCURRENCY_COUNT}.",
    )
    parser.add_argument(
        "--model",
        default=end_to_end.DEFAULT_CONCURRENCY_MODEL,
        help=f"Concurrency model id or trace alias. Defaults to {end_to_end.DEFAULT_CONCURRENCY_MODEL}.",
    )
    parser.add_argument(
        "--prefix-cache-enabled",
        action="store_true",
        help="Enable provider-local prefix-cache discounts.",
    )
    parser.add_argument(
        "--cached-input-price-fraction",
        type=float,
        default=end_to_end.DEFAULT_CACHED_INPUT_PRICE_FRACTION,
        help=(
            "Cached-input price as a fraction of each API provider's normal input "
            f"price. Defaults to {end_to_end.DEFAULT_CACHED_INPUT_PRICE_FRACTION}."
        ),
    )
    parser.add_argument(
        "--slo-ms",
        type=float,
        default=end_to_end.DEFAULT_SLO_MS,
        help=f"Single-request SLO threshold in ms. Defaults to {end_to_end.DEFAULT_SLO_MS}.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel simulation cells per error level. Defaults to 1.",
    )
    parser.add_argument(
        "--skip-run", action="store_true", help="Only merge and plot existing err_* outputs."
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Run ablations and merge summaries without plotting.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print planned steps without running them."
    )
    args = parser.parse_args(argv)

    error_pcts = tuple(args.error_pcts or DEFAULT_ERROR_PCTS)
    alpha_values = tuple(args.alpha_values or DEFAULT_ALPHA_VALUES)
    policies = tuple(args.policy or _default_policies(alpha_values))
    config = {
        "output_root": str(args.output_root),
        "error_pcts": error_pcts,
        "base_predictor": args.base_predictor,
        "scenario": args.scenario,
        "policies": policies,
        "alpha_values": alpha_values,
        "seeds": tuple(args.seed or end_to_end.DEFAULT_SEEDS),
        "workload": args.workload,
        "quota_plan": args.quota_plan,
        "quota_count": args.quota_count,
        "concurrency_plan": args.concurrency_plan,
        "concurrency_count": args.concurrency_count,
        "model": args.model,
        "prefix_cache_enabled": args.prefix_cache_enabled,
        "cached_input_price_fraction": args.cached_input_price_fraction,
        "slo_ms": args.slo_ms,
        "jobs": args.jobs,
        "duration_sec": args.duration_sec,
        "max_requests": args.max_requests,
        "skip_run": args.skip_run,
        "skip_plot": args.skip_plot,
    }
    _validate_error_pcts(error_pcts)

    planned_runs = {
        str(error): {
            "dir": str(args.output_root / _error_dir_name(error)),
            "predictor": _scaled_predictor_name(args.base_predictor, error),
        }
        for error in error_pcts
    }
    if args.dry_run:
        print(
            json.dumps({"dry_run": True, **config, "runs": planned_runs}, indent=2, sort_keys=True)
        )
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = dict(config)
    manifest["runs"] = planned_runs
    manifest["outputs"] = {}

    if not args.skip_run:
        for error in error_pcts:
            run_dir = args.output_root / _error_dir_name(error)
            end_to_end.main(_run_args(args, run_dir, error, policies, alpha_values))

    merged_rows = _merge_summaries(args.output_root, error_pcts)
    summary_path = args.output_root / "summary.csv"
    _write_csv(summary_path, merged_rows)
    manifest["outputs"]["summary_csv"] = str(summary_path)

    if not args.skip_plot:
        plot_output_length_prediction_ablation.main(
            [
                "--input-dir",
                str(args.output_root),
                "--output-dir",
                str(args.output_root),
                "--scenario",
                args.scenario,
            ]
        )
        manifest["outputs"]["figures_dir"] = str(args.output_root / "figures")
        manifest["outputs"]["delta_summary_csv"] = str(
            args.output_root / "output_length_prediction_delta_summary.csv"
        )
        manifest["outputs"]["worst_case_summary_csv"] = str(
            args.output_root / "output_length_prediction_worst_case_summary.csv"
        )

    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps({"manifest": str(manifest_path), "outputs": manifest["outputs"]}, sort_keys=True)
    )
    return 0


def _default_policies(alpha_values: tuple[float, ...]) -> tuple[str, ...]:
    return (
        *(end_to_end.routewise_lp_policy_name(value) for value in alpha_values),
        *(end_to_end.routewise_hedging_policy_name(value) for value in alpha_values),
    )


def _run_args(
    args: argparse.Namespace,
    output_dir: Path,
    error_pct: int,
    policies: tuple[str, ...],
    alpha_values: tuple[float, ...],
) -> list[str]:
    argv = [
        "--scenario",
        args.scenario,
        "--workload",
        args.workload,
        "--predictor",
        _scaled_predictor_name(args.base_predictor, error_pct),
        "--output-dir",
        str(output_dir),
        "--quota-plan",
        args.quota_plan,
        "--quota-count",
        str(args.quota_count),
        "--concurrency-plan",
        args.concurrency_plan,
        "--concurrency-count",
        str(args.concurrency_count),
        "--model",
        args.model,
        "--cached-input-price-fraction",
        str(args.cached_input_price_fraction),
        "--slo-ms",
        str(args.slo_ms),
        "--jobs",
        str(args.jobs),
    ]
    _append_repeated(argv, "--policy", policies)
    _append_repeated(argv, "--alpha", alpha_values)
    _append_repeated(argv, "--seed", tuple(args.seed or end_to_end.DEFAULT_SEEDS))
    if args.duration_sec is not None:
        argv.extend(["--duration-sec", str(args.duration_sec)])
    if args.max_requests is not None:
        argv.extend(["--max-requests", str(args.max_requests)])
    if args.prefix_cache_enabled:
        argv.append("--prefix-cache-enabled")
    return argv


def _scaled_predictor_name(base_predictor: str, error_pct: int) -> str:
    multiplier = 1.0 + float(error_pct) / 100.0
    return f"scaled:{base_predictor}:{multiplier:.12g}"


def _validate_error_pcts(error_pcts: tuple[int, ...]) -> None:
    if not error_pcts:
        raise ValueError("at least one --error-pct value is required")
    if 0 not in error_pcts:
        raise ValueError("include --error-pct 0 so delta plots have a baseline")
    too_low = [value for value in error_pcts if value < -100]
    if too_low:
        raise ValueError(f"prediction error cannot be below -100%, got {too_low}")


def _merge_summaries(output_root: Path, error_pcts: tuple[int, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for error in error_pcts:
        summary_path = output_root / _error_dir_name(error) / "summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        with summary_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                merged = dict(row)
                merged["prediction_error_pct"] = error
                merged["prediction_multiplier"] = 1.0 + float(error) / 100.0
                rows.append(merged)
    if not rows:
        raise ValueError(f"no summary rows found under {output_root}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_repeated(argv: list[str], flag: str, values: tuple[Any, ...]) -> None:
    for value in values:
        argv.extend([flag, str(value)])


def _error_dir_name(error_pct: int) -> str:
    sign = "-" if error_pct < 0 else ""
    return f"err_{sign}{abs(error_pct):03d}"


if __name__ == "__main__":
    raise SystemExit(main())
