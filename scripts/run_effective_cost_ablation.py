"""One-command pipeline for effective-cost ablation runs and plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.ablations.effective_cost import harness
from experiments.ablations.effective_cost.presets import (
    DEFAULT_CONCURRENCY_CURVES,
    DEFAULT_QUOTA_CURVES,
)
from plots.ablations.effective_cost import plot_concurrency_lines, plot_quota_lines

DEFAULT_OUTPUT_ROOT = Path("outputs/ablations/effective_cost")
DEFAULT_QUOTA_LIMITS = (10_000, 20_000, 40_000, 60_000, 80_000)
DEFAULT_ALPHA_VALUES = (0.0,)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("all", harness.PHASE_QUOTA, harness.PHASE_CONCURRENCY),
        default="all",
        help="Ablation phase to run. Defaults to all.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root directory for phase outputs. Defaults to {DEFAULT_OUTPUT_ROOT}.",
    )
    parser.add_argument(
        "--workload",
        default=harness.DEFAULT_WORKLOAD,
        choices=("burstgpt", "sharegpt_burstgpt"),
        help=f"Trace workload to replay. Defaults to {harness.DEFAULT_WORKLOAD}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        help="Seed to run. Repeat to run multiple. Defaults to 42.",
    )
    parser.add_argument(
        "--p",
        "--alpha",
        type=float,
        action="append",
        dest="alpha_values",
        help=f"LP budget alpha/p value. Repeat to sweep. Defaults to {DEFAULT_ALPHA_VALUES}.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel scenario-policy-seed cells. Defaults to 1.",
    )
    parser.add_argument("--duration-sec", type=float, help="Optional trace truncation for smoke runs.")
    parser.add_argument("--max-requests", type=int, help="Optional request-count truncation for smoke runs.")
    parser.add_argument(
        "--quota-limit",
        type=int,
        action="append",
        dest="quota_limits",
        help=f"Quota request limit for Phase A. Repeat to sweep. Defaults to {DEFAULT_QUOTA_LIMITS}.",
    )
    parser.add_argument(
        "--quota-curve",
        action="append",
        choices=DEFAULT_QUOTA_CURVES,
        dest="quota_curves",
        help=f"Quota curve for Phase A. Repeat to sweep. Defaults to {DEFAULT_QUOTA_CURVES}.",
    )
    parser.add_argument(
        "--concurrency-count",
        type=int,
        action="append",
        dest="concurrency_counts",
        help=(
            "Concurrency account count for Phase B. Repeat to sweep. "
            f"Defaults to {harness.DEFAULT_CONCURRENCY_COUNTS}."
        ),
    )
    parser.add_argument(
        "--concurrency-curve",
        action="append",
        choices=DEFAULT_CONCURRENCY_CURVES,
        dest="concurrency_curves",
        help=f"Concurrency curve for Phase B. Repeat to sweep. Defaults to {DEFAULT_CONCURRENCY_CURVES}.",
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=plot_concurrency_lines.DEFAULT_BASELINE_SUMMARY,
        help="Baseline summary.csv used by concurrency plots.",
    )
    parser.add_argument("--skip-run", action="store_true", help="Only run plotting/manifest steps.")
    parser.add_argument("--skip-plot", action="store_true", help="Run ablations without plotting.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned steps without running them.")
    args = parser.parse_args(argv)

    phases = _selected_phases(args.phase)
    alpha_values = tuple(args.alpha_values or DEFAULT_ALPHA_VALUES)
    if not args.skip_plot and len(alpha_values) != 1:
        raise ValueError("plotting expects exactly one --p/--alpha value; use --skip-plot for multi-p sweeps")

    config = {
        "phase": args.phase,
        "phases": phases,
        "output_root": str(args.output_root),
        "workload": args.workload,
        "seeds": tuple(args.seed or (42,)),
        "alpha_values": alpha_values,
        "quota_limits": tuple(args.quota_limits or DEFAULT_QUOTA_LIMITS),
        "quota_curves": tuple(args.quota_curves or DEFAULT_QUOTA_CURVES),
        "concurrency_counts": tuple(args.concurrency_counts or harness.DEFAULT_CONCURRENCY_COUNTS),
        "concurrency_curves": tuple(args.concurrency_curves or DEFAULT_CONCURRENCY_CURVES),
        "jobs": args.jobs,
        "duration_sec": args.duration_sec,
        "max_requests": args.max_requests,
        "skip_run": args.skip_run,
        "skip_plot": args.skip_plot,
        "baseline_summary": str(args.baseline_summary),
    }

    if args.dry_run:
        print(json.dumps({"dry_run": True, **config}, indent=2, sort_keys=True))
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = dict(config)
    manifest["outputs"] = {}

    if harness.PHASE_QUOTA in phases:
        quota_dir = args.output_root / "quota"
        if not args.skip_run:
            harness.main(_quota_run_args(args, quota_dir, config))
        if not args.skip_plot:
            plot_quota_lines.main(_quota_plot_args(args, quota_dir, config))
        manifest["outputs"]["quota"] = _phase_outputs(quota_dir)

    if harness.PHASE_CONCURRENCY in phases:
        concurrency_dir = args.output_root / "concurrency"
        if not args.skip_run:
            harness.main(_concurrency_run_args(args, concurrency_dir, config))
        if not args.skip_plot:
            plot_concurrency_lines.main(_concurrency_plot_args(args, concurrency_dir, config))
        manifest["outputs"]["concurrency"] = _phase_outputs(concurrency_dir)

    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "outputs": manifest["outputs"]}, sort_keys=True))
    return 0


def _selected_phases(phase: str) -> tuple[str, ...]:
    if phase == "all":
        return (harness.PHASE_QUOTA, harness.PHASE_CONCURRENCY)
    return (phase,)


def _quota_run_args(args: argparse.Namespace, output_dir: Path, config: dict[str, Any]) -> list[str]:
    argv = ["--phase", harness.PHASE_QUOTA, "--workload", args.workload, "--output-dir", str(output_dir)]
    _append_repeated(argv, "--curve", config["quota_curves"])
    _append_repeated(argv, "--quota-limit", config["quota_limits"])
    _append_common_run_args(argv, args, config)
    return argv


def _concurrency_run_args(args: argparse.Namespace, output_dir: Path, config: dict[str, Any]) -> list[str]:
    argv = ["--phase", harness.PHASE_CONCURRENCY, "--workload", args.workload, "--output-dir", str(output_dir)]
    _append_repeated(argv, "--concurrency-count", config["concurrency_counts"])
    _append_repeated(argv, "--concurrency-curve", config["concurrency_curves"])
    _append_common_run_args(argv, args, config)
    return argv


def _append_common_run_args(argv: list[str], args: argparse.Namespace, config: dict[str, Any]) -> None:
    _append_repeated(argv, "--p", config["alpha_values"])
    _append_repeated(argv, "--seed", config["seeds"])
    argv.extend(["--jobs", str(args.jobs)])
    if args.duration_sec is not None:
        argv.extend(["--duration-sec", str(args.duration_sec)])
    if args.max_requests is not None:
        argv.extend(["--max-requests", str(args.max_requests)])


def _quota_plot_args(args: argparse.Namespace, output_dir: Path, config: dict[str, Any]) -> list[str]:
    argv = ["--input-dir", str(output_dir), "--output-dir", str(output_dir), "--workload", args.workload]
    _append_repeated(argv, "--quota-limit", config["quota_limits"])
    _append_repeated(argv, "--curve", config["quota_curves"])
    argv.extend(["--p", str(config["alpha_values"][0])])
    return argv


def _concurrency_plot_args(args: argparse.Namespace, output_dir: Path, config: dict[str, Any]) -> list[str]:
    argv = [
        "--input-dir",
        str(output_dir),
        "--output-dir",
        str(output_dir),
        "--baseline-summary",
        str(args.baseline_summary),
    ]
    _append_repeated(argv, "--n", config["concurrency_counts"])
    _append_repeated(argv, "--curve", config["concurrency_curves"])
    argv.extend(["--p", str(config["alpha_values"][0])])
    return argv


def _append_repeated(argv: list[str], flag: str, values: tuple[Any, ...]) -> None:
    for value in values:
        argv.extend([flag, str(value)])


def _phase_outputs(output_dir: Path) -> dict[str, str]:
    return {
        "dir": str(output_dir),
        "summary_csv": str(output_dir / "summary.csv"),
        "summary_json": str(output_dir / "summary.json"),
        "metadata_json": str(output_dir / "metadata.json"),
        "figures_dir": str(output_dir / "figures"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
