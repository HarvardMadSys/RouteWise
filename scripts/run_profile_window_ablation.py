"""One-command pipeline for the online latency-profile window ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.ablations.profile_window import harness, presets
from plots.ablations import plot_profile_window_ablation

DEFAULT_OUTPUT_ROOT = Path("outputs/ablations/profile_window")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root directory for ablation outputs. Defaults to {DEFAULT_OUTPUT_ROOT}.",
    )
    parser.add_argument(
        "--window-min",
        type=float,
        action="append",
        dest="window_minutes",
        help=(
            "Rolling-profile window length in minutes. Repeat to sweep. "
            f"Defaults to {presets.DEFAULT_WINDOW_MINUTES}."
        ),
    )
    parser.add_argument(
        "--period-min",
        type=float,
        action="append",
        dest="period_minutes",
        help=(
            "Environment change period in minutes; 0 means static. Repeat to "
            f"sweep. Defaults to {harness.DEFAULT_PERIOD_MINUTES}."
        ),
    )
    parser.add_argument(
        "--magnitude",
        type=float,
        default=harness.DEFAULT_MAGNITUDE,
        help=f"Degraded-phase TTFT scale factor. Defaults to {harness.DEFAULT_MAGNITUDE}.",
    )
    parser.add_argument(
        "--alpha",
        "--p",
        type=float,
        action="append",
        dest="alpha_values",
        help=f"RouteWise alpha value. Repeat to sweep. Defaults to {presets.DEFAULT_ALPHA_VALUES}.",
    )
    parser.add_argument(
        "--no-oracle",
        action="store_true",
        help="Skip the configured-mode oracle reference policies.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        help="Seed to run. Repeat to run multiple. Defaults to the section default.",
    )
    parser.add_argument(
        "--workload",
        default=harness.DEFAULT_WORKLOAD,
        help=f"Trace workload to replay. Defaults to {harness.DEFAULT_WORKLOAD}.",
    )
    parser.add_argument(
        "--duration-sec", type=float, help="Optional trace truncation for smoke runs."
    )
    parser.add_argument(
        "--max-requests", type=int, help="Optional request-count truncation for smoke runs."
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel scenario-policy-seed cells. Defaults to 1.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only plot from an existing summary.csv.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Run the grid without plotting.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print planned steps without running them."
    )
    args = parser.parse_args(argv)

    window_minutes = tuple(args.window_minutes or presets.DEFAULT_WINDOW_MINUTES)
    period_minutes = tuple(
        args.period_minutes if args.period_minutes is not None else harness.DEFAULT_PERIOD_MINUTES
    )
    alpha_values = tuple(args.alpha_values or presets.DEFAULT_ALPHA_VALUES)
    config: dict[str, Any] = {
        "output_root": str(args.output_root),
        "window_minutes": window_minutes,
        "period_minutes": period_minutes,
        "magnitude": args.magnitude,
        "alpha_values": alpha_values,
        "include_oracle": not args.no_oracle,
        "seeds": args.seed,
        "workload": args.workload,
        "duration_sec": args.duration_sec,
        "max_requests": args.max_requests,
        "jobs": args.jobs,
        "skip_run": args.skip_run,
        "skip_plot": args.skip_plot,
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, **config}, indent=2, sort_keys=True))
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = dict(config)
    manifest["outputs"] = {"summary_csv": str(args.output_root / "summary.csv")}

    if not args.skip_run:
        harness.main(_harness_args(args, window_minutes, period_minutes, alpha_values))

    if not args.skip_plot:
        plot_profile_window_ablation.main(
            [
                "--input-dir",
                str(args.output_root),
                "--output-dir",
                str(args.output_root),
            ]
        )
        manifest["outputs"]["figures_dir"] = str(args.output_root / "figures")
        manifest["outputs"]["delta_summary_csv"] = str(
            args.output_root / "profile_window_delta_summary.csv"
        )

    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps({"manifest": str(manifest_path), "outputs": manifest["outputs"]}, sort_keys=True)
    )
    return 0


def _harness_args(
    args: argparse.Namespace,
    window_minutes: tuple[float, ...],
    period_minutes: tuple[float, ...],
    alpha_values: tuple[float, ...],
) -> list[str]:
    argv = [
        "--magnitude",
        str(args.magnitude),
        "--workload",
        args.workload,
        "--output-dir",
        str(args.output_root),
        "--jobs",
        str(args.jobs),
    ]
    for value in window_minutes:
        argv.extend(["--window-min", str(value)])
    for value in period_minutes:
        argv.extend(["--period-min", str(value)])
    for value in alpha_values:
        argv.extend(["--alpha", str(value)])
    for seed in args.seed or ():
        argv.extend(["--seed", str(seed)])
    if args.no_oracle:
        argv.append("--no-oracle")
    if args.duration_sec is not None:
        argv.extend(["--duration-sec", str(args.duration_sec)])
    if args.max_requests is not None:
        argv.extend(["--max-requests", str(args.max_requests)])
    return argv


if __name__ == "__main__":
    raise SystemExit(main())
