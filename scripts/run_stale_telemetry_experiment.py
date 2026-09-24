"""One-command pipeline for the stale-telemetry robustness experiment (run + plot)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.ablations.stale_telemetry import harness, presets
from plots.ablations import plot_stale_telemetry

DEFAULT_OUTPUT_ROOT = Path("outputs/ablations/stale_telemetry")


def main(argv: list[str] | None = None) -> int:
    """Run the spike grid, plot it, and write a manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root directory for experiment outputs. Defaults to {DEFAULT_OUTPUT_ROOT}.",
    )
    parser.add_argument(
        "--magnitude",
        type=float,
        action="append",
        dest="magnitudes",
        help=(
            f"Spike TTFT scale factor. Repeat to sweep. Defaults to {harness.DEFAULT_MAGNITUDES}."
        ),
    )
    parser.add_argument(
        "--spike-duration-min",
        type=float,
        action="append",
        dest="duration_minutes",
        help=(
            "Spike duration in minutes. Repeat to sweep. "
            f"Defaults to {harness.DEFAULT_SPIKE_DURATION_MINUTES}."
        ),
    )
    parser.add_argument(
        "--period-min",
        type=float,
        default=harness.DEFAULT_PERIOD_MIN,
        help=f"Spike repetition period in minutes. Defaults to {harness.DEFAULT_PERIOD_MIN:g}.",
    )
    parser.add_argument(
        "--warmup-min",
        type=float,
        default=harness.DEFAULT_WARMUP_MIN,
        help=(
            "Stationary minutes before the first spike. "
            f"Defaults to {harness.DEFAULT_WARMUP_MIN:g}."
        ),
    )
    parser.add_argument(
        "--pre-onset-min",
        type=float,
        default=harness.DEFAULT_PRE_ONSET_MIN,
        help=(
            "Minutes before onset covered by the time series. "
            f"Defaults to {harness.DEFAULT_PRE_ONSET_MIN:g}."
        ),
    )
    parser.add_argument(
        "--pool",
        default=harness.DEFAULT_POOL,
        choices=list(harness.POOL_SCENARIOS),
        help=f"§2.2 same-cost real-world provider pool. Defaults to {harness.DEFAULT_POOL}.",
    )
    parser.add_argument(
        "--spiked-provider",
        help="Provider to spike. Defaults to the lowest baseline mean TTFT in the pool.",
    )
    parser.add_argument(
        "--window-min",
        type=float,
        default=presets.DEFAULT_PROFILE_WINDOW_MIN,
        help=(
            "Rolling-profile window (minutes) for the observed regime. "
            f"Defaults to {presets.DEFAULT_PROFILE_WINDOW_MIN:g}."
        ),
    )
    parser.add_argument(
        "--policy",
        action="append",
        dest="policies",
        help="Policy to run. Repeat to run multiple. Defaults to all six presets.",
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
        "--duration-sec", type=float, help="Optional trace truncation (seconds of trace)."
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
        help="Only plot from existing summary.csv / spike_timeseries.csv.",
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

    magnitudes = tuple(args.magnitudes or harness.DEFAULT_MAGNITUDES)
    duration_minutes = tuple(args.duration_minutes or harness.DEFAULT_SPIKE_DURATION_MINUTES)
    config: dict[str, Any] = {
        "output_root": str(args.output_root),
        "magnitudes": magnitudes,
        "spike_duration_minutes": duration_minutes,
        "period_min": args.period_min,
        "warmup_min": args.warmup_min,
        "pre_onset_min": args.pre_onset_min,
        "pool": args.pool,
        "spiked_provider": args.spiked_provider,
        "window_min": args.window_min,
        "policies": args.policies,
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
    manifest["outputs"] = {
        "summary_csv": str(args.output_root / "summary.csv"),
        "spike_timeseries_csv": str(args.output_root / "spike_timeseries.csv"),
    }

    if not args.skip_run:
        harness.main(_harness_args(args, magnitudes, duration_minutes))

    if not args.skip_plot:
        plot_stale_telemetry.main(
            [
                "--input-dir",
                str(args.output_root),
                "--output-dir",
                str(args.output_root),
            ]
        )
        manifest["outputs"]["figures_dir"] = str(args.output_root / "figures")
        manifest["outputs"]["spike_summary_csv"] = str(
            args.output_root / "stale_telemetry_spike_summary.csv"
        )

    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps({"manifest": str(manifest_path), "outputs": manifest["outputs"]}, sort_keys=True)
    )
    return 0


def _harness_args(
    args: argparse.Namespace,
    magnitudes: tuple[float, ...],
    duration_minutes: tuple[float, ...],
) -> list[str]:
    argv = [
        "--period-min",
        str(args.period_min),
        "--warmup-min",
        str(args.warmup_min),
        "--pre-onset-min",
        str(args.pre_onset_min),
        "--pool",
        args.pool,
        "--window-min",
        str(args.window_min),
        "--workload",
        args.workload,
        "--output-dir",
        str(args.output_root),
        "--jobs",
        str(args.jobs),
    ]
    for value in magnitudes:
        argv.extend(["--magnitude", str(value)])
    for value in duration_minutes:
        argv.extend(["--spike-duration-min", str(value)])
    for policy in args.policies or ():
        argv.extend(["--policy", policy])
    for seed in args.seed or ():
        argv.extend(["--seed", str(seed)])
    if args.spiked_provider is not None:
        argv.extend(["--spiked-provider", args.spiked_provider])
    if args.duration_sec is not None:
        argv.extend(["--duration-sec", str(args.duration_sec)])
    if args.max_requests is not None:
        argv.extend(["--max-requests", str(args.max_requests)])
    return argv


if __name__ == "__main__":
    raise SystemExit(main())
