"""Parallel Stage 3 quota/concurrency sensitivity runner.

This suite runs only the joint-provider Stage 3 cell while sweeping the
S_Q quota and S_C concurrency settings. It is intentionally separate from
``run_eval_grid_parallel`` because the paper grid keeps Stage 3 at one
canonical capacity point, while this runner answers the robustness question:
"does the Stage 3 conclusion hold under different subscription capacities?"

Default sweep:

    quota_size in {1000, 5000, 10000}
    concurrency_limit in {2, 4, 8}
    latency_family = heavy_tail

Usage:

    python -m experiments.simulation.suites.run_stage3_capacity_sweep \\
        --jobs 12 \\
        --seed 42 --seed 43 --seed 44 \\
        --output-root outputs/stage3_capacity_sweep
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


DEFAULT_OUTPUT_ROOT = _ROOT / "outputs" / "stage3_capacity_sweep"
DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_QUOTA_SIZES = [1000, 5000, 10000]
DEFAULT_CONCURRENCY_LIMITS = [2, 4, 8]
DEFAULT_LATENCY_FAMILIES = ["heavy_tail"]


@dataclass(frozen=True)
class CapacityJobSpec:
    """One unit of work: one dataset x one Stage 3 capacity scenario."""

    dataset_name: str
    latency_family: str
    quota_size: int
    concurrency_limit: int
    variants: tuple[str, ...]
    seeds: tuple[int, ...]
    output_dir: str
    backup_scope: str


@dataclass
class CapacityJobResult:
    """Result from one completed capacity-sweep job."""

    dataset_name: str
    scenario_name: str
    latency_family: str
    quota_size: int
    concurrency_limit: int
    summary_rows: list[dict[str, object]]
    diagnostic_rows: list[dict[str, object]]
    delta_rows: list[dict[str, object]]
    elapsed_seconds: float
    n_requests: int
    error: str | None = None


def _annotate_capacity(
    rows: list[dict[str, object]],
    *,
    latency_family: str,
    quota_size: int,
    concurrency_limit: int,
) -> list[dict[str, object]]:
    """Attach capacity-sweep coordinates to every merged CSV row."""
    for row in rows:
        row["latency_family"] = latency_family
        row["quota_size"] = quota_size
        row["concurrency_limit"] = concurrency_limit
    return rows


def _run_job(spec: CapacityJobSpec) -> CapacityJobResult:
    """Execute one capacity-sweep job in a child process."""
    t0 = time.monotonic()
    try:
        from experiments.simulation.eval_grid import (
            LatencyFamily,
            ProviderSetup,
            make_scenario,
        )
        from experiments.simulation.lp_budget_eval import (
            build_hedge_delta,
            generate_scenario_workload,
            run_variant,
            summarize_diagnostics,
            summarize_main_metrics,
        )
        from experiments.simulation.suites.run_eval_grid_parallel import (
            _build_delta_rows,
            _build_diagnostic_rows,
            _build_summary_rows,
            _write_csv,
        )

        family = LatencyFamily(spec.latency_family)
        scenario = make_scenario(
            ProviderSetup.JOINT_PROVIDER,
            family,
            stage3_quota_size=spec.quota_size,
            stage3_concurrency_limit=spec.concurrency_limit,
        )

        # Trace-driven workloads use the full natural trace, so the seed is
        # currently unused. Keep a stable seed for compatibility with the
        # synthetic path and to avoid accidental per-capacity resampling if
        # this runner is extended later.
        wl_seed = sum(
            (i + 1) * ord(c)
            for i, c in enumerate(f"{spec.dataset_name}:stage3_capacity")
        ) if spec.dataset_name != "synthetic" else 0

        workload = generate_scenario_workload(
            scenario,
            seed=wl_seed,
            dataset_name=spec.dataset_name,
        )
        n_requests = len(workload)

        out_dir = Path(spec.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        summary: dict[str, dict[str, object]] = {}
        diagnostics: dict[str, dict[str, object]] = {}

        for variant in spec.variants:
            evaluated_runs = [
                run_variant(
                    scenario,
                    workload,
                    variant,
                    seed=seed,
                    backup_scope=spec.backup_scope,
                )
                for seed in spec.seeds
            ]
            summary[variant] = summarize_main_metrics(scenario, evaluated_runs)
            diagnostics[variant] = summarize_diagnostics(evaluated_runs)

        with (out_dir / "summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2)
        with (out_dir / "diagnostics.json").open("w") as handle:
            json.dump(diagnostics, handle, indent=2)

        summary_rows = _annotate_capacity(
            _build_summary_rows(spec.dataset_name, scenario.name, summary),
            latency_family=spec.latency_family,
            quota_size=spec.quota_size,
            concurrency_limit=spec.concurrency_limit,
        )
        diagnostic_rows = _annotate_capacity(
            _build_diagnostic_rows(spec.dataset_name, scenario.name, diagnostics),
            latency_family=spec.latency_family,
            quota_size=spec.quota_size,
            concurrency_limit=spec.concurrency_limit,
        )
        delta_rows = _annotate_capacity(
            _build_delta_rows(
                spec.dataset_name,
                scenario.name,
                summary,
                build_hedge_delta,
            ),
            latency_family=spec.latency_family,
            quota_size=spec.quota_size,
            concurrency_limit=spec.concurrency_limit,
        )

        with (out_dir / "hedge_deltas.json").open("w") as handle:
            json.dump(delta_rows, handle, indent=2)

        _write_csv(out_dir / "summary.csv", summary_rows)
        _write_csv(out_dir / "diagnostics.csv", diagnostic_rows)
        _write_csv(out_dir / "hedge_deltas.csv", delta_rows)

        elapsed = time.monotonic() - t0
        return CapacityJobResult(
            dataset_name=spec.dataset_name,
            scenario_name=scenario.name,
            latency_family=spec.latency_family,
            quota_size=spec.quota_size,
            concurrency_limit=spec.concurrency_limit,
            summary_rows=summary_rows,
            diagnostic_rows=diagnostic_rows,
            delta_rows=delta_rows,
            elapsed_seconds=elapsed,
            n_requests=n_requests,
        )

    except Exception:
        elapsed = time.monotonic() - t0
        scenario_name = (
            f"grid_joint_provider_{spec.latency_family}"
            f"_q{spec.quota_size}_c{spec.concurrency_limit}"
        )
        return CapacityJobResult(
            dataset_name=spec.dataset_name,
            scenario_name=scenario_name,
            latency_family=spec.latency_family,
            quota_size=spec.quota_size,
            concurrency_limit=spec.concurrency_limit,
            summary_rows=[],
            diagnostic_rows=[],
            delta_rows=[],
            elapsed_seconds=elapsed,
            n_requests=0,
            error=traceback.format_exc(),
        )


def _write_metadata(
    output_root: Path,
    *,
    datasets: list[str],
    latency_families: list[str],
    quota_sizes: list[int],
    concurrency_limits: list[int],
    variants: list[str],
    seeds: list[int],
    jobs: int,
    backup_scope: str,
) -> None:
    meta = {
        "runner": "run_stage3_capacity_sweep",
        "scenario_family": "stage3_capacity_sweep",
        "stage": "joint_provider",
        "datasets": datasets,
        "latency_families": latency_families,
        "quota_sizes": quota_sizes,
        "concurrency_limits": concurrency_limits,
        "variants": variants,
        "seeds": seeds,
        "parallel_jobs": jobs,
        "backup_scope": backup_scope,
        "trace_timing_mode": "natural_full_trace",
        "lp_tbar_source": "ground-truth analytical mean",
        "active_probing": "disabled",
    }
    with (output_root / "metadata.json").open("w") as handle:
        json.dump(meta, handle, indent=2)


def _parse_args() -> argparse.Namespace:
    default_jobs = min(12, max(1, os.cpu_count() - 2)) if os.cpu_count() else 4
    parser = argparse.ArgumentParser(
        description=(
            "Parallel Stage 3 capacity sweep over S_Q quota and S_C concurrency."
        ),
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=default_jobs,
        help=(
            "Number of parallel worker processes. Default is conservative "
            f"({default_jobs}) so shared servers are not saturated."
        ),
    )
    parser.add_argument(
        "--seed",
        action="append",
        dest="seeds",
        type=int,
        default=[],
        help="RNG seed. May be repeated. Default: 42 43 44.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        default=[],
        help=(
            "Workload dataset. May be repeated. Default: all 3 paper workloads "
            "(sharegpt / freeinference / rednote)."
        ),
    )
    parser.add_argument(
        "--latency-family",
        action="append",
        dest="latency_families",
        choices=["uniform", "normal", "heavy_tail"],
        default=[],
        help="Latency family. May be repeated. Default: heavy_tail.",
    )
    parser.add_argument(
        "--quota-size",
        action="append",
        dest="quota_sizes",
        type=int,
        default=[],
        help="S_Q quota size. May be repeated. Default: 1000 5000 10000.",
    )
    parser.add_argument(
        "--concurrency-limit",
        action="append",
        dest="concurrency_limits",
        type=int,
        default=[],
        help="S_C concurrency limit. May be repeated. Default: 2 4 8.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--backup-scope",
        choices=["any_provider", "cross_tier"],
        default="any_provider",
        help="Backup candidate scope for Hedge-ProbTarget.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the job plan and exit without running.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    from experiments.simulation.eval_grid import (
        PAPER_GRID_VARIANTS,
        PAPER_WORKLOADS,
        WORKLOAD_DATASET_IDS,
    )

    datasets = args.datasets or [WORKLOAD_DATASET_IDS[w] for w in PAPER_WORKLOADS]
    latency_families = args.latency_families or DEFAULT_LATENCY_FAMILIES
    quota_sizes = args.quota_sizes or DEFAULT_QUOTA_SIZES
    concurrency_limits = args.concurrency_limits or DEFAULT_CONCURRENCY_LIMITS
    seeds = tuple(args.seeds or DEFAULT_SEEDS)
    variants = tuple(PAPER_GRID_VARIANTS)

    if any(value <= 0 for value in quota_sizes):
        raise SystemExit(f"quota sizes must be positive: {quota_sizes}")
    if any(value <= 0 for value in concurrency_limits):
        raise SystemExit(
            f"concurrency limits must be positive: {concurrency_limits}"
        )

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    jobs: list[CapacityJobSpec] = []
    for dataset_name in datasets:
        for family in latency_families:
            for quota_size in quota_sizes:
                for concurrency_limit in concurrency_limits:
                    scenario_name = (
                        f"grid_joint_provider_{family}_q{quota_size}_c"
                        f"{concurrency_limit}"
                    )
                    out_dir = output_root / dataset_name / scenario_name
                    jobs.append(
                        CapacityJobSpec(
                            dataset_name=dataset_name,
                            latency_family=family,
                            quota_size=quota_size,
                            concurrency_limit=concurrency_limit,
                            variants=variants,
                            seeds=seeds,
                            output_dir=str(out_dir),
                            backup_scope=args.backup_scope,
                        )
                    )

    n_jobs = len(jobs)
    n_workers = min(args.jobs, n_jobs)

    print("Stage 3 capacity sweep runner")
    print(f"  datasets:      {datasets}")
    print(f"  family:        {latency_families}")
    print(f"  S_Q quota:     {quota_sizes}")
    print(f"  S_C slots:     {concurrency_limits}")
    print(f"  variants:      {len(variants)} ({', '.join(variants[:5])}...)")
    print(f"  seeds:         {list(seeds)}")
    print(f"  total jobs:    {n_jobs} (dataset x family x quota x concurrency)")
    print(f"  workers:       {n_workers}")
    print(f"  output:        {output_root}")
    print()

    if args.dry_run:
        for i, job in enumerate(jobs, 1):
            print(
                f"  [{i:3d}] {job.dataset_name} / {job.latency_family} / "
                f"q={job.quota_size} / c={job.concurrency_limit}"
            )
        print(f"\n  (dry run - {n_jobs} jobs would be dispatched)")
        return

    trace_datasets = [d for d in datasets if d != "synthetic"]
    if trace_datasets:
        from experiments.simulation.dataset_cache import ensure_caches

        print("Warming dataset caches...")
        ensure_caches(trace_datasets)
        print()

    _write_metadata(
        output_root,
        datasets=datasets,
        latency_families=latency_families,
        quota_sizes=quota_sizes,
        concurrency_limits=concurrency_limits,
        variants=list(variants),
        seeds=list(seeds),
        jobs=n_workers,
        backup_scope=args.backup_scope,
    )

    t_start = time.monotonic()
    results: list[CapacityJobResult] = []
    completed = 0
    failed = 0

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        future_to_job = {pool.submit(_run_job, job): job for job in jobs}
        for future in as_completed(future_to_job):
            result = future.result()
            results.append(result)
            completed += 1

            capacity_label = f"q={result.quota_size},c={result.concurrency_limit}"
            if result.error:
                failed += 1
                print(
                    f"  [{completed}/{n_jobs}] FAIL  "
                    f"{result.dataset_name}/{result.latency_family}/"
                    f"{capacity_label} ({result.elapsed_seconds:.1f}s)"
                )
                print(result.error)
            else:
                print(
                    f"  [{completed}/{n_jobs}] done  "
                    f"{result.dataset_name}/{result.latency_family}/"
                    f"{capacity_label}  {result.n_requests:>7,} reqs  "
                    f"{result.elapsed_seconds:.1f}s"
                )

    from experiments.simulation.suites.run_eval_grid_parallel import (
        _merge_results,
    )

    _merge_results(results, output_root)

    elapsed_total = time.monotonic() - t_start
    print(
        f"\nAll done. {completed - failed}/{n_jobs} succeeded, "
        f"{failed} failed. Total {elapsed_total:.0f}s"
    )
    print(f"Results: {output_root}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
