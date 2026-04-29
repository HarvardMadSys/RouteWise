"""Parallel eval-grid runner.

Runs the 4-stage × 3-distribution × N-workload evaluation grid with
Python-level parallelism.  Each **job** is one ``(dataset, scenario)``
pair; within a job, all variants and seeds run sequentially (they share
a loaded workload).  Jobs are embarrassingly parallel because they touch
independent scenario configs and independent output directories.

Usage
-----
Run the full paper grid (12 scenarios × 3 workloads = 36 jobs)::

    python -m experiments.tiered_capacity.suites.run_eval_grid_parallel \\
        --jobs 6 --seeds 42 43 44 --output-root outputs/eval_grid

Run a single dataset for smoke-testing::

    python -m experiments.tiered_capacity.suites.run_eval_grid_parallel \\
        --jobs 4 --seeds 42 --dataset sharegpt

Include the shadow-price ablation::

    python -m experiments.tiered_capacity.suites.run_eval_grid_parallel \\
        --jobs 6 --seeds 42 43 44 --include-shadow-price-ablation

Pre-build caches first (recommended for first run)::

    python -m experiments.tiered_capacity.dataset_cache build
    python -m experiments.tiered_capacity.suites.run_eval_grid_parallel --jobs 6
"""

from __future__ import annotations

import argparse
import csv
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


# ---------------------------------------------------------------------------
# Lazy imports — deferred to reduce startup cost in the main process.
# Workers import the heavy modules; the main process only needs this
# module's argument parsing and job scheduling.
# ---------------------------------------------------------------------------


def _worker_imports():
    """Import heavy modules inside worker processes."""
    # These are module-level in lp_budget_eval / eval_grid, but we
    # defer them here so the main process spawns workers quickly.
    from experiments.tiered_capacity.eval_grid import (  # noqa: F811
        PAPER_GRID_VARIANTS,
        PAPER_WORKLOADS,
        SHADOW_PRICE_ABLATION_VARIANTS as GRID_SHADOW_ABLATION,
        WORKLOAD_DATASET_IDS,
        make_eval_grid_scenarios,
    )
    from experiments.tiered_capacity.lp_budget_eval import (  # noqa: F811
        SHADOW_PRICE_ABLATION_VARIANTS,
        build_hedge_delta,
        canonicalize_variant_name,
        generate_scenario_workload,
        run_variant,
        summarize_diagnostics,
        summarize_main_metrics,
    )

    return {
        "PAPER_GRID_VARIANTS": PAPER_GRID_VARIANTS,
        "PAPER_WORKLOADS": PAPER_WORKLOADS,
        "WORKLOAD_DATASET_IDS": WORKLOAD_DATASET_IDS,
        "SHADOW_PRICE_ABLATION_VARIANTS": SHADOW_PRICE_ABLATION_VARIANTS,
        "make_eval_grid_scenarios": make_eval_grid_scenarios,
        "build_hedge_delta": build_hedge_delta,
        "canonicalize_variant_name": canonicalize_variant_name,
        "generate_scenario_workload": generate_scenario_workload,
        "run_variant": run_variant,
        "summarize_diagnostics": summarize_diagnostics,
        "summarize_main_metrics": summarize_main_metrics,
    }


# ---------------------------------------------------------------------------
# Job dataclass — picklable specification for one (dataset, scenario) pair.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JobSpec:
    """One unit of work: run all variants × seeds for a single
    (dataset_name, scenario_name) pair."""

    dataset_name: str
    scenario_name: str
    variants: tuple[str, ...]
    seeds: tuple[int, ...]
    output_dir: str  # str for pickle-friendliness
    backup_scope: str


@dataclass
class JobResult:
    """Result from one completed job."""

    dataset_name: str
    scenario_name: str
    summary_rows: list[dict[str, object]]
    diagnostic_rows: list[dict[str, object]]
    delta_rows: list[dict[str, object]]
    elapsed_seconds: float
    n_requests: int
    error: str | None = None


# ---------------------------------------------------------------------------
# Hedge-delta pairs (subset relevant to the paper grid).
# ---------------------------------------------------------------------------

_DELTA_PAIRS = [
    ("original_lp", "original_lp_hedge"),
    *[
        (f"budget_range_p{p}", f"budget_range_p{p}_hedge")
        for p in (0, 25, 50, 75, 100)
    ],
    ("original_lp_rawcost", "original_lp_rawcost_hedge"),
]


# ---------------------------------------------------------------------------
# Worker function — runs inside a child process.
# ---------------------------------------------------------------------------


def _run_job(spec: JobSpec) -> JobResult:
    """Execute one (dataset, scenario) job.  Runs in a child process."""
    t0 = time.monotonic()
    try:
        mods = _worker_imports()
        scenarios = mods["make_eval_grid_scenarios"]()
        scenario = scenarios[spec.scenario_name]

        # Workload seed is stable across runs.
        wl_seed = sum(
            (i + 1) * ord(c)
            for i, c in enumerate(f"{spec.dataset_name}:{spec.scenario_name}")
        ) if spec.dataset_name != "synthetic" else 0

        workload = mods["generate_scenario_workload"](
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
                mods["run_variant"](
                    scenario,
                    workload,
                    variant,
                    seed=seed,
                    backup_scope=spec.backup_scope,
                )
                for seed in spec.seeds
            ]
            summary[variant] = mods["summarize_main_metrics"](
                scenario, evaluated_runs
            )
            diagnostics[variant] = mods["summarize_diagnostics"](evaluated_runs)

        # --- per-scenario outputs ----------------------------------------
        with (out_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        with (out_dir / "diagnostics.json").open("w") as f:
            json.dump(diagnostics, f, indent=2)

        # --- build CSV rows for merge ------------------------------------
        summary_rows = _build_summary_rows(
            spec.dataset_name, spec.scenario_name, summary
        )
        diagnostic_rows = _build_diagnostic_rows(
            spec.dataset_name, spec.scenario_name, diagnostics
        )
        delta_rows = _build_delta_rows(
            spec.dataset_name,
            spec.scenario_name,
            summary,
            mods["build_hedge_delta"],
        )

        with (out_dir / "hedge_deltas.json").open("w") as f:
            json.dump(delta_rows, f, indent=2)

        _write_csv(out_dir / "summary.csv", summary_rows)
        _write_csv(out_dir / "diagnostics.csv", diagnostic_rows)
        _write_csv(out_dir / "hedge_deltas.csv", delta_rows)

        elapsed = time.monotonic() - t0
        return JobResult(
            dataset_name=spec.dataset_name,
            scenario_name=spec.scenario_name,
            summary_rows=summary_rows,
            diagnostic_rows=diagnostic_rows,
            delta_rows=delta_rows,
            elapsed_seconds=elapsed,
            n_requests=n_requests,
        )

    except Exception:
        elapsed = time.monotonic() - t0
        return JobResult(
            dataset_name=spec.dataset_name,
            scenario_name=spec.scenario_name,
            summary_rows=[],
            diagnostic_rows=[],
            delta_rows=[],
            elapsed_seconds=elapsed,
            n_requests=0,
            error=traceback.format_exc(),
        )


# ---------------------------------------------------------------------------
# Row builders (mirrored from run_joint_lp_budget_eval.py but importable).
# ---------------------------------------------------------------------------


def _json_string(value: dict) -> str:
    return json.dumps(value, sort_keys=True)


def _build_summary_rows(
    dataset_name: str,
    scenario_name: str,
    summary: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant, metrics in summary.items():
        rows.append(
            {
                "dataset": dataset_name,
                "scenario": scenario_name,
                "variant": variant,
                "mean_ttft_ms": metrics["mean_ttft_ms"],
                "p50_ms": metrics["p50_ms"],
                "p90_ms": metrics["p90_ms"],
                "p99_ms": metrics["p99_ms"],
                "slo_violation_rate": metrics["slo_violation_rate"],
                "avg_cost_usd": metrics["avg_cost_usd"],
                "hedge_rate": metrics["hedge_rate"],
                "provider_mix": _json_string(metrics["provider_mix"]),
                "tier_mix": _json_string(metrics["tier_mix"]),
            }
        )
    return rows


def _build_diagnostic_rows(
    dataset_name: str,
    scenario_name: str,
    diagnostics: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant, diag in diagnostics.items():
        rows.append(
            {
                "dataset": dataset_name,
                "scenario": scenario_name,
                "variant": variant,
                "mean_B_tau": diag["mean_B_tau"],
                "mean_E_pi_c_eff": diag["mean_E_pi_c_eff"],
                "mean_budget_utilization": diag["mean_budget_utilization"],
                "mean_budget_slack": diag["mean_budget_slack"],
                "budget_utilization_p10": diag["budget_utilization_p10"],
                "budget_utilization_p50": diag["budget_utilization_p50"],
                "budget_utilization_p90": diag["budget_utilization_p90"],
                "budget_slack_p10": diag["budget_slack_p10"],
                "budget_slack_p50": diag["budget_slack_p50"],
                "budget_slack_p90": diag["budget_slack_p90"],
                "solver_status_counts": _json_string(diag["solver_status_counts"]),
                "fallback_counts": _json_string(diag["fallback_counts"]),
                "non_optimal_decisions": diag["non_optimal_decisions"],
                "single_feasible_provider_decisions": diag[
                    "single_feasible_provider_decisions"
                ],
                "trivial_single_provider_outcomes": diag[
                    "trivial_single_provider_outcomes"
                ],
                "true_p50_fallback_count": diag["true_p50_fallback_count"],
                "explorer_feedback_count": diag["explorer_feedback_count"],
                "backup_selection_counts": _json_string(
                    diag["backup_selection_counts"]
                ),
                "mean_backup_random_prob": diag["mean_backup_random_prob"],
                "total_decisions": diag["total_decisions"],
            }
        )
    return rows


def _build_delta_rows(
    dataset_name: str,
    scenario_name: str,
    summary: dict[str, dict[str, object]],
    build_hedge_delta_fn,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for baseline, comparison in _DELTA_PAIRS:
        if baseline not in summary or comparison not in summary:
            continue
        delta = build_hedge_delta_fn(summary[baseline], summary[comparison])
        rows.append(
            {
                "dataset": dataset_name,
                "scenario": scenario_name,
                "baseline_variant": baseline,
                "comparison_variant": comparison,
                **delta,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Merge: combine per-job results into global CSV files.
# ---------------------------------------------------------------------------


def _merge_results(
    results: list[JobResult],
    output_root: Path,
) -> None:
    """Merge all job results into global CSV files at *output_root*."""
    all_summary: list[dict[str, object]] = []
    all_diagnostics: list[dict[str, object]] = []
    all_deltas: list[dict[str, object]] = []

    for result in results:
        if result.error:
            continue
        all_summary.extend(result.summary_rows)
        all_diagnostics.extend(result.diagnostic_rows)
        all_deltas.extend(result.delta_rows)

    _write_csv(output_root / "all_results.csv", all_summary)
    _write_csv(output_root / "all_diagnostics.csv", all_diagnostics)
    _write_csv(output_root / "all_hedge_deltas.csv", all_deltas)

    # Also write a compact JSON with just the summary metrics, sorted by
    # (dataset, scenario, variant) for easy diffing.
    all_summary.sort(key=lambda r: (r["dataset"], r["scenario"], r["variant"]))
    with (output_root / "all_results.json").open("w") as f:
        json.dump(all_summary, f, indent=2)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _write_metadata(
    output_root: Path,
    *,
    datasets: list[str],
    scenarios: list[str],
    variants: list[str],
    seeds: list[int],
    jobs: int,
    backup_scope: str,
) -> None:
    meta = {
        "runner": "run_eval_grid_parallel",
        "scenario_family": "eval_grid",
        "datasets": datasets,
        "scenarios": scenarios,
        "variants": variants,
        "seeds": seeds,
        "parallel_jobs": jobs,
        "backup_scope": backup_scope,
        "trace_timing_mode": "natural_full_trace",
        "lp_tbar_source": "ground-truth analytical mean",
        "active_probing": "disabled",
    }
    with (output_root / "metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_ROOT = _ROOT / "outputs" / "eval_grid"
DEFAULT_SEEDS = [42, 43, 44]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parallel eval-grid runner.  Runs the 4-stage × 3-distribution "
            "× N-workload grid with Python-level process parallelism."
        ),
    )
    parser.add_argument(
        "--jobs", "-j",
        type=int,
        default=max(1, os.cpu_count() - 2) if os.cpu_count() else 4,
        help=(
            "Number of parallel worker processes.  Default: cpu_count - 2 "
            f"(= {max(1, os.cpu_count() - 2) if os.cpu_count() else 4} "
            f"on this machine)."
        ),
    )
    parser.add_argument(
        "--seed",
        action="append",
        dest="seeds",
        type=int,
        default=[],
        help="RNG seed.  May be repeated.  Default: 42 43 44.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        default=[],
        help=(
            "Workload dataset.  May be repeated.  Default: all 3 paper "
            "workloads (sharegpt / freeinference / rednote)."
        ),
    )
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        default=[],
        help="Scenario name.  May be repeated.  Default: all 12 grid cells.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output directory.  Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--include-shadow-price-ablation",
        action="store_true",
        help="Also run the shadow-price ablation (original_lp_rawcost*).",
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

    # We need eval_grid constants even in the main process (for planning),
    # but avoid importing the full lp_budget_eval machinery until workers.
    from experiments.tiered_capacity.eval_grid import (
        PAPER_GRID_VARIANTS,
        PAPER_WORKLOADS,
        WORKLOAD_DATASET_IDS,
        make_eval_grid_scenarios,
    )
    from experiments.tiered_capacity.lp_budget_eval import (
        SHADOW_PRICE_ABLATION_VARIANTS,
    )

    scenarios_map = make_eval_grid_scenarios()
    selected_scenarios = args.scenarios or sorted(scenarios_map)
    seeds = tuple(args.seeds or DEFAULT_SEEDS)
    datasets = args.datasets or [WORKLOAD_DATASET_IDS[w] for w in PAPER_WORKLOADS]

    # Validate scenarios.
    unknown = [s for s in selected_scenarios if s not in scenarios_map]
    if unknown:
        raise SystemExit(f"Unknown scenarios: {unknown}. Available: {sorted(scenarios_map)}")

    # Build variant list.
    variants = list(PAPER_GRID_VARIANTS)
    if args.include_shadow_price_ablation:
        variants.extend(SHADOW_PRICE_ABLATION_VARIANTS)
    variants = tuple(dict.fromkeys(variants))  # deduplicate, preserve order

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    # --- Build job specs -------------------------------------------------
    jobs: list[JobSpec] = []
    for dataset_name in datasets:
        for scenario_name in selected_scenarios:
            out_dir = output_root / dataset_name / scenario_name
            jobs.append(
                JobSpec(
                    dataset_name=dataset_name,
                    scenario_name=scenario_name,
                    variants=variants,
                    seeds=seeds,
                    output_dir=str(out_dir),
                    backup_scope=args.backup_scope,
                )
            )

    n_jobs = len(jobs)
    n_workers = min(args.jobs, n_jobs)

    print(f"Eval-grid parallel runner")
    print(f"  datasets:   {datasets}")
    print(f"  scenarios:  {len(selected_scenarios)} cells")
    print(f"  variants:   {len(variants)} ({', '.join(variants[:5])}{'...' if len(variants) > 5 else ''})")
    print(f"  seeds:      {list(seeds)}")
    print(f"  total jobs: {n_jobs} (dataset × scenario)")
    print(f"  workers:    {n_workers}")
    print(f"  output:     {output_root}")
    print()

    if args.dry_run:
        for i, job in enumerate(jobs, 1):
            print(f"  [{i:3d}] {job.dataset_name} / {job.scenario_name}")
        print(f"\n  (dry run — {n_jobs} jobs would be dispatched)")
        return

    # --- Warm / verify dataset caches ------------------------------------
    # This runs in the parent process BEFORE spawning workers, so every
    # worker can load from pickle without touching the raw source.  This
    # prevents the race where N workers all parse the same 14 GB CSV and
    # fight over the same .pkl.tmp file.
    trace_datasets = [d for d in datasets if d != "synthetic"]
    if trace_datasets:
        from experiments.tiered_capacity.dataset_cache import ensure_caches

        print("Warming dataset caches...")
        ensure_caches(trace_datasets)
        print()

    # --- Write metadata --------------------------------------------------
    _write_metadata(
        output_root,
        datasets=datasets,
        scenarios=selected_scenarios,
        variants=list(variants),
        seeds=list(seeds),
        jobs=n_workers,
        backup_scope=args.backup_scope,
    )

    # --- Dispatch --------------------------------------------------------
    t_start = time.monotonic()
    results: list[JobResult] = []
    completed = 0
    failed = 0

    # Use "spawn" to avoid fork-safety issues with numpy / scipy.
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=n_workers, mp_context=ctx
    ) as pool:
        future_to_job = {
            pool.submit(_run_job, job): job for job in jobs
        }

        for future in as_completed(future_to_job):
            job = future_to_job[future]
            result = future.result()
            results.append(result)
            completed += 1

            if result.error:
                failed += 1
                print(
                    f"  [{completed}/{n_jobs}] FAIL  "
                    f"{result.dataset_name}/{result.scenario_name} "
                    f"({result.elapsed_seconds:.1f}s)"
                )
                print(result.error)
            else:
                print(
                    f"  [{completed}/{n_jobs}] done  "
                    f"{result.dataset_name}/{result.scenario_name}  "
                    f"{result.n_requests:>7,} reqs  "
                    f"{result.elapsed_seconds:.1f}s"
                )

    # --- Merge -----------------------------------------------------------
    _merge_results(results, output_root)

    elapsed_total = time.monotonic() - t_start
    print(f"\nAll done.  {completed - failed}/{n_jobs} succeeded, "
          f"{failed} failed.  Total {elapsed_total:.0f}s")
    print(f"Results: {output_root}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
