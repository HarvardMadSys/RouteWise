"""Run the paper simulator grid evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.simulation.eval_grid import (  # noqa: E402
    PAPER_GRID_VARIANTS,
    PAPER_WORKLOADS,
    WORKLOAD_DATASET_IDS,
    make_eval_grid_scenarios,
)
from experiments.simulation.lp_budget_eval import (  # noqa: E402
    BACKUP_SCOPES,
    TRACE_WORKLOAD_DATASETS,
    build_hedge_delta,
    canonicalize_variant_name,
    generate_scenario_workload,
    run_variant,
    summarize_diagnostics,
    summarize_main_metrics,
)

OUTPUT_ROOT = _ROOT / "outputs" / "simulator_grid"
DEFAULT_SEEDS = [42, 43, 44]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paper simulator grid evaluation."
    )
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        default=[],
        help=(
            "Scenario name to run. May be repeated. Defaults to all grid "
            "cells defined in eval_grid.py."
        ),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        choices=list(TRACE_WORKLOAD_DATASETS),
        default=[],
        help=(
            "Trace workload dataset to use. May be repeated. Defaults to the "
            "paper trace workload set."
        ),
    )
    # Note: ``--trace-replay-natural`` was removed on 2026-04-29. Trace
    # replay is now unconditional natural arrival rate (see
    # _generate_trace_driven_workload docstring in lp_budget_eval.py).
    # The previous scaled-replay mode artificially compressed real
    # arrival patterns by up to 73× and produced fictitious capacity
    # stress; keeping it as opt-in left the wrong default exposed.
    parser.add_argument(
        "--seed",
        action="append",
        dest="seeds",
        type=int,
        default=[],
        help="RNG seed to run. May be repeated. Defaults to 42, 43, 44.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help=(
            "Directory where evaluation outputs should be written. Defaults to "
            "outputs/simulator_grid under the worktree root."
        ),
    )
    parser.add_argument(
        "--variant",
        action="append",
        dest="variants",
        default=[],
        help="Variant to run. May be repeated. Defaults to all main variants.",
    )
    parser.add_argument(
        "--freeze-golden",
        action="store_true",
        help="Write a sidecar golden snapshot under outputs/simulator_grid/golden/.",
    )
    # Note: ``--probe-rate`` was removed on 2026-04-29 along with the
    # active-probing helper in lp_budget_eval.py. The simulator paper line
    # uses ground-truth analytical T̄ for the LP body and does not exercise
    # probe-driven profile updates; production probing cost / capacity is
    # a real-experiment concern handled outside this runner. If a future
    # debug experiment needs active probing, write it as a self-contained
    # script under ``experiments/debug/`` rather than re-adding a flag here.
    parser.add_argument(
        "--backup-scope",
        choices=BACKUP_SCOPES,
        default="any_provider",
        help=(
            "Backup candidate scope for probability-target hedging. "
            "The default any_provider is the canonical Hedge-ProbTarget "
            "candidate set; use cross_tier only for the tier-separation ablation."
        ),
    )
    return parser.parse_args()


def _json_string(value: dict[str, float]) -> str:
    return json.dumps(value, sort_keys=True)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
                "backup_selection_counts": _json_string(diag["backup_selection_counts"]),
                "mean_backup_random_prob": diag["mean_backup_random_prob"],
                "total_decisions": diag["total_decisions"],
            }
        )
    return rows


def _build_delta_rows(
    dataset_name: str,
    scenario_name: str,
    summary: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    # Paper-level ablation deltas. Each pair answers one paper question:
    #   greedy_cost  vs routewise            : does RouteWise beat the cost baseline?
    #   greedy_latency vs routewise          : does RouteWise beat the latency baseline?
    #   random vs routewise                  : does RouteWise beat random routing?
    #   ablation_lp_only vs ablation_lp_hedging : does hedging help on top of LP?
    #   ablation_lp_hedging vs routewise     : does explorer help on top of hedging?
    pairs = [
        ("greedy_cost", "routewise"),
        ("greedy_latency", "routewise"),
        ("random", "routewise"),
        ("ablation_lp_only", "ablation_lp_hedging"),
        ("ablation_lp_hedging", "routewise"),
    ]
    for baseline, comparison in pairs:
        if baseline not in summary or comparison not in summary:
            continue
        delta = build_hedge_delta(summary[baseline], summary[comparison])
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


def _style_for_variant(variant: str) -> tuple[str, str]:
    """Return (color, marker) for a paper-name policy preset.

    Kept in sync with ``plots/palettes.py::ROUTER_STRATEGY_COLORS``. Inlined so
    this suite stays runnable without depending on the top-level ``plots``
    package, which is a separate paper-figures concern.
    """
    variant = canonicalize_variant_name(variant)
    style = {
        "greedy_cost":         ("#1f77b4", "o"),
        "greedy_latency":      ("#2ca02c", "o"),
        "random":              ("#7f8c8d", "o"),
        "ablation_lp_only":    ("#d62728", "X"),
        "ablation_lp_hedging": ("#ff7f0e", "X"),
        "routewise":           ("#9467bd", "P"),
    }
    return style.get(variant, ("#8c564b", "s"))


def _plot_tradeoff(
    summary: dict[str, dict[str, object]],
    *,
    y_key: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    plotted = sorted(summary.items(), key=lambda item: item[1]["avg_cost_usd"])
    for variant, metrics in plotted:
        color, marker = _style_for_variant(variant)
        ax.scatter(
            metrics["avg_cost_usd"],
            metrics[y_key],
            color=color,
            marker=marker,
            s=90,
            label=variant,
        )
        ax.annotate(
            variant,
            (metrics["avg_cost_usd"], metrics[y_key]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Average billed cost (USD)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _metadata(
    backup_scope: str,
    datasets: list[str],
    active_scenarios: list[str] | None = None,
    active_variants: list[str] | None = None,
    active_seeds: list[int] | None = None,
) -> dict[str, object]:
    """Build the run metadata dict.

    The catalogue lists describe what the runner *can* dispatch; the
    ``active_*`` lists describe what this specific invocation *did* dispatch.
    Reviewers and artifact evaluators should read the active lists to
    reconstruct the exact run.
    """
    return {
        "implementation_root": str(_ROOT),
        "driver": str(Path(__file__).resolve()),
        "sidecar_module": str(
            _ROOT
            / "experiments"
            / "simulation"
            / "lp_budget_eval.py"
        ),
        # ---- catalogue ----------------------------------------------------
        "policies": list(PAPER_GRID_VARIANTS),
        "paper_workloads": list(PAPER_WORKLOADS),
        # ---- active (what this run actually dispatched) -------------------
        "active_scenarios": list(active_scenarios) if active_scenarios else [],
        "active_variants": list(active_variants) if active_variants else [],
        "active_seeds": list(active_seeds) if active_seeds else [],
        "datasets": datasets,
        "backup_scope": backup_scope,
        # ---- algorithm contract (RouteWise paper §3) ----------------------
        "cost_router": (
            "Effective cost per provider j: c_eff = marginal API cost (S_A) | "
            "psi(z) = L * (U/L)^z (S_Q quota shadow price) | "
            "lambda(u) = U * u^alpha (S_C concurrency shadow price)."
        ),
        "latency_router": (
            "LP-TTFT-budget: min sum_j pi_j * Tbar_j subject to "
            "sum_j pi_j * c_eff_j <= c_min + p * (c_max - c_min), "
            "where c_min/c_max are the current feasible-provider cost envelope "
            "and p in [0, 1] is the cost-latency Pareto knob (default 0.75)."
        ),
        "hedge_rule": (
            "Probability-target: dispatch backup at the latest in-flight "
            "checkpoint t such that P(not violate | t) + P(violate | t) * "
            "P(backup succeeds in remaining SLO budget) >= 0.99, with dispatch "
            "overhead delta = 50 ms. Checkpoints are P25/P50/P75/P90 of SLO."
        ),
        "explorer_rule": (
            "When a hedge fires under the `routewise` preset, the backup's "
            "observed TTFT is fed back into the policy's rolling latency "
            "profile (hedge-as-probe). The `ablation_lp_hedging` preset "
            "disables this feedback loop."
        ),
        "workload_mode": (
            "Trace-driven replay only. The runner replays the full trace at "
            "natural arrival timestamps with no slicing, no rescaling, and no "
            "load scaling."
        ),
        "trace_timing_mode": "natural_full_trace",
        "lp_tbar_source": (
            "Ground-truth analytical expected TTFT from the provider's active "
            "distribution at decision time, not the rolling-profile sample "
            "mean. Matches the algorithm spec T_bar_j(t) = E[T_j(t)] without "
            "contaminating the LP objective with finite-sample estimator noise."
        ),
    }


def _workload_seed(dataset_name: str, scenario_name: str) -> int:
    """Return a stable workload-selection seed for one dataset/scenario pair."""
    return sum(
        (index + 1) * ord(ch)
        for index, ch in enumerate(f"{dataset_name}:{scenario_name}")
    )


def main() -> None:
    args = _parse_args()
    scenarios = make_eval_grid_scenarios()
    default_scenarios = sorted(scenarios)
    default_variants = list(PAPER_GRID_VARIANTS)
    default_datasets = [WORKLOAD_DATASET_IDS[w] for w in PAPER_WORKLOADS]

    selected_scenarios = args.scenarios or default_scenarios
    seeds = args.seeds or DEFAULT_SEEDS
    datasets = args.datasets or default_datasets

    unknown = [name for name in selected_scenarios if name not in scenarios]
    if unknown:
        raise SystemExit(
            f"Unknown scenarios: {unknown}. "
            f"Available: {sorted(scenarios)}"
        )

    variants = list(dict.fromkeys(
        canonicalize_variant_name(variant)
        for variant in (args.variants or default_variants)
    ))
    output_root = args.output_root

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "metadata.json").open("w") as handle:
        json.dump(
            _metadata(
                args.backup_scope,
                datasets,
                active_scenarios=selected_scenarios,
                active_variants=variants,
                active_seeds=seeds,
            ),
            handle,
            indent=2,
        )

    all_summary_rows: list[dict[str, object]] = []
    all_diagnostic_rows: list[dict[str, object]] = []
    all_delta_rows: list[dict[str, object]] = []
    golden_snapshot: dict[str, object] = {}

    for dataset_name in datasets:
        dataset_root = output_root / dataset_name
        dataset_root.mkdir(parents=True, exist_ok=True)

        for scenario_name in selected_scenarios:
            scenario = scenarios[scenario_name]
            print(f"\n=== {dataset_name} / {scenario_name} ===")
            workload = generate_scenario_workload(
                scenario,
                seed=_workload_seed(dataset_name, scenario_name),
                dataset_name=dataset_name,
            )
            print(
                f"  workload: {len(workload)} requests, "
                f"span={(workload[-1].timestamp - workload[0].timestamp) / 3600:.2f}h"
                if workload else "  workload: 0 requests"
            )
            scenario_dir = dataset_root / scenario_name
            scenario_dir.mkdir(parents=True, exist_ok=True)

            scenario_summary: dict[str, dict[str, object]] = {}
            scenario_diagnostics: dict[str, dict[str, object]] = {}

            for variant in variants:
                evaluated_runs = [
                    run_variant(
                        scenario,
                        workload,
                        variant,
                        seed=seed,
                        backup_scope=args.backup_scope,
                    )
                    for seed in seeds
                ]
                scenario_summary[variant] = summarize_main_metrics(scenario, evaluated_runs)
                scenario_diagnostics[variant] = summarize_diagnostics(evaluated_runs)
                metrics = scenario_summary[variant]
                print(
                    f"  {variant:<24s}"
                    f" cost={metrics['avg_cost_usd']:.2e}"
                    f" mean={metrics['mean_ttft_ms']:.0f}ms"
                    f" p50={metrics['p50_ms']:.0f}ms"
                    f" p99={metrics['p99_ms']:.0f}ms"
                    f" slo={metrics['slo_violation_rate']:.2%}"
                )

            summary_rows = _build_summary_rows(dataset_name, scenario_name, scenario_summary)
            diagnostic_rows = _build_diagnostic_rows(
                dataset_name,
                scenario_name,
                scenario_diagnostics,
            )
            delta_rows = _build_delta_rows(dataset_name, scenario_name, scenario_summary)

            with (scenario_dir / "summary.json").open("w") as handle:
                json.dump(scenario_summary, handle, indent=2)
            with (scenario_dir / "diagnostics.json").open("w") as handle:
                json.dump(scenario_diagnostics, handle, indent=2)
            with (scenario_dir / "hedge_deltas.json").open("w") as handle:
                json.dump(delta_rows, handle, indent=2)

            _write_csv(scenario_dir / "summary.csv", summary_rows)
            _write_csv(scenario_dir / "diagnostics.csv", diagnostic_rows)
            _write_csv(scenario_dir / "hedge_deltas.csv", delta_rows)

            _plot_tradeoff(
                scenario_summary,
                y_key="mean_ttft_ms",
                y_label="Mean TTFT (ms)",
                title=f"{dataset_name} / {scenario_name}: Cost vs Mean TTFT",
                output_path=scenario_dir / "cost_vs_mean_latency.png",
            )
            _plot_tradeoff(
                scenario_summary,
                y_key="p99_ms",
                y_label="P99 TTFT (ms)",
                title=f"{dataset_name} / {scenario_name}: Cost vs P99 TTFT",
                output_path=scenario_dir / "cost_vs_p99.png",
            )

            all_summary_rows.extend(summary_rows)
            all_diagnostic_rows.extend(diagnostic_rows)
            all_delta_rows.extend(delta_rows)
            golden_snapshot[f"{dataset_name}/{scenario_name}"] = {
                "summary": scenario_summary,
                "diagnostics": scenario_diagnostics,
                "hedge_deltas": delta_rows,
            }

    _write_csv(output_root / "all_results.csv", all_summary_rows)
    _write_csv(output_root / "all_diagnostics.csv", all_diagnostic_rows)
    _write_csv(output_root / "all_hedge_deltas.csv", all_delta_rows)

    if args.freeze_golden:
        golden_dir = output_root / "golden"
        golden_dir.mkdir(parents=True, exist_ok=True)
        with (golden_dir / "summary_snapshot.json").open("w") as handle:
            json.dump(golden_snapshot, handle, indent=2)

    print(f"\nDone. Results written to {output_root}")


if __name__ == "__main__":
    main()
