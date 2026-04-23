"""Run the sidecar LP-budget evaluation on merged tiered scenarios."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiment.scripts.simulate.synthetic.tiered.lp_budget_eval import (  # noqa: E402
    CONTROL_VARIANTS,
    FIRST_BATCH_SCENARIOS,
    HEDGE_ABLATION_VARIANTS,
    MAIN_VARIANTS,
    PROVIDER_PERCENTILE_ABLATION_VARIANTS,
    TRACE_WORKLOAD_DATASETS,
    build_all_scenarios,
    build_first_batch_scenarios,
    build_hedge_delta,
    canonicalize_variant_name,
    generate_scenario_workload,
    run_variant,
    summarize_diagnostics,
    summarize_main_metrics,
)

OUTPUT_ROOT = _ROOT / "results" / "lp_budget"
DEFAULT_SEEDS = [42, 43, 44]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the sidecar LP-budget evaluation on tiered scenarios."
    )
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        default=[],
        help=(
            "Scenario name to run. May be repeated. Defaults to the mandatory "
            "first-batch scenarios."
        ),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        choices=["synthetic", *TRACE_WORKLOAD_DATASETS],
        default=[],
        help=(
            "Workload dataset to use. May be repeated. Defaults to legacy "
            "synthetic workload generation. Use freeinference / rednote / "
            "sharegpt for trace-driven synthetic evaluation."
        ),
    )
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
            "results/lp_budget under the worktree root."
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
        "--skip-controls",
        action="store_true",
        help=(
            "Do not run the deterministic control baselines "
            "(cheapest_available / fastest_available / quota_first / "
            "concurrency_first)."
        ),
    )
    parser.add_argument(
        "--include-provider-percentile-ablation",
        action="store_true",
        help=(
            "Also run the older provider-percentile budget family as an ablation "
            "or comparator."
        ),
    )
    parser.add_argument(
        "--include-hedge-ablation",
        action="store_true",
        help=(
            "Also run the minimal 2x2 hedge ablation variants that keep the "
            "body selector fixed but switch back to the old hedge rule."
        ),
    )
    parser.add_argument(
        "--freeze-golden",
        action="store_true",
        help="Write a sidecar golden snapshot under results/lp_budget/golden/.",
    )
    parser.add_argument(
        "--probe-rate",
        type=float,
        default=None,
        help=(
            "Override the dedicated background probing rate for the sidecar "
            "evaluation. Use 0.0 to run a pure explorer/no-probe ablation."
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
    pairs = [
        ("original_lp", "original_lp_hedge"),
        ("original_lp", "original_lp_oldhedge"),
        ("budget_body_p25", "budget_body_p25_hedge"),
        ("budget_body_p50", "budget_body_p50_hedge"),
        ("budget_body_p75", "budget_body_p75_hedge"),
        ("budget_vhat_t25", "budget_vhat_t25_hedge"),
        ("budget_vhat_t50", "budget_vhat_t50_hedge"),
        ("budget_vhat_t75", "budget_vhat_t75_hedge"),
        ("budget_vhat_t75", "budget_vhat_t75_oldhedge"),
    ]
    for no_hedge, hedged in pairs:
        if no_hedge not in summary or hedged not in summary:
            continue
        delta = build_hedge_delta(summary[no_hedge], summary[hedged])
        rows.append(
            {
                "dataset": dataset_name,
                "scenario": scenario_name,
                "no_hedge_variant": no_hedge,
                "hedged_variant": hedged,
                **delta,
            }
        )
    return rows


def _style_for_variant(variant: str) -> tuple[str, str]:
    variant = canonicalize_variant_name(variant)
    colors = {
        "original_lp": "#1f1f1f",
        "original_lp_hedge": "#1f1f1f",
        "original_lp_oldhedge": "#1f1f1f",
        "budget_body_p25": "#1f77b4",
        "budget_body_p25_hedge": "#1f77b4",
        "budget_body_p50": "#2ca02c",
        "budget_body_p50_hedge": "#2ca02c",
        "budget_body_p75": "#d62728",
        "budget_body_p75_hedge": "#d62728",
        "budget_vhat_t25": "#ff7f0e",
        "budget_vhat_t25_hedge": "#ff7f0e",
        "budget_vhat_t50": "#9467bd",
        "budget_vhat_t50_hedge": "#9467bd",
        "budget_vhat_t75": "#8c564b",
        "budget_vhat_t75_hedge": "#8c564b",
        "budget_vhat_t75_oldhedge": "#8c564b",
        "cheapest_available": "#7f7f7f",
        "fastest_available": "#17becf",
        "quota_first": "#bcbd22",
        "concurrency_first": "#e377c2",
    }
    if variant.endswith("_oldhedge"):
        marker = "^"
    elif variant.endswith("_hedge"):
        marker = "X"
    else:
        marker = "o"
    return colors.get(variant, "#8c564b"), marker


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
    probe_rate: float | None,
    datasets: list[str],
) -> dict[str, object]:
    return {
        "implementation_root": str(_ROOT),
        "driver": str(_ROOT / "run_joint_lp_budget_eval.py"),
        "sidecar_module": str(
            _ROOT
            / "experiment"
            / "scripts"
            / "simulate"
            / "synthetic"
            / "tiered"
            / "lp_budget_eval.py"
        ),
        "main_variants": MAIN_VARIANTS,
        "hedge_ablation_variants": HEDGE_ABLATION_VARIANTS,
        "provider_percentile_ablation_variants": PROVIDER_PERCENTILE_ABLATION_VARIANTS,
        "control_variants": CONTROL_VARIANTS,
        "mandatory_first_batch_scenarios": FIRST_BATCH_SCENARIOS,
        "datasets": datasets,
        "old_selector": (
            "min sum_j pi_j * c_eff_j subject to sum_j pi_j * F_j(SLO) >= 0.99 "
            "with relaxation targets 0.98 / 0.95 / 0.90 and best-effort fallback"
        ),
        "new_selector": (
            "min sum_j pi_j * Tbar_j subject to sum_j pi_j * c_eff_j <= tau * v_hat_i"
        ),
        "hedge_rule": (
            "Dispatch backup at the latest wait time t such that "
            "P(not violate | t) + P(violate | t) * P(backup succeeds) >= 0.99, "
            "with dispatch overhead delta = 50 ms"
        ),
        "ablation_selector": (
            "Comparator only: min sum_j pi_j * Tbar_j subject to "
            "sum_j pi_j * c_eff_j <= percentile_tau({c_eff_j over feasible providers})"
        ),
        "v_hat_i_assumption": (
            "v_hat_i is implemented as the estimated per-request API price anchor, "
            "computed as request.total_tokens multiplied by the cheapest S_A "
            "provider price in the current synthetic scenario"
        ),
        "workload_mode": (
            "synthetic generator by default; trace-driven mode uses real request "
            "traces and rescales contiguous slices into each scenario's target "
            "duration while preserving token counts and local burst structure"
        ),
        "hedge_as_probe": (
            "Enabled for all hedged variants: when a hedge fires, the backup "
            "TTFT sample is fed back into the rolling profile while keeping the "
            "5% dedicated background probe path unchanged"
        ),
        "probe_rate_override": probe_rate,
        "backup_selection_policy": (
            "New hedge variants use an adaptive safe-cheapest / random-explorer "
            "backup mix; oldhedge variants retain the legacy fastest-backup rule"
        ),
    }


def _workload_seed(dataset_name: str, scenario_name: str) -> int:
    """Return a stable workload-selection seed for one dataset/scenario pair."""
    if dataset_name == "synthetic":
        return 0
    return sum(
        (index + 1) * ord(ch)
        for index, ch in enumerate(f"{dataset_name}:{scenario_name}")
    )


def main() -> None:
    args = _parse_args()
    scenarios = build_all_scenarios()
    selected_scenarios = args.scenarios or FIRST_BATCH_SCENARIOS
    seeds = args.seeds or DEFAULT_SEEDS
    datasets = args.datasets or ["synthetic"]
    variants = [
        canonicalize_variant_name(variant)
        for variant in (args.variants or list(MAIN_VARIANTS))
    ]
    if args.include_hedge_ablation:
        variants.extend(HEDGE_ABLATION_VARIANTS)
    if args.include_provider_percentile_ablation:
        variants.extend(PROVIDER_PERCENTILE_ABLATION_VARIANTS)
    if not args.skip_controls:
        variants.extend(CONTROL_VARIANTS)
    variants = list(dict.fromkeys(canonicalize_variant_name(variant) for variant in variants))
    output_root = args.output_root

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "metadata.json").open("w") as handle:
        json.dump(_metadata(args.probe_rate, datasets), handle, indent=2)

    all_summary_rows: list[dict[str, object]] = []
    all_diagnostic_rows: list[dict[str, object]] = []
    all_delta_rows: list[dict[str, object]] = []
    golden_snapshot: dict[str, object] = {}

    use_legacy_synthetic_layout = len(datasets) == 1 and datasets[0] == "synthetic"

    for dataset_name in datasets:
        dataset_root = output_root if use_legacy_synthetic_layout else output_root / dataset_name
        dataset_root.mkdir(parents=True, exist_ok=True)

        for scenario_name in selected_scenarios:
            scenario = scenarios[scenario_name]
            print(f"\n=== {dataset_name} / {scenario_name} ===")
            workload = generate_scenario_workload(
                scenario,
                seed=_workload_seed(dataset_name, scenario_name),
                dataset_name=dataset_name,
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
                        probe_rate=(0.05 if args.probe_rate is None else args.probe_rate),
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
