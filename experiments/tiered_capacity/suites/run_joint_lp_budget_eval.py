"""Run the sidecar LP-budget evaluation on merged tiered scenarios."""

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

from experiments.tiered_capacity.eval_grid import (  # noqa: E402
    PAPER_GRID_VARIANTS,
    PAPER_WORKLOADS,
    WORKLOAD_DATASET_IDS,
    make_eval_grid_scenarios,
)
from experiments.tiered_capacity.lp_budget_eval import (  # noqa: E402
    BACKUP_EXPLORATION_VARIANTS,
    BACKUP_SCOPES,
    CONTROL_VARIANTS,
    EXPLORER_VARIANTS,
    FIRST_BATCH_SCENARIOS,
    HEDGE_ABLATION_VARIANTS,
    MAIN_VARIANTS,
    PROVIDER_PERCENTILE_ABLATION_VARIANTS,
    SHADOW_PRICE_ABLATION_VARIANTS,
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

OUTPUT_ROOT = _ROOT / "outputs" / "lp_budget"
DEFAULT_SEEDS = [42, 43, 44]

# Scenario family identifiers exposed via --scenario-family. The legacy
# family preserves all pre-existing behaviour; eval_grid switches to the
# 3-stage × 3-distribution grid declared in eval_grid.py.
SCENARIO_FAMILY_LEGACY = "legacy"
SCENARIO_FAMILY_EVAL_GRID = "eval_grid"
SCENARIO_FAMILIES = (SCENARIO_FAMILY_LEGACY, SCENARIO_FAMILY_EVAL_GRID)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the sidecar LP-budget evaluation on tiered scenarios."
    )
    parser.add_argument(
        "--scenario-family",
        choices=SCENARIO_FAMILIES,
        default=SCENARIO_FAMILY_LEGACY,
        help=(
            "Which scenario family to evaluate. 'legacy' (default) runs the "
            "hand-authored S6/S7/S8/S9/unified_pool scenarios. 'eval_grid' "
            "runs the 3-stage x 3-distribution grid defined in eval_grid.py "
            "and switches the variant / dataset defaults to PAPER_GRID_VARIANTS "
            "and PAPER_WORKLOADS respectively."
        ),
    )
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        default=[],
        help=(
            "Scenario name to run. May be repeated. Defaults to the mandatory "
            "first-batch scenarios for legacy family, or all 9 grid cells for "
            "the eval_grid family."
        ),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        choices=["synthetic", *TRACE_WORKLOAD_DATASETS],
        default=[],
        help=(
            "Workload dataset to use. May be repeated. Defaults to built-in "
            "synthetic workload generation. Use freeinference / rednote / "
            "sharegpt for trace-driven synthetic evaluation."
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
            "outputs/lp_budget under the worktree root."
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
        "--include-backup-exploration-ablation",
        action="store_true",
        help=(
            "Also run variants that enable adaptive random backup selection. "
            "Default *_hedge and *_explorer variants use deterministic backups."
        ),
    )
    parser.add_argument(
        "--include-shadow-price-ablation",
        action="store_true",
        help=(
            "Also run the shadow-price ablation: ``original_lp_rawcost*`` "
            "uses raw marginal cost (zero for S_Q / S_C subscription tiers) "
            "instead of effective_cost. Pair with ``original_lp*`` to "
            "isolate the contribution of ψ(z) / λ(u) shadow prices. Stays "
            "off by default so legacy main runs are not polluted."
        ),
    )
    parser.add_argument(
        "--freeze-golden",
        action="store_true",
        help="Write a sidecar golden snapshot under outputs/lp_budget/golden/.",
    )
    parser.add_argument(
        "--probe-rate",
        type=float,
        default=None,
        help=(
            "Override the dedicated background probing rate for the sidecar "
            "evaluation. Use 0.0 with *_explorer variants to run a pure "
            "hedge-as-probe/no-dedicated-probe ablation."
        ),
    )
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
    pairs = [
        ("original_lp", "original_lp_hedge"),
        ("original_lp_hedge", "original_lp_explorer"),
        ("original_lp_hedge", "original_lp_hedge_randombackup"),
        ("original_lp_explorer", "original_lp_explorer_randombackup"),
        ("original_lp", "original_lp_oldhedge"),
        ("budget_body_p25", "budget_body_p25_hedge"),
        ("budget_body_p50", "budget_body_p50_hedge"),
        ("budget_body_p75", "budget_body_p75_hedge"),
        ("budget_range_p0", "budget_range_p0_hedge"),
        ("budget_range_p0_hedge", "budget_range_p0_explorer"),
        ("budget_range_p0_hedge", "budget_range_p0_hedge_randombackup"),
        ("budget_range_p0_explorer", "budget_range_p0_explorer_randombackup"),
        ("budget_range_p25", "budget_range_p25_hedge"),
        ("budget_range_p25_hedge", "budget_range_p25_explorer"),
        ("budget_range_p25_hedge", "budget_range_p25_hedge_randombackup"),
        ("budget_range_p25_explorer", "budget_range_p25_explorer_randombackup"),
        ("budget_range_p50", "budget_range_p50_hedge"),
        ("budget_range_p50_hedge", "budget_range_p50_explorer"),
        ("budget_range_p50_hedge", "budget_range_p50_hedge_randombackup"),
        ("budget_range_p50_explorer", "budget_range_p50_explorer_randombackup"),
        ("budget_range_p75", "budget_range_p75_hedge"),
        ("budget_range_p75_hedge", "budget_range_p75_explorer"),
        ("budget_range_p75_hedge", "budget_range_p75_hedge_randombackup"),
        ("budget_range_p75_explorer", "budget_range_p75_explorer_randombackup"),
        ("budget_range_p75", "budget_range_p75_oldhedge"),
        ("budget_range_p100", "budget_range_p100_hedge"),
        ("budget_range_p100_hedge", "budget_range_p100_explorer"),
        ("budget_range_p100_hedge", "budget_range_p100_hedge_randombackup"),
        ("budget_range_p100_explorer", "budget_range_p100_explorer_randombackup"),
        ("budget_vhat_t25", "budget_vhat_t25_hedge"),
        ("budget_vhat_t25_hedge", "budget_vhat_t25_explorer"),
        ("budget_vhat_t25_hedge", "budget_vhat_t25_hedge_randombackup"),
        ("budget_vhat_t25_explorer", "budget_vhat_t25_explorer_randombackup"),
        ("budget_vhat_t50", "budget_vhat_t50_hedge"),
        ("budget_vhat_t50_hedge", "budget_vhat_t50_explorer"),
        ("budget_vhat_t50_hedge", "budget_vhat_t50_hedge_randombackup"),
        ("budget_vhat_t50_explorer", "budget_vhat_t50_explorer_randombackup"),
        ("budget_vhat_t75", "budget_vhat_t75_hedge"),
        ("budget_vhat_t75_hedge", "budget_vhat_t75_explorer"),
        ("budget_vhat_t75_hedge", "budget_vhat_t75_hedge_randombackup"),
        ("budget_vhat_t75_explorer", "budget_vhat_t75_explorer_randombackup"),
        ("budget_vhat_t75", "budget_vhat_t75_oldhedge"),
        ("budget_vhat_t100", "budget_vhat_t100_hedge"),
        ("budget_vhat_t100_hedge", "budget_vhat_t100_explorer"),
        ("budget_vhat_t100_hedge", "budget_vhat_t100_hedge_randombackup"),
        ("budget_vhat_t100_explorer", "budget_vhat_t100_explorer_randombackup"),
        ("budget_vhat_t150", "budget_vhat_t150_hedge"),
        ("budget_vhat_t150_hedge", "budget_vhat_t150_explorer"),
        ("budget_vhat_t150_hedge", "budget_vhat_t150_hedge_randombackup"),
        ("budget_vhat_t150_explorer", "budget_vhat_t150_explorer_randombackup"),
        ("budget_vhat_t200", "budget_vhat_t200_hedge"),
        ("budget_vhat_t200_hedge", "budget_vhat_t200_explorer"),
        ("budget_vhat_t200_hedge", "budget_vhat_t200_hedge_randombackup"),
        ("budget_vhat_t200_explorer", "budget_vhat_t200_explorer_randombackup"),
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
    variant = canonicalize_variant_name(variant)
    random_backup = variant.endswith("_randombackup")
    style_variant = variant[:-len("_randombackup")] if random_backup else variant
    colors = {
        "original_lp": "#1f1f1f",
        "original_lp_hedge": "#1f1f1f",
        "original_lp_explorer": "#1f1f1f",
        "original_lp_oldhedge": "#1f1f1f",
        "budget_body_p25": "#1f77b4",
        "budget_body_p25_hedge": "#1f77b4",
        "budget_body_p50": "#2ca02c",
        "budget_body_p50_hedge": "#2ca02c",
        "budget_body_p75": "#d62728",
        "budget_body_p75_hedge": "#d62728",
        "budget_range_p0": "#1f77b4",
        "budget_range_p0_hedge": "#1f77b4",
        "budget_range_p0_explorer": "#1f77b4",
        "budget_range_p25": "#2ca02c",
        "budget_range_p25_hedge": "#2ca02c",
        "budget_range_p25_explorer": "#2ca02c",
        "budget_range_p50": "#d62728",
        "budget_range_p50_hedge": "#d62728",
        "budget_range_p50_explorer": "#d62728",
        "budget_range_p75": "#9467bd",
        "budget_range_p75_hedge": "#9467bd",
        "budget_range_p75_explorer": "#9467bd",
        "budget_range_p75_oldhedge": "#9467bd",
        "budget_range_p100": "#8c564b",
        "budget_range_p100_hedge": "#8c564b",
        "budget_range_p100_explorer": "#8c564b",
        "budget_vhat_t25": "#ff7f0e",
        "budget_vhat_t25_hedge": "#ff7f0e",
        "budget_vhat_t25_explorer": "#ff7f0e",
        "budget_vhat_t50": "#9467bd",
        "budget_vhat_t50_hedge": "#9467bd",
        "budget_vhat_t50_explorer": "#9467bd",
        "budget_vhat_t75": "#8c564b",
        "budget_vhat_t75_hedge": "#8c564b",
        "budget_vhat_t75_explorer": "#8c564b",
        "budget_vhat_t75_oldhedge": "#8c564b",
        "budget_vhat_t100": "#e377c2",
        "budget_vhat_t100_hedge": "#e377c2",
        "budget_vhat_t100_explorer": "#e377c2",
        "budget_vhat_t150": "#7f7f7f",
        "budget_vhat_t150_hedge": "#7f7f7f",
        "budget_vhat_t150_explorer": "#7f7f7f",
        "budget_vhat_t200": "#bcbd22",
        "budget_vhat_t200_hedge": "#bcbd22",
        "budget_vhat_t200_explorer": "#bcbd22",
        "cheapest_available": "#7f7f7f",
        "fastest_available": "#17becf",
        "quota_first": "#bcbd22",
        "concurrency_first": "#e377c2",
    }
    if random_backup:
        marker = "*"
    elif variant.endswith("_oldhedge"):
        marker = "^"
    elif variant.endswith("_explorer"):
        marker = "P"
    elif variant.endswith("_hedge"):
        marker = "X"
    else:
        marker = "o"
    return colors.get(style_variant, "#8c564b"), marker


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
    backup_scope: str,
    datasets: list[str],
    scenario_family: str = SCENARIO_FAMILY_LEGACY,
    active_scenarios: list[str] | None = None,
    active_variants: list[str] | None = None,
    active_seeds: list[int] | None = None,
) -> dict[str, object]:
    """Build the run metadata dict.

    The catalogue lists (``main_variants`` etc.) describe what the runner
    *can* dispatch; the ``active_*`` lists describe what this specific
    invocation *did* dispatch. Reviewers and artifact evaluators should
    read the active lists to reconstruct the exact run.
    """
    return {
        "implementation_root": str(_ROOT),
        "driver": str(Path(__file__).resolve()),
        "sidecar_module": str(
            _ROOT
            / "experiments"
            / "tiered_capacity"
            / "lp_budget_eval.py"
        ),
        "scenario_family": scenario_family,
        # ---- catalogue (everything the runner knows about) ---------------
        "main_variants": MAIN_VARIANTS,
        "explorer_variants": EXPLORER_VARIANTS,
        "backup_exploration_variants": BACKUP_EXPLORATION_VARIANTS,
        "hedge_ablation_variants": HEDGE_ABLATION_VARIANTS,
        "provider_percentile_ablation_variants": PROVIDER_PERCENTILE_ABLATION_VARIANTS,
        "shadow_price_ablation_variants": SHADOW_PRICE_ABLATION_VARIANTS,
        "control_variants": CONTROL_VARIANTS,
        "mandatory_first_batch_scenarios": FIRST_BATCH_SCENARIOS,
        "paper_grid_variants": list(PAPER_GRID_VARIANTS),
        "paper_workloads": list(PAPER_WORKLOADS),
        # ---- active (what this run actually dispatched) -------------------
        "active_scenarios": list(active_scenarios) if active_scenarios else [],
        "active_variants": list(active_variants) if active_variants else [],
        "active_seeds": list(active_seeds) if active_seeds else [],
        "datasets": datasets,
        "old_selector": (
            "min sum_j pi_j * c_eff_j subject to sum_j pi_j * F_j(SLO) >= 0.99 "
            "with relaxation targets 0.98 / 0.95 / 0.90 and best-effort fallback"
        ),
        "new_selector": (
            "Canonical budget_range_p* selector: min sum_j pi_j * Tbar_j subject to "
            "sum_j pi_j * c_eff_j <= c_min + p * (c_max - c_min), "
            "where c_min/c_max are the current feasible provider cost envelope"
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
        "range_budget_assumption": (
            "Canonical LP-TTFT-budget variants use p in [0, 1] as an interpretable "
            "cost-latency knob over the feasible c_eff range for each request"
        ),
        "legacy_v_hat_i_assumption": (
            "Legacy budget_vhat_t* comparison variants still use v_hat_i as the "
            "estimated per-request API price anchor, computed from the cheapest "
            "SLO-safe S_A provider at the current time; if no SLO-safe S_A exists, "
            "it falls back to the cheapest S_A provider"
        ),
        "workload_mode": (
            "synthetic generator (Poisson with LogNormal output tokens) for "
            "--dataset synthetic; for trace-driven datasets (sharegpt / "
            "freeinference / rednote) the runner replays the **full** trace at "
            "natural arrival timestamps with no slicing, no rescaling, no load "
            "scaling. The simulated time span varies by workload (sharegpt 7d, "
            "freeinference 89d, rednote 83d). The earlier scaled-replay path "
            "(contiguous-slice 2000 requests + rescale to scenario.duration_seconds) "
            "was removed on 2026-04-29 because its compression artefacts (1.2× "
            "sharegpt, 11.5× freeinference, 73× rednote) fabricated capacity "
            "stress that does not exist in production."
        ),
        "trace_timing_mode": "natural_full_trace",
        "lp_tbar_source": (
            "ground-truth analytical expected TTFT (provider.ttft_dist.mean()), "
            "not the rolling-profile sample mean. Matches the algorithm spec "
            "T_bar_j(t) = E[T_j] without contaminating the LP objective with "
            "finite-sample estimator noise. See _body_latency_proxy_ms in "
            "experiments/tiered_capacity/lp_budget_eval.py and "
            "DESIGN_PRINCIPLES.md §1 ('reasoning beats realism')."
        ),
        "hedge_as_probe": (
            "Enabled only for explicit *_explorer variants: when a hedge fires, "
            "the backup TTFT sample is fed back into the rolling profile. "
            "*_hedge variants are hedge-only and do not update the backup profile."
        ),
        "probe_rate_override": probe_rate,
        "backup_scope": backup_scope,
        "backup_selection_policy": (
            "Default *_hedge and *_explorer variants use deterministic "
            "safe-cheapest / fastest-fallback backup selection. Explicit "
            "*_randombackup variants enable the adaptive random-backup branch. "
            "The canonical backup candidate set is any available non-primary "
            "provider. backup_scope=cross_tier is an explicit tier-separation "
            "ablation. Oldhedge variants retain the historical fastest-backup rule."
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


def _resolve_scenario_family(
    args: argparse.Namespace,
) -> tuple[dict, list[str], list[str], list[str]]:
    """Return (scenarios, default_scenarios, default_variants, default_datasets).

    Encapsulates the family-specific defaults so ``main`` stays linear.
    Legacy family preserves the historical defaults exactly; eval_grid
    swaps the scenario source, the variant list, and the dataset list.
    """
    if args.scenario_family == SCENARIO_FAMILY_EVAL_GRID:
        scenarios = make_eval_grid_scenarios()
        default_scenarios = sorted(scenarios)
        default_variants = list(PAPER_GRID_VARIANTS)
        default_datasets = [WORKLOAD_DATASET_IDS[w] for w in PAPER_WORKLOADS]
        return scenarios, default_scenarios, default_variants, default_datasets

    # Legacy: untouched behaviour.
    scenarios = build_all_scenarios()
    return scenarios, list(FIRST_BATCH_SCENARIOS), list(MAIN_VARIANTS), ["synthetic"]


def main() -> None:
    args = _parse_args()
    scenarios, default_scenarios, default_variants, default_datasets = (
        _resolve_scenario_family(args)
    )
    selected_scenarios = args.scenarios or default_scenarios
    seeds = args.seeds or DEFAULT_SEEDS
    datasets = args.datasets or default_datasets

    # Validate that every requested scenario exists in the chosen family;
    # the misleading KeyError path used to swallow eval_grid name typos.
    unknown = [name for name in selected_scenarios if name not in scenarios]
    if unknown:
        raise SystemExit(
            f"Unknown scenarios for family={args.scenario_family!r}: {unknown}. "
            f"Available: {sorted(scenarios)}"
        )

    variants = [
        canonicalize_variant_name(variant)
        for variant in (args.variants or default_variants)
    ]
    # Ablation / control flags only make sense for the legacy family — the
    # eval_grid variants are deliberately scoped to PAPER_GRID_VARIANTS.
    if args.scenario_family == SCENARIO_FAMILY_EVAL_GRID:
        # Only warn when the user *explicitly* passed an include-* ablation
        # flag, since those defaults are False (so a True value is always
        # user-set). Don't warn on `--skip-controls` because its default
        # (False) is shared with "user didn't pass it" — that would emit
        # noise on every default invocation.
        explicit_ablation_flags = [
            name
            for name, value in (
                ("--include-hedge-ablation", args.include_hedge_ablation),
                ("--include-backup-exploration-ablation",
                 args.include_backup_exploration_ablation),
                ("--include-provider-percentile-ablation",
                 args.include_provider_percentile_ablation),
                ("--include-shadow-price-ablation",
                 args.include_shadow_price_ablation),
            )
            if value
        ]
        if explicit_ablation_flags:
            # Shadow-price ablation IS still meaningful under eval_grid
            # (it's the cleanest way to isolate ψ(z)/λ(u) value on
            # Stage 3 cells), so honour it even though the other ablation
            # flags don't apply. Warn about the others only.
            other_flags = [f for f in explicit_ablation_flags
                           if f != "--include-shadow-price-ablation"]
            if other_flags:
                print(
                    "[run_joint_lp_budget_eval] eval_grid family ignores "
                    f"{', '.join(other_flags)}; PAPER_GRID_VARIANTS is the "
                    f"canonical {len(PAPER_GRID_VARIANTS)}-variant set.",
                    file=sys.stderr,
                )
        if args.include_shadow_price_ablation:
            variants.extend(SHADOW_PRICE_ABLATION_VARIANTS)
        # Controls are silently skipped for eval_grid regardless of
        # --skip-controls because they pollute the Pareto frontier.
    else:  # SCENARIO_FAMILY_LEGACY
        if args.include_hedge_ablation:
            variants.extend(HEDGE_ABLATION_VARIANTS)
        if args.include_backup_exploration_ablation:
            variants.extend(BACKUP_EXPLORATION_VARIANTS)
        if args.include_provider_percentile_ablation:
            variants.extend(PROVIDER_PERCENTILE_ABLATION_VARIANTS)
        if args.include_shadow_price_ablation:
            variants.extend(SHADOW_PRICE_ABLATION_VARIANTS)
        if not args.skip_controls:
            variants.extend(CONTROL_VARIANTS)
    variants = list(dict.fromkeys(canonicalize_variant_name(variant) for variant in variants))
    output_root = args.output_root

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "metadata.json").open("w") as handle:
        json.dump(
            _metadata(
                args.probe_rate,
                args.backup_scope,
                datasets,
                scenario_family=args.scenario_family,
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

    use_flat_synthetic_layout = len(datasets) == 1 and datasets[0] == "synthetic"

    for dataset_name in datasets:
        dataset_root = output_root if use_flat_synthetic_layout else output_root / dataset_name
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
                        # probe_rate=None lets run_variant inherit its own
                        # default (PROBE_RATE = 0.0, "no dedicated probing"
                        # per Juncheng's active-system assumption). Pass
                        # --probe-rate <float> on the CLI to opt into an
                        # explicit active-probing ablation.
                        probe_rate=args.probe_rate,
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
