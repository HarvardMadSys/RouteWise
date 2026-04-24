#!/usr/bin/env python3
r"""Plot Phase 4 Smart Hedging Results.

Generates visualization figures for hedging strategy evaluation:
1. P99 latency vs SLO for different strategies
2. SLO violation rate comparison
3. Hedge rate vs SLO
4. Cost overhead vs SLO
5. Cost vs Violation Pareto frontier

Usage:
    python -m experiments.latency_phase4.plots \
        --results-dir experiment/results/latency_phase4/
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.paper_plots.common import (
    STRATEGY_COLORS,
    STRATEGY_LABELS,
    apply_style,
)

# Apply default style
apply_style("paper")

# Backup method markers
BACKUP_MARKERS = {
    "fastest": "o",
    "lp_other": "s",
    "cheapest_viable": "^",
}


def _parse_strategy_name(strategy: str) -> tuple[str, float | None]:
    """Parse strategy name and extract base strategy and optional alpha.

    The Phase 4 ablation may encode alpha in the strategy field, e.g.:
    - fixed_timeout_0.5, fixed_timeout_0.7, fixed_timeout_0.9

    Args:
        strategy: Raw strategy string from CSV.

    Returns:
        (base_strategy, alpha) where alpha is None if not present.
    """
    if strategy.startswith("fixed_timeout_"):
        suffix = strategy.removeprefix("fixed_timeout_")
        try:
            return "fixed_timeout", float(suffix)
        except ValueError:
            return "fixed_timeout", None
    return strategy, None


def _format_strategy_label(strategy: str) -> str:
    """Return a readable label for a strategy key."""
    base, alpha = _parse_strategy_name(strategy)
    label = STRATEGY_LABELS.get(base, base)
    if base == "fixed_timeout" and alpha is not None:
        return f"{label} (alpha={alpha:g})"
    return label


def _strategy_color(strategy: str) -> str:
    """Return the color for a strategy key."""
    base, _ = _parse_strategy_name(strategy)
    return STRATEGY_COLORS.get(base, "#333333")


def load_ablation_results(results_dir: str) -> pd.DataFrame:
    """Load all ablation CSV files from directory."""
    dfs = []
    for f in Path(results_dir).glob("ablation_*.csv"):
        df = pd.read_csv(f)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No ablation_*.csv files found in {results_dir}")

    return pd.concat(dfs, ignore_index=True)


def _maybe_filter_single_slo(df: pd.DataFrame, slo_sec: float | None) -> pd.DataFrame:
    """Filter results to a single SLO if provided, or validate if multiple exist."""
    unique_slos = sorted(df["slo_sec"].unique().tolist())
    if slo_sec is None:
        if len(unique_slos) > 1:
            raise ValueError(
                "Multiple SLOs found in results-dir. Pass --slo to select one. "
                f"Found slo_sec={unique_slos}."
            )
        return df
    if slo_sec not in unique_slos:
        raise ValueError(f"Requested --slo {slo_sec} not found. Available slo_sec={unique_slos}.")
    return df[df["slo_sec"] == slo_sec].copy()


def plot_p99_vs_strategy(df: pd.DataFrame, output_dir: str) -> None:
    """Plot P99 latency for each strategy."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter to fastest backup method for clarity
    df_fastest = df[df["backup_method"] == "fastest"]

    strategies = df_fastest["strategy"].unique()
    x = np.arange(len(strategies))
    width = 0.6

    p99_values = []
    colors = []
    for s in strategies:
        row = df_fastest[df_fastest["strategy"] == s].iloc[0]
        p99_values.append(row["p99_ms"])
        colors.append(_strategy_color(s))

    bars = ax.bar(x, p99_values, width, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, p99_values, strict=True):
        ax.annotate(
            f"{val:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("P99 Latency (ms)")
    ax.set_xlabel("Hedging Strategy")
    ax.set_xticks(x)
    ax.set_xticklabels([_format_strategy_label(s) for s in strategies], rotation=15, ha="right")

    # Add SLO line
    slo_ms = df_fastest["slo_sec"].iloc[0] * 1000
    ax.axhline(y=slo_ms, color="red", linestyle="--", linewidth=1.5, label=f"SLO ({slo_ms:.0f}ms)")
    ax.legend()

    ax.set_title("P99 Latency by Hedging Strategy")
    ax.grid(axis="y", alpha=0.3)

    output_file = os.path.join(output_dir, "p99_vs_strategy.png")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved: {output_file}")


def plot_violation_rate(df: pd.DataFrame, output_dir: str) -> None:
    """Plot SLO violation rate comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    df_fastest = df[df["backup_method"] == "fastest"]
    strategies = df_fastest["strategy"].unique()
    x = np.arange(len(strategies))
    width = 0.6

    violation_rates = []
    colors = []
    for s in strategies:
        row = df_fastest[df_fastest["strategy"] == s].iloc[0]
        violation_rates.append(row["slo_violation_rate"] * 100)
        colors.append(_strategy_color(s))

    bars = ax.bar(x, violation_rates, width, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, violation_rates, strict=True):
        ax.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("SLO Violation Rate (%)")
    ax.set_xlabel("Hedging Strategy")
    ax.set_xticks(x)
    ax.set_xticklabels([_format_strategy_label(s) for s in strategies], rotation=15, ha="right")
    ax.set_title("SLO Violation Rate by Hedging Strategy")
    ax.grid(axis="y", alpha=0.3)

    output_file = os.path.join(output_dir, "violation_rate_vs_strategy.png")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved: {output_file}")


def plot_hedge_rate(df: pd.DataFrame, output_dir: str) -> None:
    """Plot hedge rate for each strategy."""
    fig, ax = plt.subplots(figsize=(10, 6))

    df_fastest = df[df["backup_method"] == "fastest"]
    strategies = df_fastest["strategy"].unique()
    x = np.arange(len(strategies))
    width = 0.6

    hedge_rates = []
    colors = []
    for s in strategies:
        row = df_fastest[df_fastest["strategy"] == s].iloc[0]
        hedge_rates.append(row["hedge_rate"] * 100)
        colors.append(_strategy_color(s))

    bars = ax.bar(x, hedge_rates, width, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, hedge_rates, strict=True):
        ax.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Hedge Rate (%)")
    ax.set_xlabel("Hedging Strategy")
    ax.set_xticks(x)
    ax.set_xticklabels([_format_strategy_label(s) for s in strategies], rotation=15, ha="right")
    ax.set_title("Hedge Trigger Rate by Strategy")
    ax.grid(axis="y", alpha=0.3)

    output_file = os.path.join(output_dir, "hedge_rate_vs_strategy.png")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved: {output_file}")


def plot_cost_overhead(df: pd.DataFrame, output_dir: str) -> None:
    """Plot cost overhead vs baseline."""
    fig, ax = plt.subplots(figsize=(10, 6))

    df_fastest = df[df["backup_method"] == "fastest"]

    # Get baseline cost (never hedge)
    baseline_cost = df_fastest[df_fastest["strategy"] == "never"]["avg_cost"].iloc[0]

    strategies = df_fastest["strategy"].unique()
    x = np.arange(len(strategies))
    width = 0.6

    cost_overheads = []
    colors = []
    for s in strategies:
        row = df_fastest[df_fastest["strategy"] == s].iloc[0]
        overhead = (row["avg_cost"] / baseline_cost - 1) * 100
        cost_overheads.append(overhead)
        colors.append(_strategy_color(s))

    bars = ax.bar(x, cost_overheads, width, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, cost_overheads, strict=True):
        label = f"+{val:.1f}%" if val >= 0 else f"{val:.1f}%"
        y_pos = bar.get_height() if bar.get_height() >= 0 else 0
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, y_pos),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Cost Overhead vs Baseline (%)")
    ax.set_xlabel("Hedging Strategy")
    ax.set_xticks(x)
    ax.set_xticklabels([_format_strategy_label(s) for s in strategies], rotation=15, ha="right")
    ax.set_title("Cost Overhead by Hedging Strategy")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    output_file = os.path.join(output_dir, "cost_overhead_vs_strategy.png")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved: {output_file}")


def plot_pareto_frontier(df: pd.DataFrame, output_dir: str) -> None:
    """Plot cost vs violation Pareto frontier."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get baseline cost for normalization
    df_never = df[(df["strategy"] == "never") & (df["backup_method"] == "fastest")]
    baseline_cost = df_never["avg_cost"].iloc[0] if len(df_never) > 0 else df["avg_cost"].min()

    # Plot each strategy-backup curve (across SLOs if present).
    for strategy in sorted(df["strategy"].unique()):
        for backup in sorted(df["backup_method"].unique()):
            subset = df[(df["strategy"] == strategy) & (df["backup_method"] == backup)].copy()
            if subset.empty:
                continue

            subset = subset.sort_values("slo_sec")
            x = subset["slo_violation_rate"].to_numpy() * 100
            y = subset["avg_cost"].to_numpy() / baseline_cost

            color = _strategy_color(strategy)
            marker = BACKUP_MARKERS.get(backup, "o")

            label = _format_strategy_label(strategy)
            if backup != "fastest":
                label += f" ({backup})"

            ax.plot(
                x,
                y,
                color=color,
                marker=marker,
                markersize=7,
                linewidth=1.5,
                label=label,
            )

    ax.set_xlabel("SLO Violation Rate (%)")
    ax.set_ylabel("Normalized Cost (1x = baseline)")
    ax.set_title("Cost vs SLO Violation Trade-off")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # Mark Pareto-optimal points
    # (Lower left is better)

    output_file = os.path.join(output_dir, "cost_vs_violation_pareto.png")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved: {output_file}")


def plot_backup_method_comparison(df: pd.DataFrame, output_dir: str) -> None:
    """Compare backup selection methods for smart strategies."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Filter to smart strategies only
    smart_strategies = ["smart_survival", "smart_residual"]
    df_smart = df[df["strategy"].isin(smart_strategies)].copy()

    if len(df_smart) == 0:
        print("No smart strategy results found, skipping backup comparison plot")
        return

    # Plot 1: P99 latency by backup method
    ax = axes[0]
    backup_methods = df_smart["backup_method"].unique()
    x = np.arange(len(backup_methods))
    width = 0.35

    for i, strategy in enumerate(smart_strategies):
        p99_values = []
        for backup in backup_methods:
            row = df_smart[
                (df_smart["strategy"] == strategy) & (df_smart["backup_method"] == backup)
            ]
            if len(row) > 0:
                p99_values.append(row["p99_ms"].iloc[0])
            else:
                p99_values.append(0)

        offset = width * (i - 0.5)
        ax.bar(
            x + offset,
            p99_values,
            width,
            label=_format_strategy_label(strategy),
            color=_strategy_color(strategy),
        )

    ax.set_ylabel("P99 Latency (ms)")
    ax.set_xlabel("Backup Selection Method")
    ax.set_xticks(x)
    ax.set_xticklabels(backup_methods)
    ax.legend()
    ax.set_title("P99 Latency by Backup Method")
    ax.grid(axis="y", alpha=0.3)

    # Plot 2: Cost by backup method
    ax = axes[1]
    baseline_cost = df[df["strategy"] == "never"]["avg_cost"].iloc[0]

    for i, strategy in enumerate(smart_strategies):
        cost_values = []
        for backup in backup_methods:
            row = df_smart[
                (df_smart["strategy"] == strategy) & (df_smart["backup_method"] == backup)
            ]
            if len(row) > 0:
                cost_values.append(row["avg_cost"].iloc[0] / baseline_cost)
            else:
                cost_values.append(0)

        offset = width * (i - 0.5)
        ax.bar(
            x + offset,
            cost_values,
            width,
            label=_format_strategy_label(strategy),
            color=_strategy_color(strategy),
        )

    ax.set_ylabel("Normalized Cost")
    ax.set_xlabel("Backup Selection Method")
    ax.set_xticks(x)
    ax.set_xticklabels(backup_methods)
    ax.axhline(y=1, color="black", linestyle="--", linewidth=0.5, label="Baseline")
    ax.legend()
    ax.set_title("Cost by Backup Method")
    ax.grid(axis="y", alpha=0.3)

    output_file = os.path.join(output_dir, "backup_method_comparison.png")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved: {output_file}")


def generate_summary_table(df: pd.DataFrame, output_dir: str) -> None:
    """Generate summary table as CSV."""
    # Aggregate by strategy (fastest backup only for main comparison)
    df_fastest = df[df["backup_method"] == "fastest"].copy()

    summary = df_fastest[
        [
            "strategy",
            "slo_sec",
            "slo_violation_rate",
            "hedge_rate",
            "avg_cost",
            "p50_ms",
            "p90_ms",
            "p99_ms",
        ]
    ].copy()

    # Add relative metrics
    baseline = summary[summary["strategy"] == "never"].iloc[0]
    summary["cost_overhead_pct"] = (summary["avg_cost"] / baseline["avg_cost"] - 1) * 100
    summary["p99_reduction_pct"] = (
        (baseline["p99_ms"] - summary["p99_ms"]) / baseline["p99_ms"] * 100
    )

    output_file = os.path.join(output_dir, "summary_table.csv")
    summary.to_csv(output_file, index=False, float_format="%.4f")
    print(f"Saved: {output_file}")

    # Print to console
    print("\n" + "=" * 80)
    print("Summary Table")
    print("=" * 80)
    print(summary.to_string(index=False))


def main():
    """Plot Phase 4 simulation results."""
    parser = argparse.ArgumentParser(description="Plot Phase 4 Results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiment/results/latency_phase4",
        help="Directory containing simulation results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for plots (default: same as results-dir)",
    )
    parser.add_argument(
        "--slo",
        type=float,
        default=None,
        help="Filter results to a single SLO (required if results-dir has multiple SLOs).",
    )

    args = parser.parse_args()

    output_dir = args.output if args.output else args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading results from {args.results_dir}...")

    try:
        df = load_ablation_results(args.results_dir)
        print(f"  Loaded {len(df)} result rows")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the ablation study first: python run_phase4_simulation.py --ablation")
        return

    try:
        df = _maybe_filter_single_slo(df, args.slo)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Generate all plots
    print("\nGenerating plots...")
    plot_p99_vs_strategy(df, output_dir)
    plot_violation_rate(df, output_dir)
    plot_hedge_rate(df, output_dir)
    plot_cost_overhead(df, output_dir)
    plot_pareto_frontier(df, output_dir)
    plot_backup_method_comparison(df, output_dir)

    # Generate summary table
    generate_summary_table(df, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
