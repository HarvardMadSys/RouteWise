#!/usr/bin/env python3
r"""Visualization for Phase 5 Online Evaluation Results.

Generates comparison plots:
- Cost vs P99 latency scatter
- SLO violation rate bars
- Provider distribution
- Time series analysis

Usage:
    python -m experiment.scripts.plot.latency.phase5_online \
        --input experiment/results/phase5_online/run_xxx
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment.scripts.plot.common import (
    POLICY_COLORS,
    POLICY_LABELS,
    apply_style,
)

# Apply default style
apply_style("paper")

# Use common colors and labels
COLORS = POLICY_COLORS
LABELS = POLICY_LABELS


def load_data(input_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Load evaluation log and summary.

    Args:
        input_dir: Directory containing evaluation results.

    Returns:
        (log_df, summary_dict)
    """
    log_path = input_dir / "evaluation_log.csv"
    summary_path = input_dir / "evaluation_summary.json"

    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    log_df = pd.read_csv(log_path)

    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    return log_df, summary


def plot_cost_vs_p99(summary: dict, output_dir: Path) -> None:
    """Create cost vs P99 latency scatter plot.

    Args:
        summary: Evaluation summary dict.
        output_dir: Output directory for plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for policy, stats in summary.items():
        if stats["total_requests"] == 0:
            continue

        color = COLORS.get(policy, "#333333")
        label = LABELS.get(policy, policy)

        ax.scatter(
            stats["avg_cost"] * 1000,  # Convert to $/1000 requests
            stats["p99_ms"],
            s=150,
            c=color,
            label=label,
            edgecolors="black",
            linewidths=1,
            zorder=3,
        )

    ax.set_xlabel("Cost ($/1000 requests)", fontsize=12)
    ax.set_ylabel("P99 TTFT (ms)", fontsize=12)
    ax.set_title("Phase 5: Cost vs Latency Trade-off", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add Pareto frontier annotation
    policies = list(summary.keys())
    costs = [summary[p]["avg_cost"] * 1000 for p in policies]
    p99s = [summary[p]["p99_ms"] for p in policies]

    # Find Pareto-optimal points
    pareto_points = []
    for i, (c, p) in enumerate(zip(costs, p99s, strict=False)):
        is_dominated = False
        for j, (c2, p2) in enumerate(zip(costs, p99s, strict=False)):
            if i != j and c2 <= c and p2 <= p and (c2 < c or p2 < p):
                is_dominated = True
                break
        if not is_dominated:
            pareto_points.append((c, p, policies[i]))

    if len(pareto_points) > 1:
        pareto_points.sort(key=lambda x: x[0])
        pareto_costs = [p[0] for p in pareto_points]
        pareto_p99s = [p[1] for p in pareto_points]
        ax.plot(
            pareto_costs,
            pareto_p99s,
            "k--",
            alpha=0.5,
            linewidth=1.5,
            label="Pareto Frontier",
            zorder=2,
        )

    plt.tight_layout()
    output_path = output_dir / "cost_vs_p99_scatter.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_slo_violation_bars(summary: dict, output_dir: Path, slo_sec: float = 2.0) -> None:
    """Create SLO violation rate bar chart.

    Args:
        summary: Evaluation summary dict.
        output_dir: Output directory for plot.
        slo_sec: SLO in seconds for title.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    policies = list(summary.keys())
    violation_rates = [summary[p]["slo_violation_rate"] * 100 for p in policies]
    colors = [COLORS.get(p, "#333333") for p in policies]
    labels = [LABELS.get(p, p) for p in policies]

    x = np.arange(len(policies))
    bars = ax.bar(x, violation_rates, color=colors, edgecolor="black", linewidth=1)

    ax.set_xlabel("Policy", fontsize=12)
    ax.set_ylabel("SLO Violation Rate (%)", fontsize=12)
    ax.set_title(f"Phase 5: SLO Violation Rate (SLO = {slo_sec}s)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)

    # Add value labels on bars
    for bar, rate in zip(bars, violation_rates, strict=False):
        height = bar.get_height()
        ax.annotate(
            f"{rate:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylim(0, max(violation_rates) * 1.2 if violation_rates else 10)
    plt.tight_layout()

    output_path = output_dir / "slo_violation_bars.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_provider_distribution(summary: dict, output_dir: Path) -> None:
    """Create provider distribution stacked bar chart.

    Args:
        summary: Evaluation summary dict.
        output_dir: Output directory for plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Collect all providers
    all_providers = set()
    for stats in summary.values():
        if "provider_distribution" in stats:
            all_providers.update(stats["provider_distribution"].keys())

    all_providers = sorted(all_providers)
    if not all_providers:
        print("No provider distribution data available")
        return

    # Build data matrix
    policies = list(summary.keys())
    labels = [LABELS.get(p, p) for p in policies]
    data = []

    for provider in all_providers:
        provider_data = []
        for policy in policies:
            dist = summary[policy].get("provider_distribution", {})
            provider_data.append(dist.get(provider, 0) * 100)
        data.append(provider_data)

    # Create stacked bar chart
    x = np.arange(len(policies))
    bottom = np.zeros(len(policies))

    # Color palette for providers
    provider_colors = plt.cm.tab20(np.linspace(0, 1, len(all_providers)))

    for i, (provider, pdata) in enumerate(zip(all_providers, data, strict=False)):
        ax.bar(
            x,
            pdata,
            bottom=bottom,
            label=provider,
            color=provider_colors[i],
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += pdata

    ax.set_xlabel("Policy", fontsize=12)
    ax.set_ylabel("Provider Selection (%)", fontsize=12)
    ax.set_title("Phase 5: Provider Distribution by Policy", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=9,
        title="Provider",
    )
    ax.set_ylim(0, 100)

    plt.tight_layout()
    output_path = output_dir / "provider_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_latency_timeseries(log_df: pd.DataFrame, output_dir: Path) -> None:
    """Create latency time series plot.

    Args:
        log_df: Evaluation log DataFrame.
        output_dir: Output directory for plot.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Convert timestamp to relative minutes
    log_df = log_df.copy()
    log_df["relative_time"] = (log_df["timestamp"] - log_df["timestamp"].min()) / 60

    # Filter successful requests
    success_df = log_df[log_df["status"] == "success"]

    for policy in success_df["policy"].unique():
        policy_df = success_df[success_df["policy"] == policy]
        color = COLORS.get(policy, "#333333")
        label = LABELS.get(policy, policy)

        # Rolling average for smoothing
        if len(policy_df) > 10:
            rolling_avg = policy_df["ttft_ms"].rolling(window=10, min_periods=1).mean()
        else:
            rolling_avg = policy_df["ttft_ms"]

        ax.plot(
            policy_df["relative_time"],
            rolling_avg,
            color=color,
            label=label,
            alpha=0.8,
            linewidth=1.5,
        )

    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_ylabel("TTFT (ms, 10-point rolling avg)", fontsize=12)
    ax.set_title("Phase 5: Latency Over Time", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "latency_timeseries.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_latency_ecdf(log_df: pd.DataFrame, output_dir: Path) -> None:
    """Create latency ECDF plot.

    Args:
        log_df: Evaluation log DataFrame.
        output_dir: Output directory for plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter successful requests
    success_df = log_df[log_df["status"] == "success"]

    for policy in success_df["policy"].unique():
        policy_df = success_df[success_df["policy"] == policy]
        latencies = sorted(policy_df["ttft_ms"].values)

        if not latencies:
            continue

        color = COLORS.get(policy, "#333333")
        label = LABELS.get(policy, policy)

        # Compute ECDF
        n = len(latencies)
        y = np.arange(1, n + 1) / n

        ax.step(latencies, y, where="post", color=color, label=label, linewidth=2)

    ax.set_xlabel("TTFT (ms)", fontsize=12)
    ax.set_ylabel("Cumulative Probability", fontsize=12)
    ax.set_title("Phase 5: Latency ECDF", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)

    # Add P99 line
    ax.axhline(y=0.99, color="gray", linestyle="--", alpha=0.5, label="P99")

    plt.tight_layout()
    output_path = output_dir / "latency_ecdf.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_summary_table(summary: dict, output_dir: Path) -> None:
    """Create summary table as image.

    Args:
        summary: Evaluation summary dict.
        output_dir: Output directory for plot.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    # Prepare table data
    columns = [
        "Policy",
        "Requests",
        "Error %",
        "SLO Viol %",
        "Avg Cost ($)",
        "P50 (ms)",
        "P90 (ms)",
        "P99 (ms)",
    ]

    rows = []
    for policy, stats in summary.items():
        label = LABELS.get(policy, policy)
        rows.append(
            [
                label,
                stats["total_requests"],
                f"{stats['error_rate']*100:.1f}",
                f"{stats['slo_violation_rate']*100:.1f}",
                f"{stats['avg_cost']:.6f}",
                f"{stats['p50_ms']:.0f}",
                f"{stats['p90_ms']:.0f}",
                f"{stats['p99_ms']:.0f}",
            ]
        )

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colColours=["#f0f0f0"] * len(columns),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    ax.set_title("Phase 5: Evaluation Summary", fontsize=14, pad=20)

    plt.tight_layout()
    output_path = output_dir / "summary_table.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_cost_breakdown(log_df: pd.DataFrame, summary: dict, output_dir: Path) -> None:
    """Create cost breakdown chart.

    Args:
        log_df: Evaluation log DataFrame.
        summary: Evaluation summary dict.
        output_dir: Output directory for plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    policies = list(summary.keys())
    total_costs = [summary[p]["total_cost"] for p in policies]
    avg_costs = [summary[p]["avg_cost"] * 1000 for p in policies]
    colors = [COLORS.get(p, "#333333") for p in policies]
    labels = [LABELS.get(p, p) for p in policies]

    # Total cost bar chart
    x = np.arange(len(policies))
    bars1 = ax1.bar(x, total_costs, color=colors, edgecolor="black", linewidth=1)
    ax1.set_xlabel("Policy", fontsize=12)
    ax1.set_ylabel("Total Cost ($)", fontsize=12)
    ax1.set_title("Total Cost", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)

    for bar, cost in zip(bars1, total_costs, strict=False):
        ax1.annotate(
            f"${cost:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Average cost bar chart
    bars2 = ax2.bar(x, avg_costs, color=colors, edgecolor="black", linewidth=1)
    ax2.set_xlabel("Policy", fontsize=12)
    ax2.set_ylabel("Avg Cost ($/1000 requests)", fontsize=12)
    ax2.set_title("Average Cost per Request", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)

    for bar, cost in zip(bars2, avg_costs, strict=False):
        ax2.annotate(
            f"${cost:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    output_path = output_dir / "cost_breakdown.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Generate all Phase 5 visualization plots."""
    parser = argparse.ArgumentParser(description="Phase 5 Results Visualization")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing evaluation results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for plots (default: same as input)",
    )
    parser.add_argument(
        "--slo",
        type=float,
        default=2.0,
        help="SLO in seconds (for plot titles)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {input_dir}...")
    log_df, summary = load_data(input_dir)
    print(f"  Loaded {len(log_df)} records")
    print(f"  Policies: {list(summary.keys())}")

    print(f"\nGenerating plots to {output_dir}...")

    # Generate all plots
    if summary:
        plot_cost_vs_p99(summary, output_dir)
        plot_slo_violation_bars(summary, output_dir, slo_sec=args.slo)
        plot_provider_distribution(summary, output_dir)
        plot_summary_table(summary, output_dir)
        plot_cost_breakdown(log_df, summary, output_dir)

    if not log_df.empty:
        plot_latency_timeseries(log_df, output_dir)
        plot_latency_ecdf(log_df, output_dir)

    print("\nAll plots generated!")


if __name__ == "__main__":
    main()
