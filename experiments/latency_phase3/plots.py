#!/usr/bin/env python3
r"""Plot Phase 3 simulation results.

Generates visualizations for:
1. Cost vs SLO violation rate across different SLOs
2. Latency distribution comparison
3. LP-Mix routing weights over time

Usage:
    python -m experiments.latency_phase3.plots \
        --results-dir experiment/results/latency_phase3/
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.paper_plots.common import (
    apply_style,
)

# Apply default style
apply_style("paper")

# Phase 3 specific colors and labels
COLORS = {
    "single_best": "#1f77b4",
    "cheapest": "#ff7f0e",
    "random": "#2ca02c",
    "lp_mix": "#d62728",
}

MARKERS = {
    "single_best": "o",
    "cheapest": "s",
    "random": "^",
    "lp_mix": "D",
}

LABELS = {
    "single_best": "Single-Best (P99)",
    "cheapest": "Cheapest",
    "random": "Random",
    "lp_mix": "LP-Mix (Ours)",
}


def plot_cost_vs_violation(results_df: pd.DataFrame, output_dir: str):
    """Plot cost vs SLO violation rate for all policies."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Group by policy
    for policy in ["single_best", "cheapest", "random", "lp_mix"]:
        policy_data = results_df[results_df["policy"] == policy]
        if len(policy_data) == 0:
            continue

        ax.scatter(
            policy_data["slo_violation_rate"] * 100,
            policy_data["avg_cost"] * 1e6,  # Convert to micro-dollars
            c=COLORS[policy],
            marker=MARKERS[policy],
            s=100,
            label=LABELS[policy],
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
        )

        # Connect points with lines
        sorted_data = policy_data.sort_values("slo_sec")
        ax.plot(
            sorted_data["slo_violation_rate"] * 100,
            sorted_data["avg_cost"] * 1e6,
            c=COLORS[policy],
            linestyle="--",
            alpha=0.5,
        )

    ax.set_xlabel("SLO Violation Rate (%)")
    ax.set_ylabel("Average Cost per Request (micro-$)")
    ax.set_title("Cost vs SLO Violation Trade-off")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, "cost_vs_violation.png")
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")


def plot_slo_sweep(results_df: pd.DataFrame, output_dir: str):
    """Plot metrics across different SLO values."""
    slo_values = sorted(results_df["slo_sec"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: Cost by SLO
    ax = axes[0]
    for policy in ["single_best", "cheapest", "random", "lp_mix"]:
        policy_data = results_df[results_df["policy"] == policy]
        if len(policy_data) == 0:
            continue
        sorted_data = policy_data.sort_values("slo_sec")
        ax.plot(
            sorted_data["slo_sec"],
            sorted_data["avg_cost"] * 1e6,
            marker=MARKERS[policy],
            color=COLORS[policy],
            label=LABELS[policy],
            markersize=8,
        )
    ax.set_xlabel("SLO (seconds)")
    ax.set_ylabel("Avg Cost (micro-$)")
    ax.set_title("Cost by SLO")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: SLO Violation Rate by SLO
    ax = axes[1]
    for policy in ["single_best", "cheapest", "random", "lp_mix"]:
        policy_data = results_df[results_df["policy"] == policy]
        if len(policy_data) == 0:
            continue
        sorted_data = policy_data.sort_values("slo_sec")
        ax.plot(
            sorted_data["slo_sec"],
            sorted_data["slo_violation_rate"] * 100,
            marker=MARKERS[policy],
            color=COLORS[policy],
            label=LABELS[policy],
            markersize=8,
        )
    ax.set_xlabel("SLO (seconds)")
    ax.set_ylabel("SLO Violation Rate (%)")
    ax.set_title("Violation Rate by SLO")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="1% target")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: P99 Latency by SLO
    ax = axes[2]
    for policy in ["single_best", "cheapest", "random", "lp_mix"]:
        policy_data = results_df[results_df["policy"] == policy]
        if len(policy_data) == 0:
            continue
        sorted_data = policy_data.sort_values("slo_sec")
        ax.plot(
            sorted_data["slo_sec"],
            sorted_data["p99_ms"] / 1000,  # Convert to seconds
            marker=MARKERS[policy],
            color=COLORS[policy],
            label=LABELS[policy],
            markersize=8,
        )
    ax.set_xlabel("SLO (seconds)")
    ax.set_ylabel("P99 Latency (seconds)")
    ax.set_title("P99 Latency by SLO")
    # Add diagonal line showing SLO = P99
    ax.plot(slo_values, slo_values, "k--", alpha=0.3, label="SLO = P99")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, "slo_sweep.png")
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")


def plot_routing_weights_over_time(log_file: str, output_dir: str):
    """Plot LP-Mix routing weights over time."""
    with open(log_file) as f:
        routing_log = json.load(f)

    # Extract timestamps and weights
    timestamps = []
    all_weights = []  # List of dicts

    for entry in routing_log:
        if "lp_weights" not in entry or not entry["lp_weights"]:
            continue

        timestamps.append(entry["timestamp"])
        all_weights.append(entry["lp_weights"])

    if not timestamps:
        print("No LP weights in routing log")
        return

    # Get all unique providers
    all_providers = set()
    for w in all_weights:
        all_providers.update(w.keys())
    providers = sorted(all_providers)

    # Build weight arrays (fill with 0 for missing providers)
    weight_data = {p: [] for p in providers}
    for w in all_weights:
        for p in providers:
            weight_data[p].append(w.get(p, 0.0))

    # Normalize timestamps to hours from start
    t0 = min(timestamps)
    hours = [(t - t0) / 3600 for t in timestamps]

    # Plot stacked area chart
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = plt.cm.tab20(np.linspace(0, 1, len(providers)))

    y_stack = np.zeros(len(timestamps))
    for i, provider in enumerate(providers):
        weights = np.array(weight_data[provider])
        ax.fill_between(
            hours,
            y_stack,
            y_stack + weights,
            label=provider,
            color=colors[i],
            alpha=0.7,
        )
        y_stack += weights

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Routing Weight")
    ax.set_title("LP-Mix Provider Weights Over Time")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, "routing_weights_over_time.png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")


def plot_latency_comparison(log_files: dict, output_dir: str):
    """Plot latency distribution comparison across policies."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for policy, log_file in log_files.items():
        if not os.path.exists(log_file):
            continue

        with open(log_file) as f:
            routing_log = json.load(f)

        latencies = [
            entry["ttft_ms"]
            for entry in routing_log
            if entry["ttft_ms"] > 0 and entry["error"] is None
        ]

        if not latencies:
            continue

        # Plot ECDF
        sorted_lat = np.sort(latencies)
        ecdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)
        ax.plot(
            sorted_lat / 1000,  # Convert to seconds
            ecdf,
            color=COLORS.get(policy, "gray"),
            label=LABELS.get(policy, policy),
            linewidth=2,
        )

    ax.set_xlabel("Latency (seconds)")
    ax.set_ylabel("CDF")
    ax.set_title("Latency Distribution Comparison")
    ax.set_xlim(0, 5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, "latency_ecdf.png")
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")


def main():
    """Plot Phase 3 simulation results."""
    parser = argparse.ArgumentParser(description="Plot Phase 3 Results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiment/results/latency_phase3",
        help="Directory containing simulation results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for plots (default: same as results-dir)",
    )

    args = parser.parse_args()

    output_dir = args.output or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    # Find all simulation result CSV files
    result_files = [f for f in os.listdir(args.results_dir) if f.startswith("simulation_results")]

    if not result_files:
        print(f"No simulation results found in {args.results_dir}")
        return

    # Load and combine results
    all_results = []
    for f in result_files:
        df = pd.read_csv(os.path.join(args.results_dir, f))
        all_results.append(df)

    results_df = pd.concat(all_results, ignore_index=True)
    print(f"Loaded {len(results_df)} result rows from {len(result_files)} files")

    # Generate plots
    if len(results_df["slo_sec"].unique()) > 1:
        plot_slo_sweep(results_df, output_dir)
        plot_cost_vs_violation(results_df, output_dir)

    # Plot routing weights if log file exists
    slo_values = results_df["slo_sec"].unique()
    for slo in slo_values:
        log_file = os.path.join(args.results_dir, f"routing_log_lp_mix_slo{slo}.json")
        if os.path.exists(log_file):
            plot_routing_weights_over_time(log_file, output_dir)
            break  # Just plot one example

    # Plot latency comparison
    for slo in slo_values:
        log_files = {
            policy: os.path.join(args.results_dir, f"routing_log_{policy}_slo{slo}.json")
            for policy in ["single_best", "cheapest", "random", "lp_mix"]
        }
        if all(os.path.exists(f) for f in log_files.values()):
            plot_latency_comparison(log_files, output_dir)
            break

    print("\nDone!")


if __name__ == "__main__":
    main()
