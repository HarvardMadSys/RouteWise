#!/usr/bin/env python3
"""Plotting script for Online Stage 2 experiments (ICML style).

Online Stage 2 evaluates online routing strategies for joint optimization
of daily quota (S_Q) and concurrency-limited (S_C) subscriptions.

Figures generated:
- Cost comparison bar chart
- Competitive ratio comparison
- Request distribution (stacked bar)
- S_C config sensitivity analysis

Usage:
    python plot_online_stage2.py --all
    python plot_online_stage2.py --cost
    python plot_online_stage2.py --sensitivity
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from experiment.scripts.plot.common import add_light_grid, apply_style

# =============================================================================
# Style Configuration
# =============================================================================

COLORS = {
    "All-API": "#8C8C8C",  # Gray (baseline)
    "SQ-Only": "#4C72B0",  # Blue
    "SC-Only": "#55A868",  # Green
    "Greedy-Online-Stage2": "#C44E52",  # Red
    "PrimalDual-Online-Stage2": "#8172B2",  # Purple
    "LA-PD-Unified-Stage2": "#DD8452",  # Orange (our method, highlight)
    "Offline-Optimal-ILP": "#937860",  # Brown
}

STRATEGY_LABELS = {
    "All-API": "All-API",
    "SQ-Only": "S_Q-Only",
    "SC-Only": "S_C-Only",
    "Greedy-Online-Stage2": "Greedy",
    "PrimalDual-Online-Stage2": "PD",
    "LA-PD-Unified-Stage2": "LA-PD (Ours)",
    "Offline-Optimal-ILP": "Offline-OPT",
}

# Order for display (excluding All-API for some plots)
STRATEGY_ORDER = [
    "SQ-Only",
    "SC-Only",
    "Greedy-Online-Stage2",
    "PrimalDual-Online-Stage2",
    "LA-PD-Unified-Stage2",
    "Offline-Optimal-ILP",
]

# S_C config display names
SC_CONFIG_LABELS = {
    "local_gpu": "Local GPU\n($0/mo)",
    "featherless_premium": "Premium\n($25/mo)",
    "featherless_scale": "Scale\n($75/mo)",
}


# =============================================================================
# Data Loading
# =============================================================================


def load_results(results_path: Path) -> dict[str, Any]:
    """Load results from JSON file."""
    with open(results_path) as f:
        return json.load(f)


def load_all_sc_configs(results_dir: Path) -> dict[str, dict]:
    """Load results for all S_C configurations."""
    configs = {}
    for sc_config in ["local_gpu", "featherless_premium", "featherless_scale"]:
        path = results_dir / f"stage2_online_results_{sc_config}.json"
        if path.exists():
            configs[sc_config] = load_results(path)
    return configs


# =============================================================================
# Cost Comparison Figures
# =============================================================================


def plot_cost_comparison(data: dict, output_dir: Path, sc_config: str):
    """Plot total cost comparison bar chart."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    strategies = data.get("strategies", data)
    all_api_cost = strategies["All-API"]["costs"]["total"]

    names = []
    costs = []
    colors = []

    # Add All-API first
    names.append(STRATEGY_LABELS["All-API"])
    costs.append(all_api_cost)
    colors.append(COLORS["All-API"])

    for key in STRATEGY_ORDER:
        if key in strategies:
            names.append(STRATEGY_LABELS.get(key, key))
            costs.append(strategies[key]["costs"]["total"])
            colors.append(COLORS.get(key, "#666666"))

    x = np.arange(len(names))
    bars = ax.bar(x, costs, color=colors, edgecolor="white", linewidth=0.5)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, costs, strict=False)):
        fontweight = "bold" if "LA-PD" in names[i] or "Ours" in names[i] else "normal"
        ax.annotate(
            f"${val:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight=fontweight,
        )

    ax.set_ylabel("Total Cost ($)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, max(costs) * 1.12)
    add_light_grid(ax, "y")

    plt.tight_layout()
    output_path = output_dir / f"online_stage2_cost_{sc_config}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_competitive_ratio(data: dict, output_dir: Path, sc_config: str):
    """Plot competitive ratio comparison."""
    fig, ax = plt.subplots(figsize=(7, 4))

    strategies = data.get("strategies", data)

    names = []
    ratios = []
    colors = []

    for key in STRATEGY_ORDER:
        if key in strategies and key != "Offline-Optimal-ILP":
            ratio = strategies[key].get("competitive_ratio", 1.0)
            if ratio is not None:
                names.append(STRATEGY_LABELS.get(key, key))
                ratios.append(ratio)
                colors.append(COLORS.get(key, "#666666"))

    x = np.arange(len(names))
    bars = ax.bar(x, ratios, color=colors, edgecolor="white", linewidth=0.5)

    # Add optimal line
    ax.axhline(y=1.0, color="#333333", linestyle="--", linewidth=1.5, label="Optimal (1.0)")

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, ratios, strict=False)):
        fontweight = "bold" if "LA-PD" in names[i] or "Ours" in names[i] else "normal"
        ax.annotate(
            f"{val:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight=fontweight,
        )

    ax.set_ylabel("Competitive Ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0.95, max(ratios) * 1.08 if ratios else 1.5)
    ax.legend(loc="upper right", fontsize=9)
    add_light_grid(ax, "y")

    plt.tight_layout()
    output_path = output_dir / f"online_stage2_competitive_ratio_{sc_config}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_request_distribution(data: dict, output_dir: Path, sc_config: str):
    """Plot request routing distribution as stacked bar chart."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    strategies = data.get("strategies", data)

    names = []
    sub_counts = []
    api_counts = []

    for key in STRATEGY_ORDER:
        if key in strategies:
            names.append(STRATEGY_LABELS.get(key, key))
            sub_counts.append(strategies[key]["requests"]["subscription"])
            api_counts.append(strategies[key]["requests"]["api"])

    x = np.arange(len(names))
    width = 0.6

    totals = [s + a for s, a in zip(sub_counts, api_counts, strict=False)]
    sub_pcts = [s / t * 100 if t > 0 else 0 for s, t in zip(sub_counts, totals, strict=False)]

    bars1 = ax.bar(
        x,
        sub_counts,
        width,
        label="Subscription (S_Q + S_C)",
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.bar(
        x,
        api_counts,
        width,
        bottom=sub_counts,
        label="API (S_A)",
        color="#DD8452",
        edgecolor="white",
        linewidth=0.5,
    )

    # Add percentage labels
    for bar, pct in zip(bars1, sub_pcts, strict=False):
        if pct > 5:
            ax.annotate(
                f"{pct:.0f}%",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
            )

    ax.set_ylabel("Number of Requests")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend(loc="upper right", fontsize=9)
    add_light_grid(ax, "y")

    plt.tight_layout()
    output_path = output_dir / f"online_stage2_request_dist_{sc_config}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# S_C Config Sensitivity Analysis
# =============================================================================


def plot_sc_sensitivity(all_configs: dict[str, dict], output_dir: Path):
    """Plot sensitivity to S_C configuration."""
    if len(all_configs) < 2:
        print("Skipping S_C sensitivity: need at least 2 configurations")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    sc_configs = list(all_configs.keys())
    sc_labels = [SC_CONFIG_LABELS.get(c, c) for c in sc_configs]

    # Focus strategies for sensitivity
    focus_strategies = [
        "SQ-Only",
        "Greedy-Online-Stage2",
        "LA-PD-Unified-Stage2",
    ]

    # (a) Total Cost by S_C Config
    ax = axes[0]
    x = np.arange(len(sc_configs))
    width = 0.25
    offsets = np.linspace(-width, width, len(focus_strategies))

    for i, strategy in enumerate(focus_strategies):
        costs = []
        for sc_config in sc_configs:
            data = all_configs[sc_config]
            strategies = data.get("strategies", data)
            if strategy in strategies:
                costs.append(strategies[strategy]["costs"]["total"])
            else:
                costs.append(0)

        ax.bar(
            x + offsets[i],
            costs,
            width * 0.9,
            label=STRATEGY_LABELS.get(strategy, strategy),
            color=COLORS.get(strategy, "#666666"),
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_ylabel("Total Cost ($)")
    ax.set_xticks(x)
    ax.set_xticklabels(sc_labels)
    ax.set_xlabel("S_C Configuration")
    ax.legend(loc="upper right", fontsize=9)
    add_light_grid(ax, "y")
    ax.set_title("(a) Total Cost by S_C Config", fontsize=11)

    # (b) Competitive Ratio by S_C Config
    ax = axes[1]
    for _i, strategy in enumerate(focus_strategies):
        if strategy == "SQ-Only":
            continue  # Skip for ratio plot (no S_C)

        ratios = []
        for sc_config in sc_configs:
            data = all_configs[sc_config]
            strategies = data.get("strategies", data)
            if strategy in strategies:
                ratio = strategies[strategy].get("competitive_ratio", 1.0)
                ratios.append(ratio if ratio is not None else 1.0)
            else:
                ratios.append(1.0)

        ax.plot(
            x,
            ratios,
            "o-",
            label=STRATEGY_LABELS.get(strategy, strategy),
            color=COLORS.get(strategy, "#666666"),
            linewidth=2,
            markersize=8,
        )

    ax.axhline(y=1.0, color="#333333", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_ylabel("Competitive Ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(sc_labels)
    ax.set_xlabel("S_C Configuration")
    ax.legend(loc="upper right", fontsize=9)
    add_light_grid(ax, "y")
    ax.set_title("(b) Competitive Ratio by S_C Config", fontsize=11)

    plt.tight_layout()
    output_path = output_dir / "online_stage2_sc_sensitivity.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_combined_figure(data: dict, output_dir: Path, sc_config: str):
    """Create combined figure with cost and competitive ratio."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    strategies = data.get("strategies", data)
    all_api_cost = strategies["All-API"]["costs"]["total"]

    # (a) Cost Comparison
    ax = axes[0]
    names = []
    costs = []
    colors = []

    for key in STRATEGY_ORDER:
        if key in strategies:
            names.append(STRATEGY_LABELS.get(key, key))
            costs.append(strategies[key]["costs"]["total"])
            colors.append(COLORS.get(key, "#666666"))

    x = np.arange(len(names))
    bars = ax.bar(x, costs, color=colors, edgecolor="white", linewidth=0.5)

    # Add savings annotation
    for i, (bar, cost) in enumerate(zip(bars, costs, strict=False)):
        savings_pct = (all_api_cost - cost) / all_api_cost * 100
        fontweight = "bold" if "LA-PD" in names[i] or "Ours" in names[i] else "normal"
        ax.annotate(
            f"-{savings_pct:.0f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight=fontweight,
        )

    # Add All-API reference line
    ax.axhline(y=all_api_cost, color="#8C8C8C", linestyle="--", linewidth=1.5, label="All-API")

    ax.set_ylabel("Total Cost ($)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, all_api_cost * 1.1)
    ax.legend(loc="upper right", fontsize=9)
    add_light_grid(ax, "y")
    ax.set_title("(a) Cost Savings vs All-API", fontsize=11)

    # (b) Competitive Ratio
    ax = axes[1]
    names = []
    ratios = []
    colors = []

    for key in STRATEGY_ORDER:
        if key in strategies and key != "Offline-Optimal-ILP":
            ratio = strategies[key].get("competitive_ratio")
            if ratio is not None:
                names.append(STRATEGY_LABELS.get(key, key))
                ratios.append(ratio)
                colors.append(COLORS.get(key, "#666666"))

    x = np.arange(len(names))
    bars = ax.bar(x, ratios, color=colors, edgecolor="white", linewidth=0.5)

    ax.axhline(y=1.0, color="#333333", linestyle="--", linewidth=1.5, label="Optimal")

    for i, (bar, val) in enumerate(zip(bars, ratios, strict=False)):
        fontweight = "bold" if "LA-PD" in names[i] or "Ours" in names[i] else "normal"
        ax.annotate(
            f"{val:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight=fontweight,
        )

    ax.set_ylabel("Competitive Ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0.95, max(ratios) * 1.08 if ratios else 1.5)
    ax.legend(loc="upper right", fontsize=9)
    add_light_grid(ax, "y")
    ax.set_title("(b) Competitive Ratio", fontsize=11)

    plt.tight_layout()
    output_path = output_dir / f"online_stage2_combined_{sc_config}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Summary
# =============================================================================


def print_summary(data: dict, sc_config: str):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print(f"Online Stage 2 Results Summary (S_C: {sc_config})")
    print("=" * 70)

    config = data.get("experiment_config", {})
    strategies = data.get("strategies", data)

    if config:
        print(f"\nDataset: {config.get('dataset', 'unknown')}")
        print(f"Total requests: {config.get('total_requests', 'N/A')}")
        print(f"Days: {config.get('num_days', 'N/A')}")
        print(f"Daily quota (S_Q): {config.get('daily_quota', 'N/A')}")
        print(f"Concurrency (S_C): {config.get('concurrency_limit', 'N/A')}")
        print(f"S_C monthly fee: ${config.get('sc_monthly_fee', 0):.2f}")

    all_api_cost = strategies["All-API"]["costs"]["total"]

    print(f"\nAll-API Baseline Cost: ${all_api_cost:.2f}")
    print("-" * 70)
    print(f"{'Strategy':<25} {'Total':>10} {'API':>10} {'Ratio':>8} {'SubReqs':>10}")
    print("-" * 70)

    for key in ["All-API", *STRATEGY_ORDER]:
        if key in strategies:
            s = strategies[key]
            total = s["costs"]["total"]
            api = s["costs"]["api"]
            ratio = s.get("competitive_ratio", "-")
            sub_reqs = s["requests"]["subscription"]
            ratio_str = f"{ratio:.4f}" if isinstance(ratio, int | float) else ratio
            print(
                f"{STRATEGY_LABELS.get(key, key):<25} ${total:>9.2f} ${api:>9.2f} {ratio_str:>8} {sub_reqs:>10}"
            )

    print("=" * 70)


# =============================================================================
# Main Entry Points
# =============================================================================


def generate_cost_figures(results_dir: Path, output_dir: Path, sc_config: str):
    """Generate cost-related figures for a specific S_C config."""
    results_path = results_dir / f"stage2_online_results_{sc_config}.json"
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return

    data = load_results(results_path)
    print_summary(data, sc_config)

    plot_cost_comparison(data, output_dir, sc_config)
    plot_competitive_ratio(data, output_dir, sc_config)
    plot_request_distribution(data, output_dir, sc_config)
    plot_combined_figure(data, output_dir, sc_config)


def generate_sensitivity_figures(results_dir: Path, output_dir: Path):
    """Generate sensitivity analysis figures."""
    all_configs = load_all_sc_configs(results_dir)
    if not all_configs:
        print("Error: No S_C config results found")
        return

    plot_sc_sensitivity(all_configs, output_dir)


def main():
    """Generate Online Stage 2 figures."""
    parser = argparse.ArgumentParser(description="Generate Online Stage 2 figures")
    parser.add_argument("--all", action="store_true", help="Generate all figures")
    parser.add_argument("--cost", action="store_true", help="Generate cost figures")
    parser.add_argument("--sensitivity", action="store_true", help="Generate sensitivity figures")
    parser.add_argument(
        "--sc-config",
        type=str,
        choices=["local_gpu", "featherless_premium", "featherless_scale", "all"],
        default="local_gpu",
        help="S_C configuration (default: local_gpu, use 'all' for all configs)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiment/results/online"),
        help="Results directory",
    )
    parser.add_argument("--output-dir", type=Path, help="Output directory for plots")
    args = parser.parse_args()

    apply_style("paper")

    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default to --all if no specific flag provided
    if not any([args.cost, args.sensitivity]):
        args.all = True

    print("Online Stage 2 Plotting")
    print(f"Results dir: {results_dir}")
    print(f"Output dir: {output_dir}")
    print()

    sc_configs = (
        ["local_gpu", "featherless_premium", "featherless_scale"]
        if args.sc_config == "all"
        else [args.sc_config]
    )

    if args.all or args.cost:
        print("=" * 50)
        print("Generating Cost Figures")
        print("=" * 50)
        for sc_config in sc_configs:
            generate_cost_figures(results_dir, output_dir, sc_config)

    if args.all or args.sensitivity:
        print("\n" + "=" * 50)
        print("Generating Sensitivity Figures")
        print("=" * 50)
        generate_sensitivity_figures(results_dir, output_dir)

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
