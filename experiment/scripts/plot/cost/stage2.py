#!/usr/bin/env python3
"""Unified plotting script for Stage 2 experiments (ICML style).

Stage 2 focuses on third-party subscription providers with:
- Daily quota constraints
- Concurrent request limits (e.g., Chutes, Featherless)

Figures generated:
- Offline: cost savings, request distribution, combined view
- Sensitivity: quota/concurrency parameter analysis

Usage:
    python plot_stage2.py --all                 # Generate all figures
    python plot_stage2.py --offline             # Offline results only
    python plot_stage2.py --sensitivity         # Sensitivity analysis only
    python plot_stage2.py --dataset sharegpt    # Specific dataset
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiment.scripts.plot.common import add_light_grid, apply_style

# =============================================================================
# Style Configuration (consistent with plot_stage1.py)
# =============================================================================

COLORS = {
    "all_api": "#8C8C8C",  # Gray (baseline)
    "ilp_optimal": "#DD8452",  # Orange (our method, highlight)
    "daily_quota_only": "#4C72B0",  # Blue
    "quota_only": "#4C72B0",  # Alias
    "concurrency_only": "#55A868",  # Green
    "greedy_online": "#C44E52",  # Red
    "subscription": "#4C72B0",  # Blue
    "api": "#DD8452",  # Orange
    # Model size colors
    "small": "#55A868",
    "medium": "#4C72B0",
    "large": "#C44E52",
}

STRATEGY_NAMES = {
    "ilp_optimal": "ILP-Optimal",
    "daily_quota_only": "Quota-Only",
    "quota_only": "Quota-Only",
    "concurrency_only": "Concurrency-Only",
    "greedy_online": "Greedy-Online",
}

STRATEGY_ORDER = ["daily_quota_only", "concurrency_only", "greedy_online", "ilp_optimal"]


# =============================================================================
# Data Loading
# =============================================================================


def load_results(results_path: Path) -> dict[str, Any]:
    """Load results from JSON file."""
    with open(results_path) as f:
        return json.load(f)


def get_all_api_cost(results: dict, results_dir: Path, dataset: str) -> float:
    """Get All-API baseline cost from Stage 1 or Stage 2 data."""
    # Try loading from Stage 1 results
    stage1_path = results_dir.parent / "stage1" / f"{dataset}_results.json"
    if stage1_path.exists():
        with open(stage1_path) as f:
            stage1_data = json.load(f)
            if "fig1_cost_comparison" in stage1_data:
                return stage1_data["fig1_cost_comparison"]["all_api"]["costs"]["api"]

    # Estimate from highest cost in Stage 2 data
    return max(r["costs"]["api"] for r in results.values() if "costs" in r)


# =============================================================================
# Offline Figures
# =============================================================================


def plot_cost_savings(results: dict, all_api_cost: float, output_dir: Path, dataset: str):
    """Plot API cost savings relative to All-API baseline."""
    fig, ax = plt.subplots(figsize=(6, 4))

    strategies = []
    savings = []
    colors = []

    for key in STRATEGY_ORDER:
        if key in results:
            api_cost = results[key]["costs"]["api"]
            saving_pct = (all_api_cost - api_cost) / all_api_cost * 100
            strategies.append(STRATEGY_NAMES.get(key, key))
            savings.append(saving_pct)
            colors.append(COLORS.get(key, "#666666"))

    x = np.arange(len(strategies))
    bars = ax.bar(x, savings, color=colors, edgecolor="white", linewidth=0.5)

    for i, (bar, val) in enumerate(zip(bars, savings, strict=False)):
        fontweight = "bold" if strategies[i] == "ILP-Optimal" else "normal"
        ax.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight=fontweight,
        )

    ax.set_ylabel("API Cost Savings (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=15, ha="right")
    ax.set_ylim(0, max(savings) * 1.15 if savings else 1)
    add_light_grid(ax, "y")

    plt.tight_layout()
    output_path = output_dir / f"cost_savings_{dataset}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_request_distribution(results: dict, output_dir: Path, dataset: str):
    """Plot request routing distribution as stacked bar chart."""
    fig, ax = plt.subplots(figsize=(6, 4))

    strategies = []
    sub_counts = []
    api_counts = []

    for key in STRATEGY_ORDER:
        if key in results:
            strategies.append(STRATEGY_NAMES.get(key, key))
            sub_counts.append(results[key]["requests"]["subscription"])
            api_counts.append(results[key]["requests"]["api"])

    x = np.arange(len(strategies))
    width = 0.6

    totals = [s + a for s, a in zip(sub_counts, api_counts, strict=False)]
    sub_pcts = [s / t * 100 if t > 0 else 0 for s, t in zip(sub_counts, totals, strict=False)]

    bars1 = ax.bar(
        x,
        sub_counts,
        width,
        label="Subscription",
        color=COLORS["subscription"],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.bar(
        x,
        api_counts,
        width,
        bottom=sub_counts,
        label="API",
        color=COLORS["api"],
        edgecolor="white",
        linewidth=0.5,
    )

    for bar, pct in zip(bars1, sub_pcts, strict=False):
        if pct > 5:
            ax.annotate(
                f"{pct:.0f}%",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
                ha="center",
                va="center",
                fontsize=10,
                color="white",
                fontweight="bold",
            )

    ax.set_ylabel("Number of Requests")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=15, ha="right")
    ax.legend(loc="upper right")
    add_light_grid(ax, "y")

    plt.tight_layout()
    output_path = output_dir / f"request_distribution_{dataset}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_combined_figure(results: dict, all_api_cost: float, output_dir: Path, dataset: str):
    """Create combined figure with cost savings and request distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # (a) Cost Savings
    ax = axes[0]
    strategies = []
    savings = []
    colors = []

    for key in STRATEGY_ORDER:
        if key in results:
            api_cost = results[key]["costs"]["api"]
            saving_pct = (all_api_cost - api_cost) / all_api_cost * 100
            strategies.append(STRATEGY_NAMES.get(key, key))
            savings.append(saving_pct)
            colors.append(COLORS.get(key, "#666666"))

    x = np.arange(len(strategies))
    bars = ax.bar(x, savings, color=colors, edgecolor="white", linewidth=0.5)

    for i, (bar, val) in enumerate(zip(bars, savings, strict=False)):
        fontweight = "bold" if strategies[i] == "ILP-Optimal" else "normal"
        ax.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight=fontweight,
        )

    ax.set_ylabel("API Cost Savings (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=15, ha="right")
    ax.set_ylim(0, max(savings) * 1.18 if savings else 1)
    add_light_grid(ax, "y")
    ax.set_title("(a) Cost Savings vs All-API", fontsize=12)

    # (b) Request Distribution
    ax = axes[1]
    strategies = []
    sub_counts = []
    api_counts = []

    for key in STRATEGY_ORDER:
        if key in results:
            strategies.append(STRATEGY_NAMES.get(key, key))
            sub_counts.append(results[key]["requests"]["subscription"])
            api_counts.append(results[key]["requests"]["api"])

    x = np.arange(len(strategies))
    width = 0.6

    totals = [s + a for s, a in zip(sub_counts, api_counts, strict=False)]
    sub_pcts = [s / t * 100 if t > 0 else 0 for s, t in zip(sub_counts, totals, strict=False)]

    bars1 = ax.bar(
        x,
        sub_counts,
        width,
        label="Subscription",
        color=COLORS["subscription"],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.bar(
        x,
        api_counts,
        width,
        bottom=sub_counts,
        label="API",
        color=COLORS["api"],
        edgecolor="white",
        linewidth=0.5,
    )

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
    ax.set_xticklabels(strategies, rotation=15, ha="right")
    ax.legend(loc="upper right", fontsize=9)
    add_light_grid(ax, "y")
    ax.set_title("(b) Request Routing Distribution", fontsize=12)

    plt.tight_layout()
    output_path = output_dir / f"combined_{dataset}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_api_cost_comparison(results: dict, all_api_cost: float, output_dir: Path, dataset: str):
    """Plot API cost comparison bar chart."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    strategies = ["All-API"]
    api_costs = [all_api_cost]
    colors = [COLORS["all_api"]]

    for key in STRATEGY_ORDER:
        if key in results:
            strategies.append(STRATEGY_NAMES.get(key, key))
            api_costs.append(results[key]["costs"]["api"])
            colors.append(COLORS.get(key, "#666666"))

    x = np.arange(len(strategies))
    bars = ax.bar(x, api_costs, color=colors, edgecolor="white", linewidth=0.5)

    for i, (bar, val) in enumerate(zip(bars, api_costs, strict=False)):
        fontweight = "bold" if strategies[i] == "ILP-Optimal" else "normal"
        ax.annotate(
            f"${val:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight=fontweight,
        )

    ax.set_ylabel("API Cost ($)")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=15, ha="right")
    ax.set_ylim(0, max(api_costs) * 1.12)
    add_light_grid(ax, "y")

    plt.tight_layout()
    output_path = output_dir / f"api_cost_{dataset}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Sensitivity Analysis
# =============================================================================


def plot_quota_sensitivity(results_dir: Path, output_dir: Path, dataset: str):
    """Plot sensitivity to daily quota parameter."""
    sensitivity_path = results_dir / f"{dataset}_quota_sensitivity.json"
    if not sensitivity_path.exists():
        print(f"Skipping quota sensitivity: {sensitivity_path} not found")
        return

    with open(sensitivity_path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(6, 4))

    quotas = data["quotas"]
    savings = data["ilp_savings"]

    ax.plot(quotas, savings, "o-", color=COLORS["ilp_optimal"], linewidth=2, markersize=8)
    ax.set_xlabel("Daily Quota")
    ax.set_ylabel("API Cost Savings (%)")
    add_light_grid(ax, "y")

    plt.tight_layout()
    output_path = output_dir / f"quota_sensitivity_{dataset}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_concurrency_sensitivity(results_dir: Path, output_dir: Path, dataset: str):
    """Plot sensitivity to concurrency limit parameter."""
    sensitivity_path = results_dir / f"{dataset}_concurrency_sensitivity.json"
    if not sensitivity_path.exists():
        print(f"Skipping concurrency sensitivity: {sensitivity_path} not found")
        return

    with open(sensitivity_path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(6, 4))

    concurrency_limits = data["concurrency_limits"]
    savings = data["ilp_savings"]

    ax.plot(
        concurrency_limits, savings, "s-", color=COLORS["ilp_optimal"], linewidth=2, markersize=8
    )
    ax.set_xlabel("Concurrency Limit")
    ax.set_ylabel("API Cost Savings (%)")
    add_light_grid(ax, "y")

    plt.tight_layout()
    output_path = output_dir / f"concurrency_sensitivity_{dataset}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Summary
# =============================================================================


def print_summary(results: dict, all_api_cost: float, dataset: str):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print(f"Stage 2 Results Summary ({dataset})")
    print("=" * 60)

    print(f"\nAll-API Baseline Cost: ${all_api_cost:.2f}")
    print("-" * 60)
    print(f"{'Strategy':<20} {'API Cost':>12} {'Savings':>10} {'Sub Reqs':>10}")
    print("-" * 60)

    for key in STRATEGY_ORDER:
        if key in results:
            data = results[key]
            api_cost = data["costs"]["api"]
            saving = (all_api_cost - api_cost) / all_api_cost * 100
            sub_reqs = data["requests"]["subscription"]
            print(
                f"{STRATEGY_NAMES.get(key, key):<20} ${api_cost:>10.2f} {saving:>9.1f}% {sub_reqs:>10}"
            )

    print("=" * 60)


# =============================================================================
# Main Entry Points
# =============================================================================


def generate_offline_figures(results_dir: Path, output_dir: Path, dataset: str):
    """Generate all offline figures."""
    results_path = results_dir / f"{dataset}_results.json"
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return

    results = load_results(results_path)
    all_api_cost = get_all_api_cost(results, results_dir, dataset)

    print_summary(results, all_api_cost, dataset)

    plot_cost_savings(results, all_api_cost, output_dir, dataset)
    plot_request_distribution(results, output_dir, dataset)
    plot_api_cost_comparison(results, all_api_cost, output_dir, dataset)
    plot_combined_figure(results, all_api_cost, output_dir, dataset)


# =============================================================================
# Single-Model Comparison
# =============================================================================


def plot_single_model_comparison(data: dict, output_dir: Path, dataset: str):
    """Plot single-model comparison across different model sizes."""
    models = list(data.keys())
    model_labels = {
        "llama-4-scout": "Small\n(mult=1)",
        "qwen3-coder-30b": "Medium\n(mult=2)",
        "llama-3.3-70b-instruct": "Large\n(mult=4)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x = np.arange(len(models))
    width = 0.25

    # (a) API Cost by Strategy
    ax = axes[0]
    quota_costs = [data[m]["results"]["daily_quota_only"]["costs"]["api"] for m in models]
    conc_costs = [data[m]["results"]["concurrency_only"]["costs"]["api"] for m in models]
    ilp_costs = [data[m]["results"]["ilp_optimal"]["costs"]["api"] for m in models]

    ax.bar(x - width, quota_costs, width, label="Quota-Only", color=COLORS["daily_quota_only"])
    ax.bar(x, conc_costs, width, label="Conc.-Only", color=COLORS["concurrency_only"])
    ax.bar(x + width, ilp_costs, width, label="ILP-Optimal", color=COLORS["ilp_optimal"])
    ax.set_ylabel("API Cost ($)")
    ax.set_xticks(x)
    ax.set_xticklabels([model_labels.get(m, m) for m in models])
    ax.legend(loc="upper left", fontsize=9)
    add_light_grid(ax, "y")
    ax.set_title("(a) API Cost by Model Size", fontsize=12)

    # (b) Subscription Requests
    ax = axes[1]
    quota_reqs = [
        data[m]["results"]["daily_quota_only"]["requests"]["subscription"] for m in models
    ]
    conc_reqs = [data[m]["results"]["concurrency_only"]["requests"]["subscription"] for m in models]
    ilp_reqs = [data[m]["results"]["ilp_optimal"]["requests"]["subscription"] for m in models]

    ax.bar(x - width, quota_reqs, width, label="Quota-Only", color=COLORS["daily_quota_only"])
    ax.bar(x, conc_reqs, width, label="Conc.-Only", color=COLORS["concurrency_only"])
    ax.bar(x + width, ilp_reqs, width, label="ILP-Optimal", color=COLORS["ilp_optimal"])
    ax.set_ylabel("Subscription Requests")
    ax.set_xticks(x)
    ax.set_xticklabels([model_labels.get(m, m) for m in models])
    ax.legend(loc="upper right", fontsize=9)
    add_light_grid(ax, "y")
    ax.set_title("(b) Subscription Utilization", fontsize=12)

    # (c) Effective Concurrency Impact
    ax = axes[2]
    eff_conc = [data[m]["effective_concurrency"] for m in models]
    conc_ratio = [c / q if q > 0 else 0 for c, q in zip(conc_reqs, quota_reqs, strict=False)]

    bars = ax.bar(x, conc_ratio, color=[COLORS["small"], COLORS["medium"], COLORS["large"]])
    ax.axhline(y=1.0, color="#666666", linestyle="--", alpha=0.7, label="Equal to Quota")
    for bar, eff_c in zip(bars, eff_conc, strict=False):
        ax.annotate(
            f"C={eff_c}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Conc. Reqs / Quota Reqs")
    ax.set_xticks(x)
    ax.set_xticklabels([model_labels.get(m, m) for m in models])
    ax.legend(loc="upper right", fontsize=9)
    add_light_grid(ax, "y")
    ax.set_title("(c) Concurrency vs Quota Ratio", fontsize=12)

    plt.tight_layout()
    output_path = output_dir / f"single_model_comparison_{dataset}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def generate_sensitivity_figures(results_dir: Path, output_dir: Path, dataset: str):
    """Generate sensitivity analysis figures."""
    plot_quota_sensitivity(results_dir, output_dir, dataset)
    plot_concurrency_sensitivity(results_dir, output_dir, dataset)


def generate_single_model_figures(results_dir: Path, output_dir: Path, dataset: str):
    """Generate single-model comparison figures."""
    single_model_path = results_dir / f"single_model_comparison_{dataset}.json"
    if not single_model_path.exists():
        print(f"Skipping single-model: {single_model_path} not found")
        return
    with open(single_model_path) as f:
        data = json.load(f)
    plot_single_model_comparison(data, output_dir, dataset)


def main():
    """Generate Stage 2 figures."""
    parser = argparse.ArgumentParser(description="Generate Stage 2 figures")
    parser.add_argument("--all", action="store_true", help="Generate all figures")
    parser.add_argument("--offline", action="store_true", help="Generate offline figures")
    parser.add_argument("--sensitivity", action="store_true", help="Generate sensitivity figures")
    parser.add_argument("--single-model", action="store_true", help="Generate single-model figures")
    parser.add_argument("--dataset", default="rednote", help="Dataset name (default: rednote)")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiment/results/stage2"),
        help="Results directory",
    )
    parser.add_argument("--output-dir", type=Path, help="Output directory for plots")
    args = parser.parse_args()

    apply_style("paper")

    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default to --all if no specific flag provided
    if not any([args.offline, args.sensitivity, args.single_model]):
        args.all = True

    print(f"Stage 2 Plotting - Dataset: {args.dataset}")
    print(f"Results dir: {results_dir}")
    print(f"Output dir: {output_dir}")
    print()

    if args.all or args.offline:
        print("=" * 40)
        print("Generating Offline Figures")
        print("=" * 40)
        generate_offline_figures(results_dir, output_dir, args.dataset)

    if args.all or args.sensitivity:
        print("\n" + "=" * 40)
        print("Generating Sensitivity Figures")
        print("=" * 40)
        generate_sensitivity_figures(results_dir, output_dir, args.dataset)

    if args.all or args.single_model:
        print("\n" + "=" * 40)
        print("Generating Single-Model Figures")
        print("=" * 40)
        generate_single_model_figures(results_dir, output_dir, args.dataset)

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
