#!/usr/bin/env python3
"""Stage 1 Plot Generator - All figures for ICML paper.

This script generates all Stage 1 figures:
1. Offline results (cost savings, competitive ratio, sensitivity)
2. Multi-model analysis (cost distribution, quota allocation)
3. Online results (cost vs quota for different strategies)

Usage:
    python plot_stage1.py                    # Generate all figures
    python plot_stage1.py --offline          # Only offline figures
    python plot_stage1.py --multimodel       # Only multi-model figures
    python plot_stage1.py --online           # Only online figures
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiment.scripts.plot.common import add_light_grid, apply_style

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Style Configuration
# =============================================================================

COLORS = {
    "all_api": "#8C8C8C",
    "greedy": "#4C72B0",
    "optimal": "#DD8452",
    "accent": "#C44E52",
    "subscription": "#55A868",
    "light_gray": "#E5E5E5",
}

DATASET_NAMES = {
    "sharegpt": "ShareGPT",
    "freeinference": "FreeInference",
    "rednote": "RedNote",
    "burstgpt": "BurstGPT",
}

VALID_DATASETS = ["sharegpt", "freeinference", "rednote"]


# =============================================================================
# Data Loading
# =============================================================================


def load_results(results_dir: Path) -> dict[str, Any]:
    """Load all result JSON files."""
    results = {}
    for f in results_dir.glob("*_results.json"):
        name = f.stem.replace("_results", "")
        if name in VALID_DATASETS:
            with open(f) as fp:
                results[name] = json.load(fp)
    return results


def load_online_results(results_dir: Path) -> dict[str, Any]:
    """Load online experiment results."""
    online_dir = results_dir.parent / "online"
    results = {}
    for f in online_dir.glob("*_results.json"):
        name = f.stem.replace("_results", "")
        with open(f) as fp:
            results[name] = json.load(fp)
    return results


# =============================================================================
# Section 1: Offline Figures
# =============================================================================


def plot_cost_savings(results: dict, output_dir: Path):
    """Plot cost savings comparison (main result)."""
    datasets = [d for d in VALID_DATASETS if d in results]

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(datasets))
    width = 0.35

    greedy_savings, optimal_savings = [], []
    for d in datasets:
        data = results[d]["fig1_cost_comparison"]
        all_api = data["all_api"]["costs"]["api"]
        greedy = data["greedy"]["costs"]["api"]
        optimal = data["optimal"]["costs"]["api"]
        greedy_savings.append((all_api - greedy) / all_api * 100)
        optimal_savings.append((all_api - optimal) / all_api * 100)

    bars1 = ax.bar(
        x - width / 2,
        greedy_savings,
        width,
        label="Greedy",
        color=COLORS["greedy"],
        edgecolor="white",
    )
    bars2 = ax.bar(
        x + width / 2,
        optimal_savings,
        width,
        label="Optimal",
        color=COLORS["optimal"],
        edgecolor="white",
    )

    for bar, val in zip(bars1, greedy_savings, strict=False):
        ax.annotate(
            f"{val:.0f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )
    for bar, val in zip(bars2, optimal_savings, strict=False):
        ax.annotate(
            f"{val:.0f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("API Cost Savings (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_NAMES.get(d, d) for d in datasets])
    ax.set_ylim(0, max(optimal_savings) * 1.15)
    ax.legend(loc="upper left")
    add_light_grid(ax)

    plt.tight_layout()
    plt.savefig(output_dir / "cost_savings.png")
    plt.close()
    logger.info("Saved: cost_savings.png")


def plot_competitive_ratio(results: dict, output_dir: Path):
    """Plot competitive ratio vs quota (legacy)."""
    datasets = [d for d in VALID_DATASETS if d in results and d != "rednote"]

    fig, ax = plt.subplots(figsize=(5, 4))
    markers = ["o", "s", "^"]

    for i, d in enumerate(datasets):
        data = results[d]["fig3_competitive_ratio"]
        ax.plot(
            data["q_values"],
            data["competitive_ratios"],
            marker=markers[i],
            linestyle="-",
            linewidth=1.5,
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.5,
            label=DATASET_NAMES.get(d, d),
        )

    ax.axhline(y=1.0, color=COLORS["all_api"], linestyle="--", alpha=0.7)
    ax.set_xlabel("Daily Quota (Q)")
    ax.set_ylabel("Competitive Ratio (Greedy / Optimal)")
    ax.legend(loc="upper left")
    add_light_grid(ax)

    plt.tight_layout()
    plt.savefig(output_dir / "competitive_ratio.png")
    plt.close()
    logger.info("Saved: competitive_ratio.png")


def plot_savings_vs_quota(results: dict, output_dir: Path):
    """Plot Optimal, Greedy, and Primal-Dual savings (%) vs quota size.

    Uses orthogonal visual encoding:
      - Color = dataset (blue for ShareGPT, red for FreeInference)
      - Line style + marker = algorithm (solid/o for Optimal, dashed/x for Greedy,
        dash-dot/s for Primal-Dual)
    """
    # Map internal dataset names to online comparison keys
    online_key_map = {
        "sharegpt": "BurstGPT (single-model)",
        "freeinference": "Freeinference (multi-model)",
    }
    datasets = [d for d in VALID_DATASETS if d in results and d != "rednote"]

    # Load online comparison data for Primal-Dual points
    online_comp_path = Path("experiment/results/online/stage1_comparison.json")
    online_comp = {}
    if online_comp_path.exists():
        with open(online_comp_path) as f:
            online_comp = json.load(f)

    fig, ax = plt.subplots(figsize=(6, 4))
    dataset_colors = ["#4C72B0", "#C44E52"]

    for i, d in enumerate(datasets):
        data = results[d]["fig2_parameter_sensitivity"]
        q_values = data["q_values"]
        all_api = data["all_api"][0]
        greedy_savings = [(all_api - g) / all_api * 100 for g in data["greedy"]]
        optimal_savings = [(all_api - o) / all_api * 100 for o in data["optimal"]]

        label = DATASET_NAMES.get(d, d)
        color = dataset_colors[i]

        # Optimal: solid line, circle marker
        ax.plot(
            q_values, optimal_savings,
            marker="o", linestyle="-", linewidth=1.5, markersize=5,
            markerfacecolor="white", markeredgewidth=1.5,
            color=color, label=f"{label} (Optimal)",
        )
        # Greedy: dashed line, x marker
        ax.plot(
            q_values, greedy_savings,
            marker="x", linestyle="--", linewidth=1.5, markersize=5,
            markeredgewidth=1.5,
            color=color, alpha=0.6, label=f"{label} (Greedy)",
        )

        # Shaded gap between Greedy and Optimal
        ax.fill_between(
            q_values, greedy_savings, optimal_savings,
            alpha=0.08, color=color,
        )

        # Primal-Dual: dash-dot line, square marker (from online comparison)
        online_key = online_key_map.get(d, "")
        if online_key in online_comp:
            plans = online_comp[online_key]["plans"]
            online_all_api = online_comp[online_key]["all_api"]
            pd_q = [v["daily_quota"] for v in plans.values()]
            pd_savings = [(online_all_api - v["primal_dual"]) / online_all_api * 100
                          for v in plans.values()]
            ax.plot(
                pd_q, pd_savings,
                marker="s", linestyle="-.", linewidth=1.5, markersize=6,
                markerfacecolor="white", markeredgewidth=1.5,
                color=color, label=f"{label} (Ours)",
            )

    ax.set_xlabel("Daily Quota (Q)")
    ax.set_ylabel("API Cost Savings (%)")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    add_light_grid(ax)

    plt.tight_layout()
    plt.savefig(output_dir / "savings_vs_quota.png")
    plt.close()
    logger.info("Saved: savings_vs_quota.png")


def plot_sensitivity(results: dict, output_dir: Path, dataset: str):
    """Plot sensitivity analysis for a dataset."""
    if dataset not in results:
        return

    data = results[dataset]
    fig2 = data["fig2_parameter_sensitivity"]
    q_values = fig2["q_values"]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.axhline(
        y=fig2["all_api"][0],
        color=COLORS["all_api"],
        linestyle="--",
        linewidth=1.5,
        label="All-API (baseline)",
    )
    ax.plot(
        q_values,
        fig2["greedy"],
        "s-",
        color=COLORS["greedy"],
        linewidth=1.5,
        markersize=5,
        markerfacecolor="white",
        label="Greedy",
    )
    ax.plot(
        q_values,
        fig2["optimal"],
        "o-",
        color=COLORS["optimal"],
        linewidth=1.5,
        markersize=5,
        markerfacecolor="white",
        label="Optimal",
    )
    ax.fill_between(q_values, fig2["greedy"], fig2["optimal"], alpha=0.15, color=COLORS["accent"], label="Opportunity Cost")

    ax.set_xlabel("Daily Quota (Q)")
    ax.set_ylabel("API Cost ($)")
    ax.legend(loc="upper right")
    add_light_grid(ax)

    plt.tight_layout()
    plt.savefig(output_dir / f"fig2_sensitivity_{dataset}.png")
    plt.close()
    logger.info(f"Saved: fig2_sensitivity_{dataset}.png")


def generate_offline_figures(results_dir: Path, output_dir: Path):
    """Generate all offline figures."""
    logger.info("=== Generating Offline Figures ===")
    results = load_results(results_dir)
    logger.info(f"Loaded {len(results)} datasets: {list(results.keys())}")

    plot_cost_savings(results, output_dir)
    plot_competitive_ratio(results, output_dir)
    plot_savings_vs_quota(results, output_dir)
    plot_sensitivity(results, output_dir, "sharegpt")
    plot_sensitivity(results, output_dir, "freeinference")


# =============================================================================
# Section 2: Multi-Model Figures
# =============================================================================


def load_multimodel_data(dataset_name: str = "rednote"):
    """Load data for multi-model analysis."""
    from experiment.config import ExperimentConfig
    from experiment.data.loader import DataLoader

    config = ExperimentConfig("config/experiment.yaml").to_dict()
    loader = DataLoader(config)
    requests = loader.load(f"data/{dataset_name}_logs.csv")
    return requests, config


def calculate_model_stats(requests, config):
    """Calculate per-model statistics."""
    model_pricing = config.get("model_pricing", {})
    stats = defaultdict(lambda: {"count": 0, "total_cost": 0.0, "costs": []})

    for r in requests:
        model = r.model or "unknown"
        pricing = model_pricing.get(model, {"input": 0, "output": 0})
        cost = r.request_tokens / 1e6 * pricing.get(
            "input", 0
        ) + r.response_tokens / 1e6 * pricing.get("output", 0)
        stats[model]["count"] += 1
        stats[model]["total_cost"] += cost
        stats[model]["costs"].append(cost)

    for model in stats:
        stats[model]["avg_cost"] = stats[model]["total_cost"] / stats[model]["count"]

    return dict(stats)


def simulate_allocation(requests, config, daily_quota=5000):
    """Simulate Greedy and Optimal allocation."""
    model_pricing = config.get("model_pricing", {})
    days = defaultdict(list)

    for r in requests:
        days[r.day].append(r)

    greedy_alloc = defaultdict(lambda: {"subscription": 0, "api": 0, "saved": 0.0})
    optimal_alloc = defaultdict(lambda: {"subscription": 0, "api": 0, "saved": 0.0})

    for day_requests in days.values():
        requests_with_cost = []
        for r in day_requests:
            model = r.model or "unknown"
            pricing = model_pricing.get(model, {"input": 0, "output": 0})
            cost = r.request_tokens / 1e6 * pricing.get(
                "input", 0
            ) + r.response_tokens / 1e6 * pricing.get("output", 0)
            requests_with_cost.append((r, cost))

        # Greedy
        quota = daily_quota
        for r, cost in requests_with_cost:
            model = r.model or "unknown"
            if quota > 0:
                greedy_alloc[model]["subscription"] += 1
                greedy_alloc[model]["saved"] += cost
                quota -= 1
            else:
                greedy_alloc[model]["api"] += 1

        # Optimal
        sorted_req = sorted(requests_with_cost, key=lambda x: x[1], reverse=True)
        quota = daily_quota
        for r, cost in sorted_req:
            model = r.model or "unknown"
            if quota > 0:
                optimal_alloc[model]["subscription"] += 1
                optimal_alloc[model]["saved"] += cost
                quota -= 1
            else:
                optimal_alloc[model]["api"] += 1

    return dict(greedy_alloc), dict(optimal_alloc)


def plot_multimodel_combined(stats, greedy_alloc, optimal_alloc, output_dir: Path):
    """Plot combined multi-model analysis figure."""
    sorted_models = sorted(stats.items(), key=lambda x: x[1]["total_cost"], reverse=True)
    top_models = [m for m, _ in sorted_models[:5]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # (a) Cost by model
    ax = axes[0]
    costs = [stats[m]["total_cost"] for m in top_models]
    colors = plt.cm.Set2(np.linspace(0, 1, len(top_models)))
    bars = ax.barh(top_models, costs, color=colors)
    ax.set_xlabel("Total API Cost ($)")
    ax.set_title("(a) API Cost by Model")
    ax.invert_yaxis()
    for bar, cost in zip(bars, costs, strict=False):
        ax.annotate(
            f"${cost:.0f}",
            xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(3, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
        )

    # (b) Quota allocation
    ax = axes[1]
    models_by_cost = sorted(top_models, key=lambda m: stats[m]["avg_cost"], reverse=True)
    x = np.arange(len(models_by_cost))
    width = 0.35
    greedy_sub = [greedy_alloc.get(m, {}).get("subscription", 0) for m in models_by_cost]
    optimal_sub = [optimal_alloc.get(m, {}).get("subscription", 0) for m in models_by_cost]
    ax.bar(x - width / 2, greedy_sub, width, label="Greedy", color=COLORS["greedy"])
    ax.bar(x + width / 2, optimal_sub, width, label="Optimal", color=COLORS["optimal"])
    ax.set_ylabel("Subscription Requests")
    ax.set_xticks(x)
    ax.set_xticklabels(models_by_cost, rotation=30, ha="right", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("(b) Quota Allocation")
    add_light_grid(ax)

    # (c) Savings
    ax = axes[2]
    greedy_saved = [greedy_alloc.get(m, {}).get("saved", 0) for m in models_by_cost]
    optimal_saved = [optimal_alloc.get(m, {}).get("saved", 0) for m in models_by_cost]
    ax.bar(x - width / 2, greedy_saved, width, label="Greedy", color=COLORS["greedy"])
    ax.bar(x + width / 2, optimal_saved, width, label="Optimal", color=COLORS["optimal"])
    ax.set_ylabel("API Cost Saved ($)")
    ax.set_xticks(x)
    ax.set_xticklabels(models_by_cost, rotation=30, ha="right", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("(c) Per-Model Savings")
    add_light_grid(ax)

    plt.tight_layout()
    plt.savefig(output_dir / "multimodel_combined.png")
    plt.close()
    logger.info("Saved: multimodel_combined.png")


def plot_cost_distribution(stats, output_dir: Path):
    """Plot model cost distribution."""
    sorted_models = sorted(stats.items(), key=lambda x: x[1]["total_cost"], reverse=True)
    top_n = 6
    top_models = sorted_models[:top_n]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Pie chart
    ax = axes[0]
    labels = [m for m, _ in top_models]
    costs = [s["total_cost"] for _, s in top_models]
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    ax.pie(
        costs,
        labels=labels,
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        colors=colors,
        startangle=90,
    )
    ax.set_title("(a) API Cost Share by Model")

    # Bar chart
    ax = axes[1]
    avg_costs = [s["avg_cost"] * 1000 for _, s in top_models]
    counts = [s["count"] for _, s in top_models]
    x = np.arange(len(labels))
    bars = ax.bar(x, avg_costs, color=colors)
    for bar, count in zip(bars, counts, strict=False):
        ax.annotate(
            f"n={count:,}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set_ylabel("Avg Cost per Request (m$)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("(b) Average Request Cost by Model")
    add_light_grid(ax)

    plt.tight_layout()
    plt.savefig(output_dir / "multimodel_cost_distribution.png")
    plt.close()
    logger.info("Saved: multimodel_cost_distribution.png")


def generate_multimodel_figures(output_dir: Path):
    """Generate all multi-model figures."""
    logger.info("=== Generating Multi-Model Figures ===")

    try:
        requests, config = load_multimodel_data("rednote")
        logger.info(f"Loaded {len(requests):,} requests")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    stats = calculate_model_stats(requests, config)
    greedy_alloc, optimal_alloc = simulate_allocation(requests, config)

    plot_cost_distribution(stats, output_dir)
    plot_multimodel_combined(stats, greedy_alloc, optimal_alloc, output_dir)


# =============================================================================
# Section 3: Online Figures
# =============================================================================


def plot_online_results(output_dir: Path):
    """Plot online experiment results."""
    logger.info("=== Generating Online Figures ===")

    online_dir = Path("experiment/results/online")
    if not online_dir.exists():
        logger.warning("Online results directory not found")
        return

    # Load online results
    for dataset in ["burstgpt", "freeinference"]:
        results_file = online_dir / f"{dataset}_results.json"
        if not results_file.exists():
            continue

        with open(results_file) as f:
            data = json.load(f)

        fig, ax = plt.subplots(figsize=(6, 4.5))

        plans = list(data.keys())
        x_labels = []
        greedy_costs, pd_costs, optimal_costs = [], [], []

        for plan in plans:
            plan_data = data[plan]
            x_labels.append(
                f"{plan_data.get('quota', 'N/A')}\n({plan_data.get('plan_name', plan)})"
            )
            greedy_costs.append(plan_data.get("greedy_cost", 0))
            pd_costs.append(plan_data.get("primal_dual_cost", 0))
            optimal_costs.append(plan_data.get("optimal_cost", 0))

        x = np.arange(len(plans))

        ax.plot(
            x, greedy_costs, "o-", color=COLORS["greedy"], label="Greedy", markersize=8, linewidth=2
        )
        ax.plot(
            x,
            pd_costs,
            "s-",
            color=COLORS["optimal"],
            label="Ours (Threshold)",
            markersize=8,
            linewidth=2,
        )
        ax.plot(
            x,
            optimal_costs,
            "^-",
            color=COLORS["subscription"],
            label="Optimal",
            markersize=8,
            linewidth=2,
        )

        # All-API baseline
        all_api = data[plans[0]].get("all_api_cost", max(greedy_costs) * 1.1)
        ax.axhline(
            y=all_api, color=COLORS["all_api"], linestyle="--", label=f"All-API (${all_api:.0f})"
        )

        ax.set_xlabel("Daily Quota (Q)")
        ax.set_ylabel("Total Cost ($)")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.legend(loc="upper right")
        ax.set_title(f"{DATASET_NAMES.get(dataset, dataset)}: Cost vs Quota")
        add_light_grid(ax)

        plt.tight_layout()
        plt.savefig(output_dir / f"online_{dataset}.png")
        plt.close()
        logger.info(f"Saved: online_{dataset}.png")


# =============================================================================
# Section 4: Online Ablation Figures (2x2)
# =============================================================================


def plot_online_ablation(output_dir: Path):
    """Plot 2x2 ablation study: Predictor x Decision Rule."""
    logger.info("=== Generating Online Ablation Figure ===")

    online_file = Path("experiment/results/online/stage1_online_results.json")
    if not online_file.exists():
        logger.warning("Online results not found")
        return

    with open(online_file) as f:
        data = json.load(f)

    # Extract costs for 2x2 ablation
    optimal_cost = data["Offline-Optimal"]["costs"]["total"]
    all_api_cost = data["All-API"]["costs"]["total"]

    # 2x2 matrix: [PD, LA] x [EMA, Histogram]
    ablation = {
        "PD-EMA": data.get("PrimalDual-Online", {}).get("costs", {}).get("total", 0),
        "PD-Hist": data.get("PD-Hist", {}).get("costs", {}).get("total", 0),
        "LA-EMA": data.get("LA-EMA", {}).get("costs", {}).get("total", 0),
        "LA-Hist": data.get("LA-PD-P50", {}).get("costs", {}).get("total", 0),
    }

    # Calculate competitive ratios
    ratios = {k: v / optimal_cost for k, v in ablation.items()}

    # Color palette - clean and professional
    BLUE_DARK = "#2C3E50"
    BLUE_MED = "#3498DB"
    BLUE_LIGHT = "#85C1E9"
    ORANGE_DARK = "#E67E22"
    ORANGE_LIGHT = "#F5B041"
    GREEN = "#27AE60"
    GRAY = "#95A5A6"
    RED_SOFT = "#E74C3C"

    # =========================================================================
    # Figure 1: Cost Comparison Bar Chart (Horizontal)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    strategies = ["Optimal", "PD-EMA", "LA-EMA", "PD-Hist", "LA-Hist", "Greedy", "All-API"]
    costs = [
        optimal_cost,
        ablation["PD-EMA"],
        ablation["LA-EMA"],
        ablation["PD-Hist"],
        ablation["LA-Hist"],
        data["Greedy-Online"]["costs"]["total"],
        all_api_cost,
    ]

    colors_list = [GREEN, BLUE_MED, BLUE_LIGHT, ORANGE_DARK, ORANGE_LIGHT, GRAY, RED_SOFT]

    y = np.arange(len(strategies))
    bars = ax.barh(y, costs, color=colors_list, height=0.65, edgecolor="white", linewidth=1.2)

    # Add value labels inside bars
    for bar, cost, _ in zip(bars, costs, strategies, strict=False):
        width = bar.get_width()
        label_x = width - 25 if width > 150 else width + 5
        color = "white" if width > 150 else BLUE_DARK
        ha = "right" if width > 150 else "left"
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"${cost:.0f}",
            va="center",
            ha=ha,
            fontsize=11,
            fontweight="bold",
            color=color,
        )

    # Add competitive ratio annotation
    for i, (strategy, cost) in enumerate(zip(strategies, costs, strict=False)):
        ratio = cost / optimal_cost
        if strategy != "Optimal":
            ax.text(
                cost + 30,
                i,
                f"{ratio:.2f}x",
                va="center",
                ha="left",
                fontsize=9,
                color=GRAY,
                style="italic",
            )

    ax.set_yticks(y)
    ax.set_yticklabels(strategies, fontsize=11)
    ax.set_xlabel("Total Cost ($)", fontsize=12)
    ax.set_xlim(0, all_api_cost * 1.15)
    ax.invert_yaxis()

    # Remove top/right spines, add subtle grid
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)

    # Add legend for strategy groups
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=GREEN, label="Offline Optimal"),
        Patch(facecolor=BLUE_MED, label="EMA Predictor"),
        Patch(facecolor=ORANGE_DARK, label="Histogram Predictor"),
        Patch(facecolor=GRAY, label="Baselines"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.95, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "online_cost_comparison.png", dpi=300)
    plt.close()
    logger.info("Saved: online_cost_comparison.png")

    # =========================================================================
    # Figure 2: 2x2 Ablation Heatmap (Clean Design)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(6, 5))

    matrix = np.array(
        [
            [ratios["PD-EMA"], ratios["PD-Hist"]],
            [ratios["LA-EMA"], ratios["LA-Hist"]],
        ]
    )

    # Custom colormap: green (good) -> yellow -> red (bad)
    from matplotlib.colors import LinearSegmentedColormap

    colors_cmap = ["#27AE60", "#F1C40F", "#E74C3C"]
    cmap = LinearSegmentedColormap.from_list("custom", colors_cmap, N=100)

    im = ax.imshow(matrix, cmap=cmap, vmin=1.0, vmax=1.25, aspect="auto")

    # Style the cells
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["EMA", "Histogram"], fontsize=12, fontweight="bold")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Primal-Dual", "LA-PD"], fontsize=12, fontweight="bold")

    # Add labels
    ax.set_xlabel("Predictor", fontsize=12, labelpad=10)
    ax.set_ylabel("Decision Rule", fontsize=12, labelpad=10)

    # Add text annotations with better styling
    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            # Highlight the best cell
            fontweight = "bold" if val == matrix.min() else "normal"
            fontsize = 16 if val == matrix.min() else 14
            text_color = "white" if val < 1.15 else BLUE_DARK
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight=fontweight,
                color=text_color,
            )

    # Add colorbar with better styling
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Competitive Ratio", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    # Add cell borders
    for i in range(2):
        for j in range(2):
            rect = plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=2
            )
            ax.add_patch(rect)

    # Mark the best cell
    best_i, best_j = np.unravel_index(matrix.argmin(), matrix.shape)
    rect = plt.Rectangle(
        (best_j - 0.5, best_i - 0.5),
        1,
        1,
        fill=False,
        edgecolor=BLUE_DARK,
        linewidth=3,
        linestyle="-",
    )
    ax.add_patch(rect)

    ax.set_title("2x2 Ablation: Predictor vs Decision Rule", fontsize=13, pad=15)

    plt.tight_layout()
    plt.savefig(output_dir / "online_ablation_heatmap.png", dpi=300)
    plt.close()
    logger.info("Saved: online_ablation_heatmap.png")

    # =========================================================================
    # Figure 3: Calibration Comparison (Improved Design)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(7, 5))

    la_stats = data.get("LA-PD-P10", {}).get("strategy_stats", {})
    hist_cal = la_stats.get("predictor_calibration", {})
    ema_cal = la_stats.get("ema_calibration", {})

    if hist_cal and ema_cal:
        quantiles = ["q10", "q50", "q90"]
        ideal = [0.10, 0.50, 0.90]
        hist_cov = [hist_cal.get(f"{q}_coverage", 0) for q in quantiles]
        ema_cov = [ema_cal.get(f"{q}_coverage", 0) for q in quantiles]

        x = np.arange(len(quantiles))
        width = 0.22

        # Plot bars with better colors
        bars_ideal = ax.bar(
            x - width,
            ideal,
            width,
            label="Ideal",
            color="#E8E8E8",
            edgecolor=BLUE_DARK,
            linewidth=1.5,
            hatch="///",
        )
        bars_hist = ax.bar(
            x, hist_cov, width, label="Histogram", color=ORANGE_DARK, edgecolor="white", linewidth=1
        )
        bars_ema = ax.bar(
            x + width, ema_cov, width, label="EMA", color=BLUE_MED, edgecolor="white", linewidth=1
        )

        # Add value labels on top of bars
        for bars, values in [(bars_ideal, ideal), (bars_hist, hist_cov), (bars_ema, ema_cov)]:
            for bar, val in zip(bars, values, strict=False):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.02,
                    f"{val:.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold" if bars == bars_ideal else "normal",
                )

        # Add deviation arrows for key insights
        # q10: Histogram is conservative (18% vs 10%)
        ax.annotate(
            "",
            xy=(0, 0.18),
            xytext=(0 - width, 0.10),
            arrowprops={"arrowstyle": "->", "color": RED_SOFT, "lw": 1.5},
        )
        ax.text(-0.15, 0.14, "+8%", fontsize=8, color=RED_SOFT)

        # q50: EMA overestimates
        ax.annotate(
            "",
            xy=(1 + width, 0.72),
            xytext=(1 - width, 0.50),
            arrowprops={"arrowstyle": "->", "color": RED_SOFT, "lw": 1.5},
        )
        ax.text(1.3, 0.61, "+22%", fontsize=8, color=RED_SOFT)

        ax.set_ylabel(r"Coverage P(actual $\leq$ predicted)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(["10th Percentile", "50th Percentile", "90th Percentile"], fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper left", framealpha=0.95, fontsize=10)

        # Style
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)

        # Add ideal line
        ax.axhline(y=0.5, color=GRAY, linestyle=":", alpha=0.5, linewidth=1)

        ax.set_title("Predictor Calibration Analysis", fontsize=13, pad=15)

        plt.tight_layout()
        plt.savefig(output_dir / "online_calibration.png", dpi=300)
        plt.close()
        logger.info("Saved: online_calibration.png")


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Generate all Stage 1 ICML figures."""
    parser = argparse.ArgumentParser(description="Generate Stage 1 figures")
    parser.add_argument("--offline", action="store_true", help="Generate offline figures only")
    parser.add_argument(
        "--multimodel", action="store_true", help="Generate multi-model figures only"
    )
    parser.add_argument("--online", action="store_true", help="Generate online figures only")
    parser.add_argument("--ablation", action="store_true", help="Generate ablation figures only")
    args = parser.parse_args()

    apply_style("paper")

    results_dir = Path("experiment/results/stage1")
    output_dir = results_dir / "plots_icml"
    output_dir.mkdir(parents=True, exist_ok=True)

    # If no specific flag, generate all
    generate_all = not (args.offline or args.multimodel or args.online or args.ablation)

    if args.offline or generate_all:
        generate_offline_figures(results_dir, output_dir)

    if args.multimodel or generate_all:
        generate_multimodel_figures(output_dir)

    if args.online or generate_all:
        plot_online_results(output_dir)

    if args.ablation or generate_all:
        plot_online_ablation(output_dir)

    logger.info(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
