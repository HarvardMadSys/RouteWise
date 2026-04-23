#!/usr/bin/env python3
"""Plot misprediction robustness results.

Two layouts:

    Per-config plots (one PNG per dataset x plan x sweep, 4 curves each):
        misprediction_{dataset}_{plan}_bias_vs_optimal.png
        misprediction_{dataset}_{plan}_bias_vs_oracle.png
        misprediction_{dataset}_{plan}_noise_vs_optimal.png
        misprediction_{dataset}_{plan}_noise_vs_oracle.png

    Regime comparison plots (one PNG per dataset x sweep x framing, overlays
    three plans as separate curves to reveal regime dependence):
        misprediction_{dataset}_regimes_bias_vs_optimal.png
        misprediction_{dataset}_regimes_bias_vs_oracle.png
        misprediction_{dataset}_regimes_noise_vs_optimal.png
        misprediction_{dataset}_regimes_noise_vs_oracle.png

Each PNG is a standalone figure (no matplotlib subplots).

Usage:
    python experiment/scripts/plot/misprediction_robustness.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Reference values loaded from existing oracle experiment (Pro plan on BurstGPT).
# See experiment/results/oracle/oracle_experiment_results.json.
REFERENCE_PD_EMA_CR = 1.1807
REFERENCE_GREEDY_CR = 1.2986

# Colors used for regime-comparison plots (Base/Plus/Pro).
PLAN_COLORS = {
    "Base": "#e34a33",    # red (tight quota)
    "Plus": "#756bb1",    # purple (medium)
    "Pro": "#2c7fb8",     # blue (loose)
}
PLAN_MARKERS = {"Base": "o", "Plus": "s", "Pro": "^"}


def _set_bias_xaxis(ax) -> None:
    """Apply a log-scale x-axis with explicit multiplier tick labels."""
    ax.set_xscale("log")
    ax.set_xticks([0.5, 0.667, 0.8, 1.0, 1.25, 1.5, 2.0])
    ax.set_xticklabels(["0.5x", "0.67x", "0.8x", "1.0x", "1.25x", "1.5x", "2.0x"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.xaxis.set_minor_formatter(plt.NullFormatter())


# =============================================================================
# Per-config single-plan plots (paper main-figure candidates).
# =============================================================================


def plot_bias_vs_optimal(plan_res: dict, output_path: Path, title: str) -> None:
    """Single-plan bias sweep vs Optimal."""
    bias_entries = plan_res["bias_sweep"]
    bias_x = [e["bias_factor"] for e in bias_entries]
    bias_y = [e["relative_cost_vs_optimal"] for e in bias_entries]
    oracle_cr = (
        plan_res["baselines"]["PD-Oracle"]["costs"]["total"]
        / plan_res["baselines"]["Optimal"]["costs"]["total"]
    )

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(
        bias_x,
        bias_y,
        marker="o",
        linewidth=2,
        markersize=7,
        color="#2c7fb8",
        label="RouteWise (biased oracle)",
        zorder=3,
    )
    ax.axhline(
        REFERENCE_PD_EMA_CR,
        linestyle="--",
        color="#31a354",
        alpha=0.85,
        linewidth=1.6,
        label=f"PD-EMA ({REFERENCE_PD_EMA_CR:.2f}x)",
    )
    ax.axhline(
        REFERENCE_GREEDY_CR,
        linestyle="--",
        color="#e34a33",
        alpha=0.85,
        linewidth=1.6,
        label=f"Greedy ({REFERENCE_GREEDY_CR:.2f}x)",
    )
    ax.axhline(
        1.0,
        linestyle=":",
        color="black",
        alpha=0.4,
        linewidth=1.2,
        label="Offline Optimal",
    )
    ax.scatter(
        [1.0],
        [oracle_cr],
        marker="*",
        s=180,
        color="#2c7fb8",
        edgecolor="black",
        linewidths=1.0,
        zorder=5,
        label=f"PD-Oracle ({oracle_cr:.3f}x)",
    )
    _set_bias_xaxis(ax)
    ax.set_xlabel("Systematic bias factor  (predicted = true x bias)")
    ax.set_ylabel("Relative Cost  (vs Offline Optimal)")
    ax.set_title(f"Bias sweep  ({title})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
    ymax = max(REFERENCE_GREEDY_CR + 0.05, max(bias_y) + 0.05)
    ymin = min(0.95, min(bias_y) - 0.02)
    ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_noise_vs_optimal(plan_res: dict, output_path: Path, title: str) -> None:
    """Single-plan noise sweep vs Optimal."""
    noise_entries = plan_res["noise_sweep"]
    noise_x = [e["noise_std"] for e in noise_entries]
    noise_y_mean = [e["relative_cost_vs_optimal_mean"] for e in noise_entries]
    noise_y_std = [e["relative_cost_vs_optimal_std"] for e in noise_entries]

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.errorbar(
        noise_x,
        noise_y_mean,
        yerr=noise_y_std,
        marker="s",
        linewidth=2,
        markersize=7,
        color="#756bb1",
        capsize=3,
        label="RouteWise (noisy oracle, 3 seeds)",
        zorder=3,
    )
    ax.axhline(
        REFERENCE_PD_EMA_CR,
        linestyle="--",
        color="#31a354",
        alpha=0.85,
        linewidth=1.6,
        label=f"PD-EMA ({REFERENCE_PD_EMA_CR:.2f}x)",
    )
    ax.axhline(
        REFERENCE_GREEDY_CR,
        linestyle="--",
        color="#e34a33",
        alpha=0.85,
        linewidth=1.6,
        label=f"Greedy ({REFERENCE_GREEDY_CR:.2f}x)",
    )
    ax.axhline(
        1.0,
        linestyle=":",
        color="black",
        alpha=0.4,
        linewidth=1.2,
        label="Offline Optimal",
    )
    ax.set_xlabel("Log-normal noise std  (predicted = true x exp(sigma*Z))")
    ax.set_ylabel("Relative Cost  (vs Offline Optimal)")
    ax.set_title(f"Noise sweep  ({title})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
    ymax = max(REFERENCE_GREEDY_CR + 0.05, max(noise_y_mean) + 0.05)
    ymin = min(0.95, min(noise_y_mean) - 0.02)
    ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_bias_vs_oracle(plan_res: dict, output_path: Path, title: str) -> None:
    """Single-plan bias sweep vs Oracle (zoomed)."""
    bias_entries = plan_res["bias_sweep"]
    bias_x = [e["bias_factor"] for e in bias_entries]
    bias_y = [e["relative_cost_vs_oracle"] for e in bias_entries]

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(
        bias_x,
        bias_y,
        marker="o",
        linewidth=2,
        markersize=7,
        color="#2c7fb8",
        zorder=3,
        label="RouteWise (biased oracle)",
    )
    ax.axhline(
        1.0,
        linestyle=":",
        color="black",
        alpha=0.4,
        linewidth=1.2,
        label="PD-Oracle (perfect prediction)",
    )
    _set_bias_xaxis(ax)
    ax.set_xlabel("Systematic bias factor")
    ax.set_ylabel("Relative Cost  (vs PD-Oracle)")
    ax.set_title(f"Bias sweep (zoomed)  ({title})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    max_idx = int(np.argmax(bias_y))
    ax.annotate(
        f"max +{(bias_y[max_idx] - 1) * 100:.2f}%",
        xy=(bias_x[max_idx], bias_y[max_idx]),
        xytext=(-70, 10),
        textcoords="offset points",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="gray", alpha=0.6),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_noise_vs_oracle(plan_res: dict, output_path: Path, title: str) -> None:
    """Single-plan noise sweep vs Oracle (zoomed)."""
    noise_entries = plan_res["noise_sweep"]
    noise_x = [e["noise_std"] for e in noise_entries]
    noise_y_mean = [e["relative_cost_vs_oracle_mean"] for e in noise_entries]
    noise_y_std = [e["relative_cost_vs_oracle_std"] for e in noise_entries]

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.errorbar(
        noise_x,
        noise_y_mean,
        yerr=noise_y_std,
        marker="s",
        linewidth=2,
        markersize=7,
        color="#756bb1",
        capsize=3,
        zorder=3,
        label="RouteWise (noisy oracle, 3 seeds)",
    )
    ax.axhline(
        1.0,
        linestyle=":",
        color="black",
        alpha=0.4,
        linewidth=1.2,
        label="PD-Oracle (perfect prediction)",
    )
    ax.set_xlabel("Log-normal noise std")
    ax.set_ylabel("Relative Cost  (vs PD-Oracle)")
    ax.set_title(f"Noise sweep (zoomed)  ({title})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    max_idx = int(np.argmax(noise_y_mean))
    ax.annotate(
        f"max +{(noise_y_mean[max_idx] - 1) * 100:.2f}%",
        xy=(noise_x[max_idx], noise_y_mean[max_idx]),
        xytext=(-70, 10),
        textcoords="offset points",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="gray", alpha=0.6),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# =============================================================================
# Regime-comparison plots (overlay three quota plans in one figure).
# =============================================================================


def plot_bias_regime_comparison_vs_oracle(
    dataset_res: dict,
    output_path: Path,
    dataset: str,
) -> None:
    """Overlay Base/Plus/Pro on one bias-vs-Oracle figure."""
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for plan_name in ["Base", "Plus", "Pro"]:
        if plan_name not in dataset_res["plans"]:
            continue
        plan_res = dataset_res["plans"][plan_name]
        quota = plan_res["quota"]
        bias_entries = plan_res["bias_sweep"]
        bias_x = [e["bias_factor"] for e in bias_entries]
        bias_y = [e["relative_cost_vs_oracle"] for e in bias_entries]
        ax.plot(
            bias_x,
            bias_y,
            marker=PLAN_MARKERS[plan_name],
            linewidth=2,
            markersize=6,
            color=PLAN_COLORS[plan_name],
            label=f"{plan_name} (Q={quota})",
            zorder=3,
        )
    ax.axhline(
        1.0,
        linestyle=":",
        color="black",
        alpha=0.4,
        linewidth=1.2,
    )
    _set_bias_xaxis(ax)
    ax.set_xlabel("Systematic bias factor")
    ax.set_ylabel("Relative Cost  (vs PD-Oracle)")
    ax.set_title(f"Bias across quota regimes  ({dataset.title()})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, title="Quota plan")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_noise_regime_comparison_vs_oracle(
    dataset_res: dict,
    output_path: Path,
    dataset: str,
) -> None:
    """Overlay Base/Plus/Pro on one noise-vs-Oracle figure."""
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for plan_name in ["Base", "Plus", "Pro"]:
        if plan_name not in dataset_res["plans"]:
            continue
        plan_res = dataset_res["plans"][plan_name]
        quota = plan_res["quota"]
        noise_entries = plan_res["noise_sweep"]
        noise_x = [e["noise_std"] for e in noise_entries]
        noise_y_mean = [e["relative_cost_vs_oracle_mean"] for e in noise_entries]
        noise_y_std = [e["relative_cost_vs_oracle_std"] for e in noise_entries]
        ax.errorbar(
            noise_x,
            noise_y_mean,
            yerr=noise_y_std,
            marker=PLAN_MARKERS[plan_name],
            linewidth=2,
            markersize=6,
            color=PLAN_COLORS[plan_name],
            capsize=3,
            label=f"{plan_name} (Q={quota})",
            zorder=3,
        )
    ax.axhline(
        1.0,
        linestyle=":",
        color="black",
        alpha=0.4,
        linewidth=1.2,
    )
    ax.set_xlabel("Log-normal noise std")
    ax.set_ylabel("Relative Cost  (vs PD-Oracle)")
    ax.set_title(f"Noise across quota regimes  ({dataset.title()})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, title="Quota plan")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_bias_regime_comparison_vs_optimal(
    dataset_res: dict,
    output_path: Path,
    dataset: str,
) -> None:
    """Overlay Base/Plus/Pro on one bias-vs-Optimal figure."""
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for plan_name in ["Base", "Plus", "Pro"]:
        if plan_name not in dataset_res["plans"]:
            continue
        plan_res = dataset_res["plans"][plan_name]
        quota = plan_res["quota"]
        bias_entries = plan_res["bias_sweep"]
        bias_x = [e["bias_factor"] for e in bias_entries]
        bias_y = [e["relative_cost_vs_optimal"] for e in bias_entries]
        ax.plot(
            bias_x,
            bias_y,
            marker=PLAN_MARKERS[plan_name],
            linewidth=2,
            markersize=6,
            color=PLAN_COLORS[plan_name],
            label=f"{plan_name} (Q={quota})",
            zorder=3,
        )
    ax.axhline(
        REFERENCE_PD_EMA_CR,
        linestyle="--",
        color="#31a354",
        alpha=0.7,
        linewidth=1.3,
        label=f"PD-EMA ({REFERENCE_PD_EMA_CR:.2f}x)",
    )
    ax.axhline(
        REFERENCE_GREEDY_CR,
        linestyle="--",
        color="#e34a33",
        alpha=0.7,
        linewidth=1.3,
        label=f"Greedy ({REFERENCE_GREEDY_CR:.2f}x)",
    )
    ax.axhline(
        1.0,
        linestyle=":",
        color="black",
        alpha=0.4,
        linewidth=1.2,
    )
    _set_bias_xaxis(ax)
    ax.set_xlabel("Systematic bias factor")
    ax.set_ylabel("Relative Cost  (vs Offline Optimal)")
    ax.set_title(f"Bias across quota regimes  ({dataset.title()})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, title="Quota plan")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_noise_regime_comparison_vs_optimal(
    dataset_res: dict,
    output_path: Path,
    dataset: str,
) -> None:
    """Overlay Base/Plus/Pro on one noise-vs-Optimal figure."""
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for plan_name in ["Base", "Plus", "Pro"]:
        if plan_name not in dataset_res["plans"]:
            continue
        plan_res = dataset_res["plans"][plan_name]
        quota = plan_res["quota"]
        noise_entries = plan_res["noise_sweep"]
        noise_x = [e["noise_std"] for e in noise_entries]
        noise_y_mean = [e["relative_cost_vs_optimal_mean"] for e in noise_entries]
        noise_y_std = [e["relative_cost_vs_optimal_std"] for e in noise_entries]
        ax.errorbar(
            noise_x,
            noise_y_mean,
            yerr=noise_y_std,
            marker=PLAN_MARKERS[plan_name],
            linewidth=2,
            markersize=6,
            color=PLAN_COLORS[plan_name],
            capsize=3,
            label=f"{plan_name} (Q={quota})",
            zorder=3,
        )
    ax.axhline(
        REFERENCE_PD_EMA_CR,
        linestyle="--",
        color="#31a354",
        alpha=0.7,
        linewidth=1.3,
        label=f"PD-EMA ({REFERENCE_PD_EMA_CR:.2f}x)",
    )
    ax.axhline(
        REFERENCE_GREEDY_CR,
        linestyle="--",
        color="#e34a33",
        alpha=0.7,
        linewidth=1.3,
        label=f"Greedy ({REFERENCE_GREEDY_CR:.2f}x)",
    )
    ax.axhline(
        1.0,
        linestyle=":",
        color="black",
        alpha=0.4,
        linewidth=1.2,
    )
    ax.set_xlabel("Log-normal noise std")
    ax.set_ylabel("Relative Cost  (vs Offline Optimal)")
    ax.set_title(f"Noise across quota regimes  ({dataset.title()})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, title="Quota plan")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot misprediction robustness results (one PNG per panel)"
    )
    parser.add_argument(
        "--results",
        type=str,
        default="experiment/results/misprediction/misprediction_results.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiment/results/misprediction",
    )
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-config plots (one per dataset x plan x sweep x framing).
    for dataset, ds_res in data["datasets"].items():
        for plan_name, plan_res in ds_res["plans"].items():
            title = f"{dataset.title()}, {plan_name} plan, Q={plan_res['quota']}"
            prefix = f"misprediction_{dataset}_{plan_name.lower()}"
            plot_bias_vs_optimal(
                plan_res, output_dir / f"{prefix}_bias_vs_optimal.png", title
            )
            plot_noise_vs_optimal(
                plan_res, output_dir / f"{prefix}_noise_vs_optimal.png", title
            )
            plot_bias_vs_oracle(
                plan_res, output_dir / f"{prefix}_bias_vs_oracle.png", title
            )
            plot_noise_vs_oracle(
                plan_res, output_dir / f"{prefix}_noise_vs_oracle.png", title
            )

    # Regime-comparison plots (one per dataset x sweep x framing).
    for dataset, ds_res in data["datasets"].items():
        plot_bias_regime_comparison_vs_oracle(
            ds_res,
            output_dir / f"misprediction_{dataset}_regimes_bias_vs_oracle.png",
            dataset,
        )
        plot_noise_regime_comparison_vs_oracle(
            ds_res,
            output_dir / f"misprediction_{dataset}_regimes_noise_vs_oracle.png",
            dataset,
        )
        plot_bias_regime_comparison_vs_optimal(
            ds_res,
            output_dir / f"misprediction_{dataset}_regimes_bias_vs_optimal.png",
            dataset,
        )
        plot_noise_regime_comparison_vs_optimal(
            ds_res,
            output_dir / f"misprediction_{dataset}_regimes_noise_vs_optimal.png",
            dataset,
        )


if __name__ == "__main__":
    main()
