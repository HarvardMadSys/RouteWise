"""Plot counterfactual simulation results.

Generates bar charts with 95% CI error bars for SLO violation rate
and P99 latency, comparing all policies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Policy display order and colors.
POLICY_ORDER = [
    "oracle_per_window",
    "smart_hedge",
    "fastest_fixed",
    "lp_mix",
    "cheapest_fixed",
    "round_robin",
]
POLICY_LABELS = {
    "oracle_per_window": "Oracle\n(per-window)",
    "smart_hedge": "Smart\nHedge",
    "fastest_fixed": "Fastest\nFixed",
    "lp_mix": "LP Mix",
    "cheapest_fixed": "Cheapest\nFixed",
    "round_robin": "Round\nRobin",
}
POLICY_COLORS = {
    "oracle_per_window": "#999999",
    "smart_hedge": "#2196F3",
    "fastest_fixed": "#4CAF50",
    "lp_mix": "#FF9800",
    "cheapest_fixed": "#F44336",
    "round_robin": "#9C27B0",
}


def load_data(results_dir: str) -> pd.DataFrame:
    """Load per-trial results CSV."""
    path = Path(results_dir) / "per_trial_results.csv"
    return pd.read_csv(path)


def plot_slo_and_p99(
    df: pd.DataFrame,
    title_suffix: str,
    output_path: str,
) -> None:
    """Plot SLO violation rate and P99 side by side with CI error bars."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    policies = [p for p in POLICY_ORDER if p in df["policy"].unique()]
    x = np.arange(len(policies))
    width = 0.6

    for ax_idx, (metric, ylabel, fmt_func) in enumerate([
        ("slo_violation_rate", "SLO Violation Rate (%)", lambda v: v * 100),
        ("p99_ms", "P99 TTFT (ms)", lambda v: v),
    ]):
        ax = axes[ax_idx]
        means = []
        ci_los = []
        ci_his = []
        colors = []

        for p in policies:
            vals = df[df["policy"] == p][metric].dropna().values
            mean = fmt_func(np.mean(vals))
            lo = fmt_func(np.percentile(vals, 2.5))
            hi = fmt_func(np.percentile(vals, 97.5))
            means.append(mean)
            ci_los.append(mean - lo)
            ci_his.append(hi - mean)
            colors.append(POLICY_COLORS.get(p, "#666666"))

        bars = ax.bar(
            x, means, width,
            yerr=[ci_los, ci_his],
            capsize=4,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
            error_kw={"linewidth": 1.2},
        )

        # Add value labels on bars.
        for i, (bar, m) in enumerate(zip(bars, means)):
            if metric == "slo_violation_rate":
                label = f"{m:.1f}%"
            else:
                label = f"{m:.0f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ci_his[i] + (max(means) * 0.02),
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels([POLICY_LABELS.get(p, p) for p in policies], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle(
        f"Counterfactual Simulation: {title_suffix}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_p99_vs_cost(
    df: pd.DataFrame,
    title_suffix: str,
    output_path: str,
) -> None:
    """Scatter plot: P99 vs cost per request (Pareto view)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    policies = [p for p in POLICY_ORDER if p in df["policy"].unique()]

    for p in policies:
        vals = df[df["policy"] == p]
        p99_mean = vals["p99_ms"].mean()
        cost_mean = vals["mean_cost"].mean()
        slo_mean = vals["slo_violation_rate"].mean() * 100

        ax.scatter(
            cost_mean * 1e6,  # Convert to micro-dollars for readability.
            p99_mean,
            s=150,
            c=POLICY_COLORS.get(p, "#666"),
            edgecolors="black",
            linewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            f"{POLICY_LABELS.get(p, p).replace(chr(10), ' ')}\n({slo_mean:.1f}% SLO)",
            (cost_mean * 1e6, p99_mean),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=8,
            ha="left",
        )

    ax.set_xlabel("Cost per Request (micro-USD)", fontsize=11)
    ax.set_ylabel("P99 TTFT (ms)", fontsize=11)
    ax.set_title(
        f"P99 vs Cost: {title_suffix}",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--title", type=str, default="")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    df = load_data(args.results_dir)

    plot_slo_and_p99(
        df,
        args.title,
        str(results_dir / "counterfactual_slo_p99.png"),
    )
    plot_p99_vs_cost(
        df,
        args.title,
        str(results_dir / "counterfactual_p99_vs_cost.png"),
    )


if __name__ == "__main__":
    main()
