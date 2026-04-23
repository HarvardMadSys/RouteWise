#!/usr/bin/env python3
r"""Deep analysis of Phase 5 24-hour OpenRouter evaluation.

Generates three publication-ready figures:
1. Provider latency drift over 24 hours (per-provider P50/P99 per hour)
2. LP routing weights over time (how the mix adapts)
3. Hedging analysis (trigger rate over time, success breakdown)

Usage:
    python -m experiment.scripts.plot.latency.phase5_24h_analysis \
        --input experiment/results/phase5_qwen3_235b/run_20260405_073250 \
        --output experiment/results/phase5_qwen3_235b/run_20260405_073250/plots_analysis
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from experiment.scripts.plot.common import (
    PROVIDER_COLORS,
    apply_style,
    save_figure,
)

apply_style("paper")

# Extended provider colors for Qwen3-235B providers
QWEN_PROVIDER_COLORS = {
    **PROVIDER_COLORS,
    "WandB": "#3498db",
    "Alibaba": "#e74c3c",
    "SiliconFlow": "#f39c12",
    "DeepInfra": "#2ecc71",
    "Chutes": "#9b59b6",
    "Google": "#1abc9c",
    "Novita": "#e67e22",
    "AtlasCloud": "#34495e",
    "Together": "#d35400",
    "Cerebras": "#27ae60",
    "Friendli": "#8e44ad",
    "Parasail": "#16a085",
    "unknown": "#bdc3c7",
}


def load_log(input_dir: Path) -> pd.DataFrame:
    """Load evaluation log and add derived columns."""
    df = pd.read_csv(input_dir / "evaluation_log.csv")
    t0 = df["timestamp"].min()
    df["elapsed_h"] = (df["timestamp"] - t0) / 3600
    df["hour_bin"] = df["elapsed_h"].astype(int)
    return df


def plot_provider_latency_drift(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot per-provider P50 and P99 latency over 24 hours.

    Uses openrouter_auto data since it distributes traffic across all providers.
    """
    auto = df[
        (df["policy"] == "openrouter_auto") & (df["status"] == "success")
    ].copy()

    # Keep providers with at least 100 total requests
    provider_counts = auto["actual_provider"].value_counts()
    top_providers = provider_counts[provider_counts >= 100].index.tolist()
    auto = auto[auto["actual_provider"].isin(top_providers)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=False)

    for ax, (metric, percentile) in zip(
        axes, [("P50", 50), ("P99", 99)]
    ):
        for provider in top_providers:
            pdata = auto[auto["actual_provider"] == provider]
            hourly = pdata.groupby("hour_bin")["ttft_ms"].quantile(
                percentile / 100
            )
            # Only plot hours with enough data (>= 3 requests)
            hour_counts = pdata.groupby("hour_bin").size()
            valid_hours = hour_counts[hour_counts >= 3].index
            hourly = hourly[hourly.index.isin(valid_hours)]

            color = QWEN_PROVIDER_COLORS.get(provider, "#7f8c8d")
            ax.plot(
                hourly.index,
                hourly.values,
                label=provider,
                color=color,
                alpha=0.8,
                linewidth=1.8,
            )

        ax.set_xlabel("Time (hours)")
        ax.set_ylabel(f"{metric} TTFT (ms)")
        ax.set_title(f"{metric} Latency by Provider")
        ax.set_xlim(0, 23)

    # Single legend for both subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(len(top_providers), 5),
        fontsize=11,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    save_figure(fig, output_dir, "provider_latency_drift_24h")
    plt.close(fig)


def plot_routing_weights_over_time(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot how LP routing mix adapts over time (stacked area chart)."""
    lp = df[(df["policy"] == "lp_mix") & (df["status"] == "success")].copy()

    # Compute per-hour provider distribution
    providers_all = lp["actual_provider"].value_counts().index.tolist()
    hour_bins = sorted(lp["hour_bin"].unique())

    # Build distribution matrix
    dist_data = {}
    for provider in providers_all:
        fracs = []
        for h in hour_bins:
            chunk = lp[lp["hour_bin"] == h]
            frac = (chunk["actual_provider"] == provider).sum() / len(chunk)
            fracs.append(frac)
        dist_data[provider] = fracs

    fig, ax = plt.subplots(figsize=(10, 4))

    # Sort providers by total share (largest at bottom)
    sorted_providers = sorted(
        providers_all,
        key=lambda p: sum(dist_data[p]),
        reverse=True,
    )

    bottom = np.zeros(len(hour_bins))
    for provider in sorted_providers:
        values = np.array(dist_data[provider])
        color = QWEN_PROVIDER_COLORS.get(provider, "#7f8c8d")
        ax.bar(
            hour_bins,
            values,
            bottom=bottom,
            width=0.9,
            label=provider,
            color=color,
            edgecolor="white",
            linewidth=0.3,
        )
        bottom += values

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Routing Fraction")
    ax.set_title("Latency-Aware Routing: Provider Mix Over 24 Hours")
    ax.set_xlim(-0.5, max(hour_bins) + 0.5)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(
        loc="upper right",
        fontsize=11,
        framealpha=0.9,
    )

    fig.tight_layout()
    save_figure(fig, output_dir, "routing_weights_over_time")
    plt.close(fig)


def plot_hedging_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot hedging trigger rate, winner distribution, and SLO impact over time."""
    sh = df[df["policy"] == "smart_hedge"].copy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # --- Panel (a): Hedge trigger rate over time ---
    ax = axes[0]
    hour_bins = sorted(sh["hour_bin"].unique())
    trigger_rates = []
    slo_rates_sh = []
    slo_rates_lp = []

    lp = df[df["policy"] == "lp_mix"].copy()

    for h in hour_bins:
        sh_h = sh[sh["hour_bin"] == h]
        lp_h = lp[lp["hour_bin"] == h]
        trigger_rates.append(sh_h["hedge_triggered"].mean())
        slo_rates_sh.append(
            sh_h["slo_violated"].mean() if len(sh_h) > 0 else 0
        )
        slo_rates_lp.append(
            lp_h["slo_violated"].mean() if len(lp_h) > 0 else 0
        )

    ax.bar(
        hour_bins,
        trigger_rates,
        color="#ff7f0e",
        alpha=0.7,
        label="Hedge Trigger Rate",
    )
    ax.plot(
        hour_bins,
        slo_rates_lp,
        "s--",
        color="#2ca02c",
        label="SLO Viol. (Latency-Aware)",
        markersize=4,
        linewidth=1.5,
    )
    ax.plot(
        hour_bins,
        slo_rates_sh,
        "o-",
        color="#d62728",
        label="SLO Viol. (Smart Hedge)",
        markersize=4,
        linewidth=1.5,
    )
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Rate")
    ax.set_title("(a) Hedge Trigger & SLO Violations")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(-0.5, max(hour_bins) + 0.5)

    # --- Panel (b): Hedge winner distribution ---
    ax = axes[1]
    hedged = sh[sh["hedge_triggered"] == True]  # noqa: E712
    winner_counts = hedged["hedge_winner"].value_counts()
    colors = ["#3498db", "#e74c3c"]
    labels = [
        f"Primary wins\n({winner_counts.get('primary', 0)})",
        f"Backup wins\n({winner_counts.get('backup', 0)})",
    ]
    values = [
        winner_counts.get("primary", 0),
        winner_counts.get("backup", 0),
    ]
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 12},
    )
    for t in autotexts:
        t.set_fontsize(13)
        t.set_fontweight("bold")
    ax.set_title("(b) Hedge Winner Distribution")

    # --- Panel (c): Backup provider distribution ---
    ax = axes[2]
    backup_providers = hedged["hedge_provider"].value_counts()
    bp_colors = [
        QWEN_PROVIDER_COLORS.get(p, "#7f8c8d") for p in backup_providers.index
    ]
    bp_labels = [
        f"{p}\n({c})" for p, c in zip(backup_providers.index, backup_providers.values)
    ]
    wedges, texts, autotexts = ax.pie(
        backup_providers.values,
        labels=bp_labels,
        colors=bp_colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 12},
    )
    for t in autotexts:
        t.set_fontsize(13)
        t.set_fontweight("bold")
    ax.set_title("(c) Backup Provider Selection")

    fig.suptitle(
        f"Smart Hedging Analysis: {len(hedged)} hedges triggered "
        f"({len(hedged)/len(sh)*100:.1f}% of {len(sh)} requests)",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, output_dir, "hedging_analysis_24h")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deep analysis of 24h OpenRouter evaluation"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to evaluation run directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for plots (default: <input>/plots_analysis)",
    )
    args = parser.parse_args()

    output_dir = args.output or args.input / "plots_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.input}...")
    df = load_log(args.input)
    print(f"  {len(df)} rows, {df['policy'].nunique()} policies")
    print(f"  Time span: {df['elapsed_h'].max():.1f} hours")

    print("\nPlot 1: Provider latency drift...")
    plot_provider_latency_drift(df, output_dir)

    print("Plot 2: Routing weights over time...")
    plot_routing_weights_over_time(df, output_dir)

    print("Plot 3: Hedging analysis...")
    plot_hedging_analysis(df, output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
