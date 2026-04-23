#!/usr/bin/env python3
r"""Phase 1 Latency Profiling Visualization.

Generates figures for latency profiling analysis:
- CDF/CCDF plots for all providers
- Rolling P99 drift analysis
- Provider variance comparison
- Percentile heatmaps

Usage:
    python -m experiment.scripts.plot.latency.phase1_profiling \
        --input data/latency_profile.csv --output results/phase1/
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment.scripts.plot.common import (
    compute_ccdf,
    get_provider_color,
)

# Project root for default output paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent

# =============================================================================
# Configuration
# =============================================================================

# Default workload type (we focus on base probing requests)
# The script can handle data with 'workload' column, but assumes 'base' if not specified
DEFAULT_WORKLOAD = "base"

# Figure settings
FIGURE_DPI = 300
FIGURE_FORMAT = "png"  # Can also be 'pdf' for paper


def setup_plot_style():
    """Setup matplotlib style for consistent figures."""
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.figsize": (10, 6),
            "figure.dpi": 100,
        }
    )


setup_plot_style()


# =============================================================================
# Utility Functions
# =============================================================================


def compute_percentiles(latencies: np.ndarray) -> dict[str, float]:
    """Compute key percentile statistics."""
    return {
        "p50": np.percentile(latencies, 50),
        "p90": np.percentile(latencies, 90),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99),
        "mean": np.mean(latencies),
        "std": np.std(latencies),
        "min": np.min(latencies),
        "max": np.max(latencies),
        "count": len(latencies),
    }


def compute_cdf(latencies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute empirical CDF."""
    sorted_data = np.sort(latencies)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf


def ensure_output_dir(output_dir: str) -> None:
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)


# =============================================================================
# Phase 1: Latency Profiling Analysis
# =============================================================================


def load_profiling_data(filepath: str) -> pd.DataFrame:
    """Load and preprocess latency profiling CSV data."""
    print(f"Loading data from: {filepath}")

    # Handle malformed CSV lines (some rows may have extra fields due to commas in error messages)
    try:
        df = pd.read_csv(filepath)
    except pd.errors.ParserError:
        print("  Warning: CSV has malformed lines, using error_bad_lines=False")
        # Try with on_bad_lines='skip' (pandas >= 1.3) or error_bad_lines=False (older)
        try:
            df = pd.read_csv(filepath, on_bad_lines="skip")
        except TypeError:
            # Older pandas version
            df = pd.read_csv(filepath, error_bad_lines=False)
        print("  Skipped malformed lines")

    # Basic info
    print(f"  Total records: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")

    # Filter successful requests only
    if "status" in df.columns:
        success_count = (df["status"] == "success").sum()
        df = df[df["status"] == "success"].copy()
        print(f"  Successful requests: {success_count:,} ({success_count/len(df)*100:.1f}%)")

    # Parse request_id to extract workload type and round number
    def parse_request_id(req_id: str) -> tuple[str, int, int]:
        """Parse request_id like 'Cerebras-base-R0-0' -> (workload, round, idx)."""
        parts = req_id.split("-")
        if len(parts) >= 4:
            workload = parts[1]  # base, prefill, decode
            round_num = int(parts[2].replace("R", ""))
            idx = int(parts[3])
            return workload, round_num, idx
        return "unknown", 0, 0

    parsed = df["request_id"].apply(parse_request_id)
    df["workload"] = [p[0] for p in parsed]
    df["round"] = [p[1] for p in parsed]
    df["round_idx"] = [p[2] for p in parsed]

    # Convert timestamp to datetime if needed
    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    # Summary by provider
    print("\n  Providers found:")
    for provider in sorted(df["provider"].unique()):
        count = len(df[df["provider"] == provider])
        print(f"    {provider}: {count:,} requests")

    return df


def filter_by_workload(df: pd.DataFrame, workload: str | None = None) -> pd.DataFrame:
    """Filter dataframe by workload type. If no workload column exists, return all data."""
    if workload is None:
        workload = DEFAULT_WORKLOAD
    if "workload" not in df.columns:
        return df.copy()
    return df[df["workload"] == workload].copy()


def compute_rolling_percentile(
    df: pd.DataFrame, provider: str, percentile: int = 99, window_size: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Compute rolling percentile for a provider over time."""
    provider_df = df[df["provider"] == provider].sort_values("timestamp")
    latencies = provider_df["ttft_ms"].values

    if len(latencies) < window_size:
        return np.array([]), np.array([])

    # Compute rolling percentile
    rolling_p = []
    indices = []
    for i in range(window_size, len(latencies) + 1):
        window = latencies[i - window_size : i]
        rolling_p.append(np.percentile(window, percentile))
        indices.append(i)

    return np.array(indices), np.array(rolling_p)


# Phase 1 Plotting Functions


def plot_cdf_all_providers(
    df: pd.DataFrame, workload: str, output_dir: str, title_suffix: str = ""
) -> str:
    """Plot CDF of TTFT for all providers on single figure.

    Figure 1: provider_latency_cdf.png
    Purpose: Show distribution of latency across providers
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    workload_df = filter_by_workload(df, workload)
    providers = sorted(workload_df["provider"].unique())

    for provider in providers:
        provider_data = workload_df[workload_df["provider"] == provider]["ttft_ms"].values
        if len(provider_data) > 0:
            x, cdf = compute_cdf(provider_data)
            ax.plot(x, cdf, label=provider, color=get_provider_color(provider), linewidth=2)

    ax.set_xlabel("TTFT (ms)", fontsize=14)
    ax.set_ylabel("CDF", fontsize=14)
    ax.set_title(f"Latency CDF by Provider{title_suffix}", fontsize=16)
    ax.legend(loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)

    # Add vertical lines for common SLO thresholds
    for slo, style in [(500, "--"), (1000, ":"), (2000, "-.")]:
        ax.axvline(x=slo, color="gray", linestyle=style, alpha=0.5, label=f"SLO={slo}ms")

    plt.tight_layout()

    output_path = os.path.join(output_dir, f"provider_latency_cdf.{FIGURE_FORMAT}")
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {output_path}")
    return output_path


def plot_ccdf_all_providers(
    df: pd.DataFrame,
    workload: str,
    output_dir: str,
    log_scale: bool = True,
    title_suffix: str = "",
    slo_lines: list[int] | None = None,
) -> str:
    """Plot CCDF (survival function) to highlight heavy-tail behavior.

    Figure 2: provider_latency_ccdf.png
    Purpose: Show heavy-tail characteristics (P(X > x))
    Key insight: Heavy-tail distribution -> static profiling insufficient
    """
    if slo_lines is None:
        slo_lines = [500, 1000, 2000]
    fig, ax = plt.subplots(figsize=(12, 7))

    workload_df = filter_by_workload(df, workload)
    providers = sorted(workload_df["provider"].unique())

    for provider in providers:
        provider_data = workload_df[workload_df["provider"] == provider]["ttft_ms"].values
        if len(provider_data) > 0:
            x, ccdf = compute_ccdf(provider_data)
            # Filter out zero values for log scale
            mask = ccdf > 0
            ax.plot(
                x[mask], ccdf[mask], label=provider, color=get_provider_color(provider), linewidth=2
            )

    ax.set_xlabel("TTFT (ms)", fontsize=14)
    ax.set_ylabel("P(Latency > x)", fontsize=14)
    ax.set_title(f"Latency CCDF by Provider{title_suffix}", fontsize=16)

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1)

    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(left=0)

    # Add horizontal lines for percentile reference
    ax.axhline(y=0.1, color="gray", linestyle="--", alpha=0.5)  # P90
    ax.axhline(y=0.01, color="gray", linestyle=":", alpha=0.5)  # P99
    ax.text(
        ax.get_xlim()[1] * 0.98, 0.11, "P90", ha="right", va="bottom", fontsize=10, color="gray"
    )
    ax.text(
        ax.get_xlim()[1] * 0.98, 0.012, "P99", ha="right", va="bottom", fontsize=10, color="gray"
    )

    # Add vertical SLO lines with labels
    slo_colors = ["#e74c3c", "#f39c12", "#27ae60"]  # red, orange, green
    for slo, color in zip(slo_lines, slo_colors, strict=False):
        ax.axvline(x=slo, color=color, linestyle="--", alpha=0.7, linewidth=1.5)
        ax.text(
            slo + 20,
            0.5,
            f"SLO={slo}ms",
            rotation=90,
            va="center",
            fontsize=10,
            color=color,
            fontweight="bold",
        )

    plt.tight_layout()

    output_path = os.path.join(output_dir, f"provider_latency_ccdf.{FIGURE_FORMAT}")
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {output_path}")
    return output_path


def plot_rolling_p99_drift(
    df: pd.DataFrame, workload: str, output_dir: str, window_size: int = 100, title_suffix: str = ""
) -> str:
    """Plot rolling P99 over time to show non-stationarity.

    Figure 3: rolling_p99_drift.png
    Purpose: Demonstrate that latency distributions drift over time
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    workload_df = filter_by_workload(df, workload)
    providers = sorted(workload_df["provider"].unique())

    for provider in providers:
        indices, rolling_p99 = compute_rolling_percentile(
            workload_df, provider, percentile=99, window_size=window_size
        )
        if len(indices) > 0:
            # Convert index to hours (assuming ~45 sec between rounds)
            hours = indices / (3600 / 45)  # Approximate hours
            ax.plot(
                hours,
                rolling_p99,
                label=provider,
                color=get_provider_color(provider),
                linewidth=1.5,
                alpha=0.8,
            )

    ax.set_xlabel("Time (hours)", fontsize=14)
    ax.set_ylabel("Rolling P99 TTFT (ms)", fontsize=14)
    ax.set_title(f"Rolling P99 Latency Over Time (window={window_size}){title_suffix}", fontsize=16)
    ax.legend(loc="upper right", ncol=3, fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(output_dir, f"rolling_p99_drift.{FIGURE_FORMAT}")
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {output_path}")
    return output_path


def plot_provider_boxplot(
    df: pd.DataFrame, workload: str, output_dir: str, title_suffix: str = ""
) -> str:
    """Boxplot of TTFT by provider showing variance and outliers.

    Figure 4: provider_latency_boxplot.png
    Purpose: Show variance at same sequence length (motivation)
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    workload_df = filter_by_workload(df, workload)

    # Sort providers by median latency
    provider_medians = workload_df.groupby("provider")["ttft_ms"].median().sort_values()
    providers = provider_medians.index.tolist()

    # Prepare data for boxplot
    data = [workload_df[workload_df["provider"] == p]["ttft_ms"].values for p in providers]
    colors = [get_provider_color(p) for p in providers]

    bp = ax.boxplot(
        data,
        tick_labels=providers,
        patch_artist=True,
        showfliers=True,
        flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
    )

    # Color the boxes
    for patch, color in zip(bp["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Provider", fontsize=14)
    ax.set_ylabel("TTFT (ms)", fontsize=14)
    ax.set_title(f"Latency Distribution by Provider{title_suffix}", fontsize=16)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")

    # Add horizontal lines for SLO thresholds
    for slo, style, label in [(500, "--", "SLO=500ms"), (1000, ":", "SLO=1000ms")]:
        ax.axhline(y=slo, color="red", linestyle=style, alpha=0.5, label=label)

    plt.tight_layout()

    output_path = os.path.join(output_dir, f"provider_latency_boxplot.{FIGURE_FORMAT}")
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {output_path}")
    return output_path


def plot_provider_violin(
    df: pd.DataFrame, workload: str, output_dir: str, title_suffix: str = ""
) -> str:
    """Violin plot showing full distribution shape.

    Figure 5: provider_latency_violin.png
    Purpose: Better visualization of multi-modal distributions
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    workload_df = filter_by_workload(df, workload)

    # Sort providers by median latency
    provider_medians = workload_df.groupby("provider")["ttft_ms"].median().sort_values()
    providers = provider_medians.index.tolist()

    # Prepare data
    data = [workload_df[workload_df["provider"] == p]["ttft_ms"].values for p in providers]

    parts = ax.violinplot(
        data, positions=range(len(providers)), showmeans=True, showmedians=True, showextrema=True
    )

    # Color the violins
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(get_provider_color(providers[i]))
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(providers)))
    ax.set_xticklabels(providers, rotation=45, ha="right")
    ax.set_xlabel("Provider", fontsize=14)
    ax.set_ylabel("TTFT (ms)", fontsize=14)
    ax.set_title(f"Latency Distribution (Violin) by Provider{title_suffix}", fontsize=16)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    output_path = os.path.join(output_dir, f"provider_latency_violin.{FIGURE_FORMAT}")
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {output_path}")
    return output_path


def plot_percentile_heatmap(
    df: pd.DataFrame, workload: str, output_dir: str, title_suffix: str = ""
) -> str:
    """Heatmap of percentiles (P50, P90, P95, P99) by provider.

    Figure 6: percentile_heatmap.png
    Purpose: Quick comparison of tail behavior across providers
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    workload_df = filter_by_workload(df, workload)
    providers = sorted(workload_df["provider"].unique())
    percentiles = [50, 75, 90, 95, 99]

    # Compute percentile matrix
    data = []
    for provider in providers:
        provider_data = workload_df[workload_df["provider"] == provider]["ttft_ms"].values
        row = [np.percentile(provider_data, p) for p in percentiles]
        data.append(row)

    data = np.array(data)

    # Create heatmap
    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto")

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("TTFT (ms)", rotation=-90, va="bottom", fontsize=12)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(percentiles)))
    ax.set_yticks(np.arange(len(providers)))
    ax.set_xticklabels([f"P{p}" for p in percentiles])
    ax.set_yticklabels(providers)

    # Rotate the tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Add text annotations
    for i in range(len(providers)):
        for j in range(len(percentiles)):
            ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center", color="black", fontsize=9)

    ax.set_title(f"Latency Percentiles by Provider{title_suffix}", fontsize=16)
    ax.set_xlabel("Percentile", fontsize=14)
    ax.set_ylabel("Provider", fontsize=14)

    plt.tight_layout()

    output_path = os.path.join(output_dir, f"percentile_heatmap.{FIGURE_FORMAT}")
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {output_path}")
    return output_path


def plot_variance_by_round(
    df: pd.DataFrame,
    workload: str,
    output_dir: str,
    top_n_providers: int = 6,
    title_suffix: str = "",
) -> str:
    """Plot variance of latency across rounds for top N providers.

    Figure 7: variance_by_round.png
    Purpose: Show that variance is high even at same sequence length
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    workload_df = filter_by_workload(df, workload)

    # Get top N providers by request count
    provider_counts = workload_df["provider"].value_counts()
    top_providers = provider_counts.head(top_n_providers).index.tolist()

    for idx, provider in enumerate(top_providers):
        if idx >= len(axes):
            break

        ax = axes[idx]
        provider_df = workload_df[workload_df["provider"] == provider]

        # Group by round and compute stats
        round_stats = provider_df.groupby("round")["ttft_ms"].agg(["mean", "std", "min", "max"])

        rounds = round_stats.index.values
        means = round_stats["mean"].values
        stds = round_stats["std"].values

        # Plot mean with error bars (std)
        ax.fill_between(
            rounds, means - stds, means + stds, alpha=0.3, color=get_provider_color(provider)
        )
        ax.plot(rounds, means, color=get_provider_color(provider), linewidth=1.5)

        ax.set_xlabel("Round", fontsize=11)
        ax.set_ylabel("TTFT (ms)", fontsize=11)
        ax.set_title(f"{provider}", fontsize=12)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(len(top_providers), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"Latency Variance Across Rounds{title_suffix}", fontsize=16, y=1.02)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"variance_by_round.{FIGURE_FORMAT}")
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {output_path}")
    return output_path


def plot_provider_tier_comparison(
    df: pd.DataFrame, workload: str, output_dir: str, title_suffix: str = ""
) -> str:
    """Compare providers grouped by performance tier.

    Figure 8: provider_tier_comparison.png
    Purpose: Summarize provider groupings for paper narrative
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    workload_df = filter_by_workload(df, workload)

    # Compute stats for each provider
    provider_stats = []
    for provider in workload_df["provider"].unique():
        data = workload_df[workload_df["provider"] == provider]["ttft_ms"].values
        stats_dict = compute_percentiles(data)
        stats_dict["provider"] = provider
        provider_stats.append(stats_dict)

    stats_df = pd.DataFrame(provider_stats)
    stats_df = stats_df.sort_values("p50")

    # Left plot: P50 vs P99 scatter
    ax1 = axes[0]
    for _, row in stats_df.iterrows():
        ax1.scatter(
            row["p50"],
            row["p99"],
            c=get_provider_color(row["provider"]),
            s=150,
            label=row["provider"],
            edgecolors="black",
            linewidth=0.5,
        )
        ax1.annotate(
            row["provider"],
            (row["p50"], row["p99"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )

    ax1.set_xlabel("P50 TTFT (ms)", fontsize=14)
    ax1.set_ylabel("P99 TTFT (ms)", fontsize=14)
    ax1.set_title("P50 vs P99 Latency", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Add diagonal reference line (P99 = 2 * P50)
    max_p50 = stats_df["p50"].max()
    ax1.plot([0, max_p50], [0, 2 * max_p50], "k--", alpha=0.3, label="P99 = 2xP50")

    # Right plot: Coefficient of Variation (std/mean)
    ax2 = axes[1]
    stats_df["cv"] = stats_df["std"] / stats_df["mean"]
    stats_df_sorted = stats_df.sort_values("cv")

    colors = [get_provider_color(p) for p in stats_df_sorted["provider"]]
    ax2.barh(stats_df_sorted["provider"], stats_df_sorted["cv"], color=colors)

    ax2.set_xlabel("Coefficient of Variation (std/mean)", fontsize=14)
    ax2.set_ylabel("Provider", fontsize=14)
    ax2.set_title("Latency Variability by Provider", fontsize=14)
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    output_path = os.path.join(output_dir, f"provider_tier_comparison.{FIGURE_FORMAT}")
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {output_path}")
    return output_path


def plot_ttft_vs_e2e(
    df: pd.DataFrame, workload: str, output_dir: str, title_suffix: str = ""
) -> str:
    """Plot TTFT vs E2E breakdown to show prefill/queueing dominance.

    Figure: ttft_vs_e2e.png
    Purpose: Show that TTFT (prefill + queueing) dominates E2E latency
    Key insight: Most latency comes from TTFT, not decode time
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    workload_df = filter_by_workload(df, workload)

    # Filter for valid e2e data
    workload_df = workload_df[workload_df["e2e_ms"] > 0].copy()

    # Calculate TTFT ratio
    workload_df["ttft_ratio"] = workload_df["ttft_ms"] / workload_df["e2e_ms"]
    workload_df["decode_time"] = workload_df["e2e_ms"] - workload_df["ttft_ms"]

    providers = sorted(workload_df["provider"].unique())

    # Left: Scatter plot TTFT vs E2E
    ax1 = axes[0]
    for provider in providers:
        pdata = workload_df[workload_df["provider"] == provider]
        # Sample if too many points
        if len(pdata) > 500:
            pdata = pdata.sample(500)
        ax1.scatter(
            pdata["ttft_ms"],
            pdata["e2e_ms"],
            alpha=0.3,
            s=10,
            label=provider,
            color=get_provider_color(provider),
        )

    # Add diagonal line (E2E = TTFT, i.e., decode is 0)
    max_val = max(workload_df["e2e_ms"].max(), workload_df["ttft_ms"].max())
    ax1.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="E2E = TTFT")

    ax1.set_xlabel("TTFT (ms)", fontsize=14)
    ax1.set_ylabel("E2E Latency (ms)", fontsize=14)
    ax1.set_title("TTFT vs E2E Latency", fontsize=14)
    ax1.legend(loc="lower right", fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    # Right: TTFT Ratio distribution by provider
    ax2 = axes[1]

    # Compute median TTFT ratio per provider
    provider_ratios = (
        workload_df.groupby("provider")["ttft_ratio"].median().sort_values(ascending=False)
    )
    providers_sorted = provider_ratios.index.tolist()

    colors = [get_provider_color(p) for p in providers_sorted]
    bars = ax2.barh(providers_sorted, provider_ratios.values * 100, color=colors)

    # Add value labels
    for bar, ratio in zip(bars, provider_ratios.values, strict=False):
        ax2.text(
            ratio * 100 + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{ratio*100:.1f}%",
            va="center",
            fontsize=10,
        )

    ax2.set_xlabel("TTFT / E2E Ratio (%)", fontsize=14)
    ax2.set_ylabel("Provider", fontsize=14)
    ax2.set_title("TTFT Dominance (Median Ratio)", fontsize=14)
    ax2.set_xlim(0, 110)
    ax2.axvline(x=90, color="red", linestyle="--", alpha=0.5, label="90% threshold")
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    output_path = os.path.join(output_dir, f"ttft_vs_e2e.{FIGURE_FORMAT}")
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {output_path}")
    return output_path


def plot_slo_violation_rate(
    df: pd.DataFrame,
    workload: str,
    output_dir: str,
    slo_range: tuple[int, int] = (100, 5000),
    title_suffix: str = "",
) -> str:
    """Plot SLO violation rate vs SLO threshold for each provider.

    Figure: slo_violation_rate.png
    Purpose: Show how violation rate decreases as SLO relaxes
    Key insight: Different providers have different "knee points"
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    workload_df = filter_by_workload(df, workload)
    providers = sorted(workload_df["provider"].unique())

    # Generate SLO thresholds to test
    slo_thresholds = np.linspace(slo_range[0], slo_range[1], 100)

    for provider in providers:
        provider_data = workload_df[workload_df["provider"] == provider]["ttft_ms"].values

        # Calculate violation rate for each SLO
        violation_rates = []
        for slo in slo_thresholds:
            violation_rate = np.mean(provider_data > slo) * 100
            violation_rates.append(violation_rate)

        ax.plot(
            slo_thresholds,
            violation_rates,
            label=provider,
            color=get_provider_color(provider),
            linewidth=2,
        )

    ax.set_xlabel("SLO Threshold (ms)", fontsize=14)
    ax.set_ylabel("SLO Violation Rate (%)", fontsize=14)
    ax.set_title(f"SLO Violation Rate vs SLO Threshold{title_suffix}", fontsize=16)
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(slo_range)
    ax.set_ylim(0, 100)

    # Add horizontal reference lines
    for pct, style in [(10, "--"), (5, ":"), (1, "-.")]:
        ax.axhline(y=pct, color="gray", linestyle=style, alpha=0.5)
        ax.text(slo_range[1] * 0.98, pct + 1, f"{pct}%", ha="right", fontsize=9, color="gray")

    plt.tight_layout()

    output_path = os.path.join(output_dir, f"slo_violation_rate.{FIGURE_FORMAT}")
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {output_path}")
    return output_path


# Phase 1 Statistics Export


def export_summary_statistics(df: pd.DataFrame, workload: str, output_dir: str) -> str:
    """Export summary statistics to CSV."""
    workload_df = filter_by_workload(df, workload)

    rows = []
    for provider in sorted(workload_df["provider"].unique()):
        data = workload_df[workload_df["provider"] == provider]["ttft_ms"].values
        stats = compute_percentiles(data)
        stats["provider"] = provider
        rows.append(stats)

    stats_df = pd.DataFrame(rows)
    stats_df = stats_df[
        ["provider", "count", "mean", "std", "min", "p50", "p90", "p95", "p99", "max"]
    ]
    stats_df = stats_df.sort_values("p50")

    output_path = os.path.join(output_dir, "latency_summary_stats.csv")
    stats_df.to_csv(output_path, index=False, float_format="%.2f")

    print(f"  Saved: {output_path}")
    return output_path


def export_provider_ranking(df: pd.DataFrame, workload: str, output_dir: str) -> str:
    """Export provider ranking by different metrics."""
    workload_df = filter_by_workload(df, workload)

    rankings = []
    for provider in workload_df["provider"].unique():
        data = workload_df[workload_df["provider"] == provider]["ttft_ms"].values
        rankings.append(
            {
                "provider": provider,
                "p50": np.percentile(data, 50),
                "p99": np.percentile(data, 99),
                "cv": np.std(data) / np.mean(data),
            }
        )

    ranking_df = pd.DataFrame(rankings)

    # Add rankings
    ranking_df["rank_p50"] = ranking_df["p50"].rank().astype(int)
    ranking_df["rank_p99"] = ranking_df["p99"].rank().astype(int)
    ranking_df["rank_cv"] = ranking_df["cv"].rank().astype(int)
    ranking_df["rank_avg"] = (
        ranking_df["rank_p50"] + ranking_df["rank_p99"] + ranking_df["rank_cv"]
    ) / 3

    ranking_df = ranking_df.sort_values("rank_avg")

    output_path = os.path.join(output_dir, "provider_ranking.csv")
    ranking_df.to_csv(output_path, index=False, float_format="%.2f")

    print(f"  Saved: {output_path}")
    return output_path


# Phase 1 Main Function


def run_phase1(args):
    """Generate all Phase 1 figures and tables."""
    print("\n" + "=" * 60)
    print("Phase 1: Latency Profiling Analysis")
    print("=" * 60)

    # Setup output directory
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = str(PROJECT_ROOT / "experiment" / "results" / f"latency_phase1_{timestamp}")

    ensure_output_dir(args.output)
    print(f"Output directory: {args.output}")

    # Load data
    df = load_profiling_data(args.input)

    # Filter to base workload only (if workload column exists)
    workload = DEFAULT_WORKLOAD
    if "workload" in df.columns:
        print(f"\nFiltering to '{workload}' workload...")
        df_filtered = filter_by_workload(df, workload)
        print(f"  Filtered records: {len(df_filtered):,}")
    else:
        print("\nNo 'workload' column found, using all data")
        df_filtered = df

    print(f"\n{'='*60}")
    print("Generating figures...")
    print("=" * 60)

    # Core figures (always generated) - aligned with latency_experiment_plan.md Phase 1
    print("\n[Core Figures]")
    plot_cdf_all_providers(df_filtered, workload, args.output, args.title_suffix)
    plot_ccdf_all_providers(df_filtered, workload, args.output, title_suffix=args.title_suffix)
    plot_rolling_p99_drift(df_filtered, workload, args.output, title_suffix=args.title_suffix)
    plot_ttft_vs_e2e(df_filtered, workload, args.output, title_suffix=args.title_suffix)
    plot_slo_violation_rate(df_filtered, workload, args.output, title_suffix=args.title_suffix)
    plot_variance_by_round(df_filtered, workload, args.output, title_suffix=args.title_suffix)

    # Export statistics (always generated)
    print("\n[Statistics]")
    export_summary_statistics(df_filtered, workload, args.output)
    export_provider_ranking(df_filtered, workload, args.output)

    # Optional figures (only if --all flag is set)
    if args.all_figures:
        print("\n[Additional Figures]")
        plot_provider_boxplot(df_filtered, workload, args.output, title_suffix=args.title_suffix)
        plot_provider_violin(df_filtered, workload, args.output, title_suffix=args.title_suffix)
        plot_percentile_heatmap(df_filtered, workload, args.output, title_suffix=args.title_suffix)
        plot_provider_tier_comparison(
            df_filtered, workload, args.output, title_suffix=args.title_suffix
        )

    print(f"\n{'='*60}")
    print("Phase 1 complete!")
    print(f"All outputs saved to: {args.output}")
    print("=" * 60)


# =============================================================================
# Phase 2: Offline ILP Analysis (Placeholder)
# =============================================================================


def run_phase2(args):
    """Generate Phase 2 figures: Offline ILP with latency constraints."""
    print("\n" + "=" * 60)
    print("Phase 2: Offline ILP Analysis")
    print("=" * 60)
    print("TODO: Implement after ILP extension is complete")
    # Future implementation:
    # - Load offline ILP results
    # - Plot cost vs SLO curves
    # - Plot Pareto frontier (theoretical)


# =============================================================================
# Phase 3: Online Routing Analysis (Placeholder)
# =============================================================================


def run_phase3(args):
    """Generate Phase 3 figures: Online latency-aware routing."""
    print("\n" + "=" * 60)
    print("Phase 3: Online Routing Analysis")
    print("=" * 60)
    print("TODO: Implement after online router integration")
    # Future implementation:
    # - Load online routing logs
    # - Plot empirical cost vs P99
    # - Plot routing decisions over time


# =============================================================================
# Phase 4: Smart Hedging Analysis (Placeholder)
# =============================================================================


def run_phase4(args):
    """Generate Phase 4 figures: Smart hedging analysis."""
    print("\n" + "=" * 60)
    print("Phase 4: Smart Hedging Analysis")
    print("=" * 60)
    print("TODO: Implement after hedging implementation")
    # Future implementation:
    # - Load hedging experiment results
    # - Plot hedge rate vs SLO
    # - Plot P99 reduction from hedging
    # - Plot cost overhead


# =============================================================================
# Phase 5: OpenRouter Evaluation (Placeholder)
# =============================================================================


def run_phase5(args):
    """Generate Phase 5 figures: OpenRouter evaluation."""
    print("\n" + "=" * 60)
    print("Phase 5: OpenRouter Evaluation")
    print("=" * 60)
    print("TODO: Implement after evaluation experiments")
    # Future implementation:
    # - Load evaluation results
    # - Plot policy comparison (cost vs P99 Pareto)
    # - Plot SLO violation rates
    # - Plot head-to-head vs OpenRouter


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point for latency visualization CLI."""
    parser = argparse.ArgumentParser(
        description="Latency Experiment Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Phase 1: Profiling analysis
    python plot_latency.py phase1 --input data/latency_profile.csv

    # Phase 1 with custom output
    python plot_latency.py phase1 -i data/latency_profile.csv -o results/phase1/

    # Phase 1 for specific workloads only
    python plot_latency.py phase1 -i data/latency_profile.csv -w base prefill
        """,
    )

    # Global options
    parser.add_argument(
        "--format",
        "-f",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output figure format (default: png)",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="phase", help="Experiment phase")

    # Phase 1 subcommand
    phase1_parser = subparsers.add_parser("phase1", help="Latency profiling analysis")
    phase1_parser.add_argument("--input", "-i", required=True, help="Path to profiling CSV")
    phase1_parser.add_argument("--output", "-o", default=None, help="Output directory")
    phase1_parser.add_argument("--title-suffix", "-t", default="", help="Title suffix")
    phase1_parser.add_argument(
        "--all",
        "-a",
        dest="all_figures",
        action="store_true",
        help="Generate all figures (including boxplot, violin, heatmap, etc.)",
    )

    # Phase 2 subcommand
    phase2_parser = subparsers.add_parser("phase2", help="Offline ILP analysis")
    phase2_parser.add_argument("--input", "-i", required=True, help="Path to ILP results")
    phase2_parser.add_argument("--output", "-o", default=None, help="Output directory")

    # Phase 3 subcommand
    phase3_parser = subparsers.add_parser("phase3", help="Online routing analysis")
    phase3_parser.add_argument("--input", "-i", required=True, help="Path to routing logs")
    phase3_parser.add_argument("--output", "-o", default=None, help="Output directory")

    # Phase 4 subcommand
    phase4_parser = subparsers.add_parser("phase4", help="Smart hedging analysis")
    phase4_parser.add_argument("--input", "-i", required=True, help="Path to hedging results")
    phase4_parser.add_argument("--output", "-o", default=None, help="Output directory")

    # Phase 5 subcommand
    phase5_parser = subparsers.add_parser("phase5", help="OpenRouter evaluation")
    phase5_parser.add_argument("--input", "-i", required=True, help="Path to evaluation results")
    phase5_parser.add_argument("--output", "-o", default=None, help="Output directory")

    args = parser.parse_args()

    # Set global figure format
    global FIGURE_FORMAT
    FIGURE_FORMAT = args.format

    # Route to appropriate phase
    if args.phase == "phase1":
        run_phase1(args)
    elif args.phase == "phase2":
        run_phase2(args)
    elif args.phase == "phase3":
        run_phase3(args)
    elif args.phase == "phase4":
        run_phase4(args)
    elif args.phase == "phase5":
        run_phase5(args)
    else:
        parser.print_help()
        print("\nError: Please specify a phase (phase1, phase2, phase3, phase4, phase5)")
        sys.exit(1)


if __name__ == "__main__":
    main()
