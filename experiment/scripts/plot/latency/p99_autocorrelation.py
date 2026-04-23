#!/usr/bin/env python3
"""P99 vs P50 temporal autocorrelation analysis.

Tests whether online probing is effective for predicting provider latency
at different percentiles. Hypothesis (Juncheng):
  - P50 shows strong temporal autocorrelation -> probing useful for P50
  - P99 shows weak temporal autocorrelation -> probing NOT useful for P99

Generates:
  1. ACF comparison: P50 vs P90 vs P99 autocorrelation at increasing lags
  2. Lag-1 scatter plots: percentile(t) vs percentile(t+1)
  3. Time series: P50 and P99 over time showing smoothness vs spikiness

Usage:
    source .venv/bin/activate
    python -m experiment.scripts.plot.latency.p99_autocorrelation \
        --input experiment/results/phase5_qwen3_7d_clean/run_20260410_171625/evaluation_log.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_window(
    path: Path,
    window_min: int = 5,
    min_samples: int = 10,
) -> pd.DataFrame:
    """Load evaluation log and compute per-provider windowed percentiles."""
    df = pd.read_csv(path)
    df["timestamp"] = df["timestamp"].astype(float)
    df["ttft_ms"] = pd.to_numeric(df["ttft_ms"], errors="coerce")

    # Keep only successful requests with valid TTFT
    df = df[(df["status"] == "success") & (df["ttft_ms"].notna())].copy()

    window_sec = window_min * 60
    df["window_start"] = np.floor(df["timestamp"] / window_sec) * window_sec
    df["window_dt"] = pd.to_datetime(df["window_start"], unit="s")

    rows: list[dict] = []
    for (window_dt, provider), group in df.groupby(
        ["window_dt", "actual_provider"]
    ):
        if len(group) < min_samples:
            continue
        ttft = group["ttft_ms"].to_numpy()
        rows.append(
            {
                "window_dt": window_dt,
                "provider": provider,
                "count": len(group),
                "p50": float(np.percentile(ttft, 50)),
                "p90": float(np.percentile(ttft, 90)),
                "p95": float(np.percentile(ttft, 95)),
                "p99": float(np.percentile(ttft, 99)),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ACF computation
# ---------------------------------------------------------------------------

def compute_acf(series: pd.Series, max_lag: int = 30) -> np.ndarray:
    """Compute autocorrelation function for a 1-D time series.

    Returns array of length max_lag with ACF at lags 1..max_lag.
    """
    vals = series.to_numpy().astype(float)
    n = len(vals)
    if n < max_lag + 2:
        return np.full(max_lag, np.nan)

    mean = np.mean(vals)
    var = np.var(vals, ddof=0)
    if var < 1e-12:
        return np.full(max_lag, np.nan)

    acf = np.empty(max_lag)
    for lag in range(1, max_lag + 1):
        if lag >= n:
            acf[lag - 1] = np.nan
        else:
            acf[lag - 1] = np.mean(
                (vals[:-lag] - mean) * (vals[lag:] - mean)
            ) / var
    return acf


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_acf_comparison(
    windowed: pd.DataFrame,
    providers: list[str],
    window_min: int,
    output_dir: Path,
    max_lag: int = 30,
) -> None:
    """ACF at increasing lags for P50 vs P90 vs P99, one subplot per provider."""
    lags_min = np.arange(1, max_lag + 1) * window_min

    n_prov = len(providers)
    if n_prov == 0:
        print("  [SKIP] No providers with enough windows for ACF plot.")
        return
    cols = min(n_prov, 3)
    rows_grid = (n_prov + cols - 1) // cols
    fig, axes = plt.subplots(
        rows_grid, cols,
        figsize=(5.5 * cols, 4 * rows_grid),
        sharex=True, sharey=True, squeeze=False,
    )

    metrics = [
        ("p50", "#2196F3", "P50"),
        ("p90", "#FF9800", "P90"),
        ("p99", "#F44336", "P99"),
    ]

    for idx, provider in enumerate(providers):
        ax = axes[idx // cols][idx % cols]
        pdata = (
            windowed[windowed["provider"] == provider]
            .sort_values("window_dt")
        )

        for metric, color, label in metrics:
            acf = compute_acf(pdata[metric], max_lag=max_lag)
            ax.plot(
                lags_min, acf,
                marker="o", markersize=3, linewidth=1.6,
                color=color, label=label, alpha=0.85,
            )

        # 95 % confidence band for white noise
        n_win = len(pdata)
        if n_win > 0:
            ci = 1.96 / np.sqrt(n_win)
            ax.axhspan(-ci, ci, color="gray", alpha=0.12)

        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title(f"{provider} ({n_win} windows)", fontsize=11)
        ax.set_ylim(-0.3, 1.0)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_prov, rows_grid * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    # Shared labels and legend
    for r in range(rows_grid):
        axes[r][0].set_ylabel("Autocorrelation")
    for c in range(cols):
        axes[-1][c].set_xlabel(f"Lag (minutes)")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", ncol=len(metrics),
        frameon=False, fontsize=11,
    )
    fig.suptitle(
        f"Temporal Autocorrelation: P50 vs P90 vs P99 "
        f"(window = {window_min} min)\n"
        "Higher ACF = more predictable from recent probing",
        fontsize=13, y=1.04,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(
        output_dir / "acf_p50_vs_p99.png", dpi=200, bbox_inches="tight",
    )
    plt.close(fig)


def plot_lag1_scatter(
    windowed: pd.DataFrame,
    providers: list[str],
    output_dir: Path,
) -> None:
    """Scatter of percentile(t) vs percentile(t+1) for P50 and P99."""
    n_prov = len(providers)
    fig, axes = plt.subplots(
        n_prov, 2, figsize=(10, 3.8 * n_prov), squeeze=False,
    )

    pairs = [("p50", "P50", "#2196F3"), ("p99", "P99", "#F44336")]

    for i, provider in enumerate(providers):
        pdata = (
            windowed[windowed["provider"] == provider]
            .sort_values("window_dt")
        )

        for j, (metric, title, color) in enumerate(pairs):
            ax = axes[i][j]
            vals = pdata[metric].to_numpy()
            if len(vals) < 3:
                ax.set_title(f"{provider} {title}: insufficient data")
                continue

            x, y = vals[:-1], vals[1:]
            ax.scatter(x, y, alpha=0.25, s=8, color=color, edgecolors="none")

            corr = np.corrcoef(x, y)[0, 1]
            ax.set_title(f"{provider} {title}:  r = {corr:.3f}", fontsize=11)

            lo = min(x.min(), y.min()) * 0.9
            hi = max(x.max(), y.max()) * 1.05
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.7, alpha=0.4)
            ax.set_xlabel(f"{title}(t) ms")
            ax.set_ylabel(f"{title}(t+1) ms")
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Lag-1 Scatter: Current Window vs Next Window\n"
        "Tight diagonal cluster = high predictability",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "lag1_scatter_p50_vs_p99.png",
        dpi=200, bbox_inches="tight",
    )
    plt.close(fig)


def plot_timeseries(
    windowed: pd.DataFrame,
    providers: list[str],
    output_dir: Path,
) -> None:
    """P50 and P99 over time for each provider."""
    n_prov = len(providers)
    fig, axes = plt.subplots(
        n_prov, 1, figsize=(14, 3 * n_prov), sharex=True, squeeze=False,
    )

    for i, provider in enumerate(providers):
        ax = axes[i][0]
        pdata = (
            windowed[windowed["provider"] == provider]
            .sort_values("window_dt")
        )

        ax.plot(
            pdata["window_dt"], pdata["p50"],
            color="#2196F3", linewidth=1.0, label="P50", alpha=0.85,
        )
        ax.plot(
            pdata["window_dt"], pdata["p99"],
            color="#F44336", linewidth=1.0, label="P99", alpha=0.85,
        )
        ax.fill_between(
            pdata["window_dt"], pdata["p50"], pdata["p99"],
            color="#FFE0B2", alpha=0.25,
        )
        ax.set_ylabel("TTFT (ms)")
        ax.set_title(provider, fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

    fig.suptitle(
        "P50 vs P99 Over Time\n"
        "Hypothesis: P50 is smooth (predictable), P99 is spiky (not predictable)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "timeseries_p50_vs_p99.png",
        dpi=200, bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="P99 temporal autocorrelation analysis",
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to evaluation_log.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--window-min", type=int, default=5,
        help="Window size in minutes (default: 5)",
    )
    parser.add_argument(
        "--min-samples", type=int, default=10,
        help="Min requests per provider per window (default: 10)",
    )
    parser.add_argument(
        "--top-n", type=int, default=6,
        help="Top N providers by request count (default: 6)",
    )
    parser.add_argument(
        "--max-lag", type=int, default=30,
        help="Max lag for ACF in number of windows (default: 30)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (
        args.input.parent / "analysis" / "autocorrelation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input} ...")
    windowed = load_and_window(
        args.input,
        window_min=args.window_min,
        min_samples=args.min_samples,
    )
    print(f"  {len(windowed)} provider-window observations")

    # Select top providers with enough windows
    provider_totals = (
        windowed.groupby("provider")["count"]
        .sum()
        .sort_values(ascending=False)
    )
    providers: list[str] = []
    for prov in provider_totals.index:
        n_win = len(windowed[windowed["provider"] == prov])
        if n_win >= args.max_lag + 5:
            providers.append(prov)
        if len(providers) >= args.top_n:
            break

    print(f"  Selected providers: {providers}")

    print("\n1/3  ACF comparison ...")
    plot_acf_comparison(
        windowed, providers, args.window_min, output_dir,
        max_lag=args.max_lag,
    )

    print("2/3  Lag-1 scatter ...")
    plot_lag1_scatter(windowed, providers, output_dir)

    print("3/3  Time series ...")
    plot_timeseries(windowed, providers, output_dir)

    # Summary table
    print(f"\n{'=' * 80}")
    print(
        f"Autocorrelation Summary  "
        f"(window = {args.window_min} min, "
        f"lag-1 = {args.window_min} min, "
        f"lag-3 = {args.window_min * 3} min)"
    )
    print(f"{'=' * 80}")
    header = (
        f"{'Provider':<15} {'Windows':>8}  "
        f"{'P50 ACF(1)':>11} {'P50 ACF(3)':>11}  "
        f"{'P99 ACF(1)':>11} {'P99 ACF(3)':>11}  "
        f"{'P50 r':>7} {'P99 r':>7}"
    )
    print(header)
    print("-" * len(header))

    for provider in providers:
        pdata = (
            windowed[windowed["provider"] == provider]
            .sort_values("window_dt")
        )
        acf50 = compute_acf(pdata["p50"], max_lag=args.max_lag)
        acf99 = compute_acf(pdata["p99"], max_lag=args.max_lag)

        # Lag-1 Pearson r
        v50 = pdata["p50"].to_numpy()
        v99 = pdata["p99"].to_numpy()
        r50 = np.corrcoef(v50[:-1], v50[1:])[0, 1] if len(v50) > 2 else np.nan
        r99 = np.corrcoef(v99[:-1], v99[1:])[0, 1] if len(v99) > 2 else np.nan

        print(
            f"{provider:<15} {len(pdata):>8}  "
            f"{acf50[0]:>11.3f} {acf50[2]:>11.3f}  "
            f"{acf99[0]:>11.3f} {acf99[2]:>11.3f}  "
            f"{r50:>7.3f} {r99:>7.3f}"
        )

    print(f"\nOutputs saved to {output_dir}")


if __name__ == "__main__":
    main()
