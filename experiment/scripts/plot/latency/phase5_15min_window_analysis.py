#!/usr/bin/env python3
"""Generate 15-minute window analysis for Phase 5 evaluation logs.

This script focuses on the mechanism-analysis view requested during the
RouteWise discussion:

1. Per-provider TTFT percentiles (p50/p90/p95/p99) over 15-minute windows.
2. Per-policy provider share over time.
3. A window summary that highlights when Alibaba receives high lp_mix traffic.

Important notes:

1. The current harness does not persist `last_lp_weights` for every LP update
   in the CSV log. Therefore, for lp_mix we use the realized provider share
   within each 15-minute window as the observable proxy for the mixture
   coefficients.
2. Provider percentiles can be computed either from all successful requests
   (provider-state overview) or from a specific policy only (decision-aligned
   view). We export both variants for clarity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "/Users/realtmxi/Desktop/NSDI2027_RouteWise/experiment/results/"
    "phase5_qwen3_7d_clean/run_20260410_171624/snapshots/snap_20260411_171625/"
    "evaluation_log.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "/Users/realtmxi/Desktop/NSDI2027_RouteWise/experiment/results/"
    "phase5_qwen3_7d_clean/run_20260410_171624/analysis/15min_windows"
)
WINDOW_SEC = 15 * 60
DEFAULT_FOCUS_PROVIDERS = ["WandB", "Alibaba", "Google"]


def build_color_map(providers: list[str]) -> dict[str, tuple[float, float, float, float]]:
    """Build a stable provider-to-color mapping."""
    cmap = plt.get_cmap("tab10")
    return {
        provider: cmap(idx % 10)
        for idx, provider in enumerate(providers)
    }


def _floor_to_window(series: pd.Series, window_sec: int) -> pd.Series:
    """Floor Unix timestamps to a fixed-size window."""
    return (np.floor(series.astype(float) / window_sec) * window_sec).astype(float)


def load_log(path: Path) -> pd.DataFrame:
    """Load and normalize the Phase 5 evaluation log."""
    df = pd.read_csv(path)
    df["timestamp"] = df["timestamp"].astype(float)
    df["ttft_ms"] = pd.to_numeric(df["ttft_ms"], errors="coerce")
    df["window_start"] = _floor_to_window(df["timestamp"], WINDOW_SEC)
    df["window_dt"] = pd.to_datetime(df["window_start"], unit="s")
    df["status"] = df["status"].fillna("unknown")
    df["actual_provider"] = df["actual_provider"].fillna("unknown")
    df["policy"] = df["policy"].fillna("unknown")
    return df


def select_top_providers(
    df: pd.DataFrame,
    policy: str = "lp_mix",
    top_n: int = 4,
) -> list[str]:
    """Select the most-used providers for a policy."""
    counts = (
        df[df["policy"] == policy]["actual_provider"]
        .value_counts()
        .head(top_n)
    )
    return counts.index.tolist()


def compute_provider_percentiles(
    df: pd.DataFrame,
    providers: list[str],
    policy: str | None = None,
) -> pd.DataFrame:
    """Compute provider TTFT percentiles over 15-minute windows."""
    success = df[(df["status"] == "success") & (df["actual_provider"].isin(providers))].copy()
    if policy is not None:
        success = success[success["policy"] == policy].copy()
    if success.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for (window_dt, provider), group in success.groupby(["window_dt", "actual_provider"]):
        if len(group) < 5:
            continue
        ttft = group["ttft_ms"].to_numpy()
        rows.append(
            {
                "window_dt": window_dt,
                "provider": provider,
                "count": len(group),
                "p50_ms": float(np.percentile(ttft, 50)),
                "p90_ms": float(np.percentile(ttft, 90)),
                "p95_ms": float(np.percentile(ttft, 95)),
                "p99_ms": float(np.percentile(ttft, 99)),
            }
        )
    return pd.DataFrame(rows).sort_values(["window_dt", "provider"])


def compute_policy_provider_shares(
    df: pd.DataFrame,
    policies: list[str],
    providers: list[str],
) -> pd.DataFrame:
    """Compute realized provider shares for each policy and window."""
    filtered = df[df["policy"].isin(policies)].copy()
    if filtered.empty:
        return pd.DataFrame()

    grouped = (
        filtered.groupby(["policy", "window_dt", "actual_provider"])
        .size()
        .rename("count")
        .reset_index()
    )
    totals = (
        filtered.groupby(["policy", "window_dt"])
        .size()
        .rename("window_total")
        .reset_index()
    )
    grouped = grouped.merge(totals, on=["policy", "window_dt"], how="left")
    grouped["share"] = grouped["count"] / grouped["window_total"]
    grouped["provider_group"] = np.where(
        grouped["actual_provider"].isin(providers),
        grouped["actual_provider"],
        "Other",
    )
    grouped = (
        grouped.groupby(["policy", "window_dt", "provider_group"], as_index=False)[["count"]]
        .sum()
    )
    grouped = grouped.merge(totals, on=["policy", "window_dt"], how="left")
    grouped["share"] = grouped["count"] / grouped["window_total"]
    return grouped.sort_values(["policy", "window_dt", "provider_group"])


def build_alibaba_window_summary(
    df: pd.DataFrame,
    provider_percentiles: pd.DataFrame,
    threshold: float = 0.2,
) -> pd.DataFrame:
    """Summarize windows where lp_mix sends meaningful share to Alibaba."""
    lp = df[df["policy"] == "lp_mix"].copy()
    if lp.empty:
        return pd.DataFrame()

    counts = (
        lp.groupby(["window_dt", "actual_provider"])
        .size()
        .rename("count")
        .reset_index()
    )
    totals = lp.groupby("window_dt").size().rename("window_total").reset_index()
    counts = counts.merge(totals, on="window_dt", how="left")
    counts["share"] = counts["count"] / counts["window_total"]

    pivot = (
        counts.pivot_table(
            index="window_dt",
            columns="actual_provider",
            values="share",
            fill_value=0.0,
        )
        .reset_index()
    )
    for provider in ["Alibaba", "WandB", "Google"]:
        if provider not in pivot.columns:
            pivot[provider] = 0.0

    perc = provider_percentiles[
        provider_percentiles["provider"].isin(["Alibaba", "WandB", "Google"])
    ].copy()
    perc = perc.pivot_table(
        index="window_dt",
        columns="provider",
        values=["p50_ms", "p99_ms", "count"],
    )
    perc.columns = [
        f"{metric}_{provider}"
        for metric, provider in perc.columns.to_flat_index()
    ]
    perc = perc.reset_index()

    summary = pivot.merge(perc, on="window_dt", how="left")
    summary = summary[summary["Alibaba"] >= threshold].copy()
    if summary.empty:
        return summary
    return summary.sort_values("Alibaba", ascending=False)


def plot_provider_percentiles(
    provider_percentiles: pd.DataFrame,
    output_path: Path,
    title_suffix: str,
    color_map: dict[str, tuple[float, float, float, float]],
) -> None:
    """Plot p50/p90/p95/p99 time series for selected providers."""
    metrics = ["p50_ms", "p90_ms", "p95_ms", "p99_ms"]
    title_map = {
        "p50_ms": "P50",
        "p90_ms": "P90",
        "p95_ms": "P95",
        "p99_ms": "P99",
    }
    providers = sorted(provider_percentiles["provider"].unique().tolist())
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.ravel()

    for axis, metric in zip(axes, metrics):
        for provider in providers:
            data = provider_percentiles[provider_percentiles["provider"] == provider]
            axis.plot(
                data["window_dt"],
                data[metric],
                marker="o",
                linewidth=1.8,
                markersize=3,
                label=provider,
                color=color_map.get(provider),
            )
        axis.set_title(f"{title_map[metric]} TTFT by 15-Minute Window")
        axis.set_ylabel("TTFT (ms)")
        axis.grid(True, alpha=0.3)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), frameon=False)
    fig.suptitle(f"Top Provider TTFT Percentiles Over Time ({title_suffix})", y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_policy_provider_shares(
    provider_shares: pd.DataFrame,
    policies: list[str],
    provider_order: list[str],
    output_path: Path,
    color_map: dict[str, tuple[float, float, float, float]],
) -> None:
    """Plot provider share over time for each policy."""
    fig, axes = plt.subplots(
        len(policies),
        1,
        figsize=(16, 3.6 * len(policies)),
        sharex=True,
    )
    if len(policies) == 1:
        axes = [axes]

    fill_order = provider_order + ["Other"]
    for axis, policy in zip(axes, policies):
        data = provider_shares[provider_shares["policy"] == policy]
        pivot = (
            data.pivot_table(
                index="window_dt",
                columns="provider_group",
                values="share",
                fill_value=0.0,
            )
            .reindex(columns=fill_order, fill_value=0.0)
            .sort_index()
        )
        if pivot.empty:
            continue
        colors = [color_map.get(col, "#cccccc") for col in pivot.columns]
        axis.stackplot(
            pivot.index,
            [pivot[col].to_numpy() for col in pivot.columns],
            labels=pivot.columns.tolist(),
            alpha=0.85,
            colors=colors,
        )
        axis.set_ylim(0, 1)
        axis.set_ylabel("Share")
        axis.set_title(f"{policy}: Realized Provider Share (15-Minute Windows)")
        axis.grid(True, alpha=0.25)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=min(len(labels), 6),
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "Provider Share Over Time\n"
        "(For lp_mix, this is the observable proxy for the mixture coefficients)",
        y=0.93,
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_policy_provider_shares_focus(
    provider_shares: pd.DataFrame,
    policies: list[str],
    provider_order: list[str],
    output_path: Path,
    color_map: dict[str, tuple[float, float, float, float]],
) -> None:
    """Plot focused provider shares for the main explanatory policies."""
    fig, axes = plt.subplots(
        len(policies),
        1,
        figsize=(16, 3.2 * len(policies)),
        sharex=True,
    )
    if len(policies) == 1:
        axes = [axes]

    for axis, policy in zip(axes, policies):
        data = provider_shares[provider_shares["policy"] == policy]
        pivot = (
            data.pivot_table(
                index="window_dt",
                columns="provider_group",
                values="share",
                fill_value=0.0,
            )
            .reindex(columns=provider_order, fill_value=0.0)
            .sort_index()
        )
        if pivot.empty:
            continue

        for provider in provider_order:
            axis.plot(
                pivot.index,
                pivot[provider].to_numpy(),
                label=provider,
                color=color_map.get(provider, "#666666"),
                linewidth=1.8 if provider in {"WandB", "Alibaba"} else 1.2,
                alpha=0.95 if provider in {"WandB", "Alibaba"} else 0.75,
            )

        axis.set_ylim(0, 1)
        axis.set_ylabel("Share")
        axis.set_title(f"{policy}: Focused Provider Share (15-Minute Windows)")
        axis.grid(True, alpha=0.25)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=min(len(labels), 6),
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "Focused Provider Share Over Time\n"
        "(Highlights WandB vs Alibaba behavior for diagnosis)",
        y=0.92,
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_autocorrelation(
    provider_percentiles: pd.DataFrame,
    providers: list[str],
    metrics: list[str],
    max_lag: int = 12,
) -> pd.DataFrame:
    """Compute lag autocorrelation for provider percentile time series."""
    rows: list[dict] = []
    for provider in providers:
        data = (
            provider_percentiles[provider_percentiles["provider"] == provider]
            .sort_values("window_dt")
        )
        if data.empty:
            continue
        for metric in metrics:
            series = data[metric].reset_index(drop=True)
            if len(series) < 3:
                continue
            for lag in range(1, min(max_lag, len(series) - 1) + 1):
                value = series.autocorr(lag=lag)
                rows.append(
                    {
                        "provider": provider,
                        "metric": metric,
                        "lag_windows": lag,
                        "lag_minutes": lag * 15,
                        "autocorr": float(value) if pd.notna(value) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def plot_autocorrelation(
    autocorr_df: pd.DataFrame,
    output_path: Path,
    providers: list[str],
    metrics: list[str],
    color_map: dict[str, tuple[float, float, float, float]],
    title: str,
) -> None:
    """Plot lag autocorrelation curves for selected metrics/providers."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.5 * len(metrics), 4.5), sharey=True)
    if len(metrics) == 1:
        axes = [axes]

    for axis, metric in zip(axes, metrics):
        metric_df = autocorr_df[autocorr_df["metric"] == metric]
        for provider in providers:
            data = metric_df[metric_df["provider"] == provider].sort_values("lag_windows")
            if data.empty:
                continue
            axis.plot(
                data["lag_minutes"],
                data["autocorr"],
                marker="o",
                linewidth=1.8,
                markersize=4,
                label=provider,
                color=color_map.get(provider),
            )
        axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.4)
        axis.set_title(f"{metric.replace('_ms', '').upper()} Autocorrelation")
        axis.set_xlabel("Lag (minutes)")
        axis.grid(True, alpha=0.3)

    axes[0].set_ylabel("Autocorrelation")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), frameon=False)
    fig.suptitle(title, y=0.99, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 15-minute window analysis")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to evaluation_log.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for output plots and CSVs")
    parser.add_argument("--top-n-providers", type=int, default=4, help="Top lp_mix providers to track explicitly")
    parser.add_argument(
        "--focus-providers",
        nargs="*",
        default=DEFAULT_FOCUS_PROVIDERS,
        help="Providers that must always be included in analysis outputs",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_log(args.input)
    top_providers = select_top_providers(df, policy="lp_mix", top_n=args.top_n_providers)
    tracked_providers = []
    for provider in top_providers + list(args.focus_providers):
        if provider not in tracked_providers:
            tracked_providers.append(provider)
    color_map = build_color_map(tracked_providers + ["Other"])

    provider_percentiles_all = compute_provider_percentiles(df, tracked_providers)
    provider_percentiles_lp_mix = compute_provider_percentiles(df, tracked_providers, policy="lp_mix")
    autocorr_all = compute_autocorrelation(
        provider_percentiles_all,
        providers=[provider for provider in ["WandB", "Alibaba", "Google"] if provider in tracked_providers],
        metrics=["p50_ms", "p99_ms"],
    )
    autocorr_lp_mix = compute_autocorrelation(
        provider_percentiles_lp_mix,
        providers=[provider for provider in ["WandB", "Alibaba", "Google"] if provider in tracked_providers],
        metrics=["p50_ms", "p99_ms"],
    )
    policies = ["lp_mix", "smart_hedge", "sort_latency", "cheapest_fixed"]
    provider_shares = compute_policy_provider_shares(df, policies=policies, providers=tracked_providers)
    alibaba_summary = build_alibaba_window_summary(df, provider_percentiles_lp_mix)

    provider_percentiles_all.to_csv(output_dir / "provider_percentiles_15min_all_policies.csv", index=False)
    provider_percentiles_lp_mix.to_csv(output_dir / "provider_percentiles_15min_lp_mix_only.csv", index=False)
    autocorr_all.to_csv(output_dir / "provider_autocorr_15min_all_policies.csv", index=False)
    autocorr_lp_mix.to_csv(output_dir / "provider_autocorr_15min_lp_mix_only.csv", index=False)
    provider_shares.to_csv(output_dir / "policy_provider_shares_15min.csv", index=False)
    alibaba_summary.to_csv(output_dir / "lp_mix_alibaba_windows.csv", index=False)

    if not provider_percentiles_all.empty:
        plot_provider_percentiles(
            provider_percentiles_all,
            output_dir / "provider_ttft_percentiles_15min_all_policies.png",
            title_suffix="all policies",
            color_map=color_map,
        )
    if not provider_percentiles_lp_mix.empty:
        plot_provider_percentiles(
            provider_percentiles_lp_mix,
            output_dir / "provider_ttft_percentiles_15min_lp_mix_only.png",
            title_suffix="lp_mix only",
            color_map=color_map,
        )
    if not provider_shares.empty:
        plot_policy_provider_shares(
            provider_shares,
            policies=policies,
            provider_order=tracked_providers,
            output_path=output_dir / "policy_provider_shares_15min.png",
            color_map=color_map,
        )
        focus_providers = [
            provider
            for provider in ["WandB", "Alibaba", "Google", "Other"]
            if provider in set(provider_shares["provider_group"])
        ]
        plot_policy_provider_shares_focus(
            provider_shares,
            policies=["lp_mix", "smart_hedge"],
            provider_order=focus_providers,
            output_path=output_dir / "policy_provider_shares_15min_focus.png",
            color_map=color_map,
        )
    if not autocorr_all.empty:
        plot_autocorrelation(
            autocorr_all,
            output_dir / "provider_autocorr_15min_all_policies.png",
            providers=[provider for provider in ["WandB", "Alibaba", "Google"] if provider in tracked_providers],
            metrics=["p50_ms", "p99_ms"],
            color_map=color_map,
            title="Provider Autocorrelation Over 15-Minute Windows (all policies)",
        )
    if not autocorr_lp_mix.empty:
        plot_autocorrelation(
            autocorr_lp_mix,
            output_dir / "provider_autocorr_15min_lp_mix_only.png",
            providers=[provider for provider in ["WandB", "Alibaba", "Google"] if provider in tracked_providers],
            metrics=["p50_ms", "p99_ms"],
            color_map=color_map,
            title="Provider Autocorrelation Over 15-Minute Windows (lp_mix only)",
        )

    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Top lp_mix providers: {top_providers}")
    print(f"Tracked providers: {tracked_providers}")
    print(f"Saved outputs to {output_dir}")
    if not alibaba_summary.empty:
        print("\nTop windows with high Alibaba share in lp_mix:")
        display_cols = [
            "window_dt",
            "Alibaba",
            "WandB",
            "Google",
            "p50_ms_Alibaba",
            "p99_ms_Alibaba",
            "p50_ms_WandB",
            "p99_ms_WandB",
            "p50_ms_Google",
            "p99_ms_Google",
        ]
        existing_cols = [col for col in display_cols if col in alibaba_summary.columns]
        print(alibaba_summary[existing_cols].head(10).to_string(index=False))
    else:
        print("\nNo 15-minute windows found with Alibaba share above threshold.")


if __name__ == "__main__":
    main()
