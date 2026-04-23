"""Plotting functions for synthetic simulation results.

Generates four plot types per scenario:
  1. slo_violation.png  — grouped bar: SLO violation rate × strategy × SLO threshold
  2. cost_comparison.png — bar: mean cost per request by strategy
  3. provider_selection.png — time series: provider fraction over time (all strategies)
  4. latency_cdf.png — CDF of TTFT by strategy
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

from .runner import StrategyRun

# ---------------------------------------------------------------------------
# Colour palette (consistent across plots)
# ---------------------------------------------------------------------------

_STRATEGY_COLORS = {
    "cheapest_fixed":    "#4C72B0",
    "fastest_fixed":     "#55A868",
    "round_robin":       "#C44E52",
    "lp_mix":            "#8172B2",
    "lp_hedge":          "#B47CC7",  # lighter purple — LP family
    "v2_only":           "#937860",  # muted gold — V2 family, no hedge
    "v2_p50_hedge":      "#CCB974",  # bright gold — V2 family, with hedge
    "oracle_per_window": "#64B5CD",
}

_STRATEGY_LABELS = {
    "cheapest_fixed":    "Cheapest",
    "fastest_fixed":     "Fastest",
    "round_robin":       "Round-robin",
    "lp_mix":            "LP Mix",
    "lp_hedge":          "LP+Hedge",
    "v2_only":           "V2 only",
    "v2_p50_hedge":      "V2+Hedge",
    "oracle_per_window": "Oracle",
}


def _strategy_label(s: str) -> str:
    return _STRATEGY_LABELS.get(s, s)


def _strategy_color(s: str) -> str:
    return _STRATEGY_COLORS.get(s, "#888888")


def _avg_runs(runs: list[StrategyRun], fn) -> float:
    """Average a scalar metric across multiple seeds."""
    return float(np.mean([fn(r) for r in runs]))


# ---------------------------------------------------------------------------
# Plot 1: SLO violation bar chart
# ---------------------------------------------------------------------------


def plot_slo_violation(
    results: dict[str, list[StrategyRun]],
    slo_thresholds_ms: list[float],
    output_path: Path,
    title: str = "",
) -> None:
    """Grouped bar chart: SLO violation rate by strategy, grouped by SLO threshold."""
    if not _HAS_MPL:
        return

    strategies = list(results.keys())
    n_slo = len(slo_thresholds_ms)
    n_strat = len(strategies)

    fig, ax = plt.subplots(figsize=(max(8, n_strat * 1.5), 5))

    x = np.arange(n_slo)
    bar_width = 0.8 / n_strat
    offsets = np.linspace(-(n_strat - 1) / 2, (n_strat - 1) / 2, n_strat) * bar_width

    for i, strat in enumerate(strategies):
        runs = results[strat]
        rates = [
            _avg_runs(runs, lambda r, s=slo: r.slo_violation_rate(s))
            for slo in slo_thresholds_ms
        ]
        bars = ax.bar(
            x + offsets[i],
            rates,
            width=bar_width * 0.9,
            color=_strategy_color(strat),
            label=_strategy_label(strat),
        )
        # Annotate bars with value if > 1%.
        for bar, rate in zip(bars, rates):
            if rate >= 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{rate:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=45,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(s)} ms" for s in slo_thresholds_ms])
    ax.set_xlabel("SLO Threshold")
    ax.set_ylabel("Violation Rate")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title or "SLO Violation Rate by Strategy")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Cost bar chart
# ---------------------------------------------------------------------------


def plot_cost_comparison(
    results: dict[str, list[StrategyRun]],
    output_path: Path,
    title: str = "",
) -> None:
    """Bar chart: mean cost per request by strategy."""
    if not _HAS_MPL:
        return

    strategies = list(results.keys())
    means = [_avg_runs(results[s], lambda r: r.mean_cost_usd()) for s in strategies]

    fig, ax = plt.subplots(figsize=(max(6, len(strategies) * 1.2), 4))
    colors = [_strategy_color(s) for s in strategies]
    bars = ax.bar([_strategy_label(s) for s in strategies], means, color=colors)

    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{val:.2e}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylabel("Mean Cost per Request (USD)")
    ax.set_title(title or "Mean Cost per Request by Strategy")
    ax.set_ylim(bottom=0, top=max(means) * 1.15 if means else 1)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Provider selection time series
# ---------------------------------------------------------------------------


def plot_provider_selection(
    results: dict[str, list[StrategyRun]],
    output_path: Path,
    title: str = "",
    window_sec: float = 300.0,
) -> None:
    """Time series showing provider selection fractions over time per strategy."""
    if not _HAS_MPL:
        return

    strategies = list(results.keys())
    n = len(strategies)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    # Colour providers consistently across subplots.
    all_providers: list[str] = []
    for runs in results.values():
        for r in runs[:1]:
            for pname in sorted(set(r.provider)):
                if pname not in all_providers:
                    all_providers.append(pname)
    prov_colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(all_providers), 1)))
    pcolor = {p: prov_colors[i] for i, p in enumerate(all_providers)}

    for ax, strat in zip(axes, strategies):
        # Use the first seed run.
        run = results[strat][0]
        mids, fracs = run.provider_fractions_over_time(window_sec=window_sec)
        if len(mids) == 0:
            ax.set_title(f"{_strategy_label(strat)} — no data")
            continue

        # Convert seconds to minutes for readability.
        mids_min = mids / 60.0

        bottom = np.zeros(len(mids))
        for pname, frac in fracs.items():
            ax.fill_between(
                mids_min, bottom, bottom + frac,
                alpha=0.7,
                color=pcolor.get(pname, "grey"),
                label=pname,
            )
            bottom += frac

        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction")
        ax.set_title(f"{_strategy_label(strat)}", fontsize=9)
        ax.legend(loc="upper right", fontsize=7, ncol=len(all_providers))
        ax.grid(axis="y", alpha=0.2)

    axes[-1].set_xlabel("Time (minutes)")
    fig.suptitle(title or "Provider Selection Over Time", y=1.01, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: Latency CDF
# ---------------------------------------------------------------------------


def plot_latency_cdf(
    results: dict[str, list[StrategyRun]],
    output_path: Path,
    title: str = "",
    slo_ms: float | None = 2000.0,
    max_ms: float = 6000.0,
) -> None:
    """CDF of TTFT (ms) by strategy on a single axes."""
    if not _HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for strat, runs in results.items():
        # Pool all samples across seeds.
        all_ttft = np.concatenate([r.ttft_ms for r in runs])
        sorted_ttft = np.sort(all_ttft)
        cdf = np.arange(1, len(sorted_ttft) + 1) / len(sorted_ttft)
        # Clip display range.
        mask = sorted_ttft <= max_ms
        ax.plot(
            sorted_ttft[mask],
            cdf[mask],
            label=_strategy_label(strat),
            color=_strategy_color(strat),
            linewidth=1.8,
        )

    if slo_ms is not None:
        ax.axvline(slo_ms, color="black", linestyle="--", linewidth=1, label=f"SLO={int(slo_ms)} ms")

    ax.set_xlabel("TTFT (ms)")
    ax.set_ylabel("CDF")
    ax.set_xlim(left=0, right=max_ms)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=8)
    ax.set_title(title or "TTFT CDF by Strategy")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------


def make_plots(
    scenario_name: str,
    results: dict[str, list[StrategyRun]],
    output_dir: Path,
    slo_thresholds_ms: list[float] | None = None,
    primary_slo_ms: float = 2000.0,
) -> None:
    """Generate all four plots for a scenario and save to output_dir."""
    if not _HAS_MPL:
        print("  [plots] matplotlib not available — skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if slo_thresholds_ms is None:
        slo_thresholds_ms = [1000.0, 2000.0, 3000.0, 5000.0]

    plot_slo_violation(
        results,
        slo_thresholds_ms,
        output_dir / "slo_violation.png",
        title=f"[{scenario_name}] SLO Violation Rate",
    )
    plot_cost_comparison(
        results,
        output_dir / "cost_comparison.png",
        title=f"[{scenario_name}] Mean Cost per Request",
    )
    plot_provider_selection(
        results,
        output_dir / "provider_selection.png",
        title=f"[{scenario_name}] Provider Selection Over Time",
    )
    plot_latency_cdf(
        results,
        output_dir / "latency_cdf.png",
        title=f"[{scenario_name}] TTFT CDF",
        slo_ms=primary_slo_ms,
    )
