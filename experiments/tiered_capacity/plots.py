"""Plots for the tiered-scenario comparison.

Produces three views per scenario:
  slo_cost_pareto.png   SLO violation rate vs mean cost, one dot per strategy
  provider_mix.png      Stacked bar of tier fractions per strategy
  cost_over_time.png    Cumulative cost vs time for two_layer vs joint_nohedge

A summary figure across all scenarios is emitted to `summary.png`.

All figures are saved as PNG (no PDF, per project convention).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rwsim.world import StrategyRun


_STRATEGY_COLORS = {
    "two_layer": "#1f77b4",
    "joint_nohedge": "#2ca02c",
    "joint_hedge": "#17becf",
    "joint_p50band_nohedge": "#d62728",
    "joint_p50band_hedge": "#ff7f0e",
}

_STRATEGY_SHORT = {
    "two_layer": "two_layer",
    "joint_nohedge": "joint",
    "joint_hedge": "joint+hedge",
    "joint_p50band_nohedge": "joint(p50band)",
    "joint_p50band_hedge": "joint(p50band)+hedge",
}

_TIER_COLORS = {
    "quota": "#2ca02c",         # S_Q green
    "concurrency": "#9467bd",   # S_C purple
    "api": "#1f77b4",           # S_A blue
}


def _mean_metric(runs: list[StrategyRun], fn) -> float:
    return float(np.mean([fn(r) for r in runs]))


def plot_slo_cost_pareto(
    scenario_name: str,
    primary_slo_ms: float,
    results: dict[str, list[StrategyRun]],
    output_path: Path,
) -> None:
    """Scatter of (mean cost, SLO violation rate) per strategy."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    for strat, runs in results.items():
        x = _mean_metric(runs, lambda r: r.slo_violation_rate(primary_slo_ms))
        y = _mean_metric(runs, lambda r: r.mean_cost_usd())
        ax.scatter(
            x * 100.0, y,
            s=120,
            color=_STRATEGY_COLORS.get(strat, "gray"),
            label=_STRATEGY_SHORT.get(strat, strat),
            edgecolors="black",
            linewidths=0.8,
            alpha=0.85,
        )
        # Annotate each point.
        ax.annotate(
            _STRATEGY_SHORT.get(strat, strat),
            xy=(x * 100.0, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel(f"SLO violation rate (%) at SLO = {primary_slo_ms:.0f} ms")
    ax.set_ylabel("Mean cost per request (USD)")
    ax.set_title(f"{scenario_name}: cost vs SLO violation")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Place legend outside plot.
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_provider_mix(
    scenario_name: str,
    results: dict[str, list[StrategyRun]],
    output_path: Path,
) -> None:
    """Stacked bar of tier fractions for each strategy."""
    strategies = list(results.keys())
    labels = [_STRATEGY_SHORT.get(s, s) for s in strategies]

    # Gather per-strategy tier fractions (averaged across seeds).
    tiers = sorted({
        tier
        for runs in results.values()
        for r in runs
        for tier in r.tier_fractions().keys()
    })

    tier_data: dict[str, list[float]] = {tier: [] for tier in tiers}
    for strat in strategies:
        per_run = [r.tier_fractions() for r in results[strat]]
        for tier in tiers:
            tier_data[tier].append(
                float(np.mean([tf.get(tier, 0.0) for tf in per_run]))
            )

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bottom = np.zeros(len(strategies))
    for tier in tiers:
        vals = np.array(tier_data[tier])
        ax.bar(
            labels, vals,
            bottom=bottom,
            label=tier,
            color=_TIER_COLORS.get(tier, "gray"),
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += vals

    ax.set_ylabel("Fraction of requests")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{scenario_name}: provider tier mix")
    ax.legend(title="tier", loc="upper right", fontsize=8, frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_cost_over_time(
    scenario_name: str,
    results: dict[str, list[StrategyRun]],
    output_path: Path,
    focus_strategies: list[str] | None = None,
) -> None:
    """Cumulative cost over simulated time, per strategy."""
    if focus_strategies is None:
        focus_strategies = ["two_layer", "joint_nohedge", "joint_p50band_nohedge"]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))

    for strat in focus_strategies:
        if strat not in results:
            continue
        r0 = results[strat][0]  # first seed is representative
        if len(r0.timestamp) == 0:
            continue
        ts = r0.timestamp - r0.timestamp[0]  # normalize to 0
        cum = np.cumsum(r0.cost_usd)
        ax.plot(
            ts / 60.0, cum,
            label=_STRATEGY_SHORT.get(strat, strat),
            color=_STRATEGY_COLORS.get(strat, "gray"),
            linewidth=1.8,
        )

    ax.set_xlabel("Simulated time (minutes)")
    ax.set_ylabel("Cumulative cost (USD)")
    ax.set_title(f"{scenario_name}: cumulative cost over time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_scenario_plots(
    scenario_name: str,
    primary_slo_ms: float,
    results: dict[str, list[StrategyRun]],
    output_dir: Path,
) -> None:
    """Emit all three plots for one scenario."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_slo_cost_pareto(
        scenario_name, primary_slo_ms, results,
        output_dir / "slo_cost_pareto.png",
    )
    plot_provider_mix(
        scenario_name, results,
        output_dir / "provider_mix.png",
    )
    plot_cost_over_time(
        scenario_name, results,
        output_dir / "cost_over_time.png",
    )


def plot_summary_across_scenarios(
    scenarios: dict[str, dict[str, list[StrategyRun]]],
    primary_slo_ms_map: dict[str, float],
    output_path: Path,
) -> None:
    """One grouped bar chart comparing all strategies across all scenarios.

    X axis: scenario. Groups of bars per scenario show each strategy's mean
    cost (left bar cluster) and SLO violation rate (right bar cluster), on
    twin y-axes.
    """
    scenario_ids = list(scenarios.keys())
    strategies = list(next(iter(scenarios.values())).keys())

    fig, (ax_cost, ax_slo) = plt.subplots(
        1, 2, figsize=(11.0, 4.0), sharex=True,
    )

    n = len(strategies)
    bar_width = 0.8 / n
    x = np.arange(len(scenario_ids))

    for j, strat in enumerate(strategies):
        costs = []
        viols = []
        for sid in scenario_ids:
            runs = scenarios[sid][strat]
            costs.append(_mean_metric(runs, lambda r: r.mean_cost_usd()))
            viols.append(_mean_metric(
                runs, lambda r, s=sid: r.slo_violation_rate(primary_slo_ms_map[s])
            ))
        offset = (j - (n - 1) / 2.0) * bar_width
        ax_cost.bar(
            x + offset, costs,
            width=bar_width,
            color=_STRATEGY_COLORS.get(strat, "gray"),
            label=_STRATEGY_SHORT.get(strat, strat),
            edgecolor="white",
            linewidth=0.5,
        )
        ax_slo.bar(
            x + offset, np.array(viols) * 100.0,
            width=bar_width,
            color=_STRATEGY_COLORS.get(strat, "gray"),
            edgecolor="white",
            linewidth=0.5,
        )

    for ax, ylabel, title in [
        (ax_cost, "Mean cost per request (USD)", "Cost"),
        (ax_slo, "SLO violation rate (%)", "SLO violation"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_ids, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

    ax_cost.set_yscale("log")
    ax_cost.legend(
        loc="upper center",
        bbox_to_anchor=(1.1, -0.2),
        ncol=min(len(strategies), 5),
        fontsize=8,
        frameon=False,
    )

    fig.suptitle("Tiered scenarios: cost and SLO violation by strategy", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
