"""Plots for the simulation scenario comparison.

Produces three views per scenario:
  slo_cost_pareto.png   SLO violation rate vs mean cost, one dot per policy
  provider_mix.png      Stacked bar of tier fractions per policy
  cost_over_time.png    Cumulative cost vs time for selected policies

A summary figure across all scenarios is emitted to `summary.png`.

All figures are saved as PNG (no PDF, per project convention).

Consumes:
  Run objects produced by experiments/simulation/* runners.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from plots.palettes import (
    ROUTER_STRATEGY_COLORS,
    ROUTER_STRATEGY_LABELS,
    TIER_COLORS,
)

if TYPE_CHECKING:
    from pathlib import Path

    from routewise.metrics import Run


def _mean_metric(runs: list[Run], fn) -> float:
    return float(np.mean([fn(r) for r in runs]))


def plot_slo_cost_pareto(
    scenario_name: str,
    primary_slo_ms: float,
    results: dict[str, list[Run]],
    output_path: Path,
) -> None:
    """Scatter of (mean cost, SLO violation rate) per policy."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    for policy_name, runs in results.items():
        x = _mean_metric(runs, lambda r: r.slo_violation_rate(primary_slo_ms))
        y = _mean_metric(runs, lambda r: r.mean_cost_usd())
        ax.scatter(
            x * 100.0,
            y,
            s=120,
            color=ROUTER_STRATEGY_COLORS.get(policy_name, "gray"),
            label=ROUTER_STRATEGY_LABELS.get(policy_name, policy_name),
            edgecolors="black",
            linewidths=0.8,
            alpha=0.85,
        )
        # Annotate each point.
        ax.annotate(
            ROUTER_STRATEGY_LABELS.get(policy_name, policy_name),
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
    results: dict[str, list[Run]],
    output_path: Path,
) -> None:
    """Stacked bar of tier fractions for each policy."""
    policies = list(results.keys())
    labels = [ROUTER_STRATEGY_LABELS.get(s, s) for s in policies]

    # Gather per-policy tier fractions (averaged across seeds).
    tiers = sorted({tier for runs in results.values() for r in runs for tier in r.tier_fractions()})

    tier_data: dict[str, list[float]] = {tier: [] for tier in tiers}
    for policy_name in policies:
        per_run = [r.tier_fractions() for r in results[policy_name]]
        for tier in tiers:
            tier_data[tier].append(float(np.mean([tf.get(tier, 0.0) for tf in per_run])))

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bottom = np.zeros(len(policies))
    for tier in tiers:
        vals = np.array(tier_data[tier])
        ax.bar(
            labels,
            vals,
            bottom=bottom,
            label=tier,
            color=TIER_COLORS.get(tier, "gray"),
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
    results: dict[str, list[Run]],
    output_path: Path,
    focus_strategies: list[str] | None = None,
) -> None:
    """Cumulative cost over simulated time, per policy."""
    if focus_strategies is None:
        focus_strategies = ["greedy_cost", "ablation_lp_only", "routewise"]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))

    for strat in focus_strategies:
        if strat not in results:
            continue
        r0 = results[strat][0]  # first seed is representative
        if not r0.records:
            continue
        elapsed = np.asarray([record.elapsed_sec for record in r0.records], dtype=float)
        ts = elapsed - elapsed[0]  # normalize to 0
        cum = np.cumsum([record.total_cost_usd for record in r0.records])
        ax.plot(
            ts / 60.0,
            cum,
            label=ROUTER_STRATEGY_LABELS.get(strat, strat),
            color=ROUTER_STRATEGY_COLORS.get(strat, "gray"),
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
    results: dict[str, list[Run]],
    output_dir: Path,
) -> None:
    """Emit all three plots for one scenario."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_slo_cost_pareto(
        scenario_name,
        primary_slo_ms,
        results,
        output_dir / "slo_cost_pareto.png",
    )
    plot_provider_mix(
        scenario_name,
        results,
        output_dir / "provider_mix.png",
    )
    plot_cost_over_time(
        scenario_name,
        results,
        output_dir / "cost_over_time.png",
    )


def plot_summary_across_scenarios(
    scenarios: dict[str, dict[str, list[Run]]],
    primary_slo_ms_map: dict[str, float],
    output_path: Path,
) -> None:
    """One grouped bar chart comparing all policies across all scenarios.

    X axis: scenario. Groups of bars per scenario show each policy's mean
    cost (left bar cluster) and SLO violation rate (right bar cluster), on
    twin y-axes.
    """
    scenario_ids = list(scenarios.keys())
    strategies = list(next(iter(scenarios.values())).keys())

    fig, (ax_cost, ax_slo) = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.0),
        sharex=True,
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
            viols.append(
                _mean_metric(runs, lambda r, s=sid: r.slo_violation_rate(primary_slo_ms_map[s]))
            )
        offset = (j - (n - 1) / 2.0) * bar_width
        ax_cost.bar(
            x + offset,
            costs,
            width=bar_width,
            color=ROUTER_STRATEGY_COLORS.get(strat, "gray"),
            label=ROUTER_STRATEGY_LABELS.get(strat, strat),
            edgecolor="white",
            linewidth=0.5,
        )
        ax_slo.bar(
            x + offset,
            np.array(viols) * 100.0,
            width=bar_width,
            color=ROUTER_STRATEGY_COLORS.get(strat, "gray"),
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

    fig.suptitle("Simulation scenarios: cost and SLO violation by policy", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
