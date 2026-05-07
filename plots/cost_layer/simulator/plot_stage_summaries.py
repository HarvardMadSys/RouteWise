"""Summary figures for Cost Layer Stage 1.1 and Stage 1.2.

These plots are designed to sit next to the markdown summaries generated from
the simulator CSVs. They emphasize the sanity checks:

* Stage 1.1: p increases spend and provider diversity, but latency is flat
  because all on-demand providers share the same TTFT distribution.
* Stage 1.2: q produces the expected quota subscription U-shape, and effective
  cost uses quota more selectively than greedy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from plots.helpers import save_figure
from plots.style import apply_style

STAGE11_SCENARIOS = (
    "cost_layer_uniform",
    "cost_layer_normal",
    "cost_layer_heavy_tail",
    "cost_layer_real_world",
)

SCENARIO_LABELS = {
    "cost_layer_uniform": "Uniform",
    "cost_layer_normal": "Normal",
    "cost_layer_heavy_tail": "LogNormal",
    "cost_layer_real_world": "Real-world",
}

SCENARIO_COLORS = {
    "cost_layer_uniform": "#4C78A8",
    "cost_layer_normal": "#F58518",
    "cost_layer_heavy_tail": "#54A24B",
    "cost_layer_real_world": "#E45756",
}

P_POLICIES = (
    "ablation_lp_only_p0",
    "ablation_lp_only_p25",
    "ablation_lp_only_p50",
    "ablation_lp_only_p75",
    "ablation_lp_only_p100",
)

P_VALUES = {
    "ablation_lp_only_p0": 0.0,
    "ablation_lp_only_p25": 0.25,
    "ablation_lp_only_p50": 0.50,
    "ablation_lp_only_p75": 0.75,
    "ablation_lp_only_p100": 1.0,
}

P_LABELS = {
    "ablation_lp_only_p0": "p=0",
    "ablation_lp_only_p25": "p=.25",
    "ablation_lp_only_p50": "p=.50",
    "ablation_lp_only_p75": "p=.75",
    "ablation_lp_only_p100": "p=1",
}

PROVIDER_ORDER_STAGE11 = ("api_cheap", "api_mid", "api_expensive")
PROVIDER_ORDER_STAGE12 = ("chutes_quota", "api_cheap", "api_mid", "api_expensive")

PROVIDER_LABELS = {
    "chutes_quota": "quota",
    "api_cheap": "cheap",
    "api_mid": "mid",
    "api_expensive": "expensive",
}

PROVIDER_COLORS = {
    "chutes_quota": "#59A14F",
    "api_cheap": "#4C78A8",
    "api_mid": "#F58518",
    "api_expensive": "#E45756",
}

POLICY_ORDER_STAGE12 = (
    "offline",
    "ablation_lp_only_p0",
    "greedy_cost",
    "ablation_lp_only_p25",
    "ablation_lp_only_p50",
    "ablation_lp_only_p75",
    "ablation_lp_only_p100",
    "random",
)

MAIN_QUOTA_POLICIES = (
    "offline",
    "ablation_lp_only_p0",
    "greedy_cost",
    "random",
)

POLICY_LABELS = {
    "offline": "Offline",
    "greedy_cost": "Greedy",
    "random": "Random",
    "ablation_lp_only_p0": "RW p=0",
    "ablation_lp_only_p25": "RW p=.25",
    "ablation_lp_only_p50": "RW p=.50",
    "ablation_lp_only_p75": "RW p=.75",
    "ablation_lp_only_p100": "RW p=1",
}

POLICY_COLORS = {
    "offline": "#555555",
    "greedy_cost": "#1f77b4",
    "random": "#7f8c8d",
    "ablation_lp_only_p0": "#9467bd",
    "ablation_lp_only_p25": "#8c6bb1",
    "ablation_lp_only_p50": "#6a51a3",
    "ablation_lp_only_p75": "#54278f",
    "ablation_lp_only_p100": "#3f007d",
}

POLICY_LINESTYLES = {
    "offline": "-",
    "greedy_cost": "-",
    "random": "--",
    "ablation_lp_only_p0": "-.",
}


def apply_summary_style() -> None:
    """Apply a compact style for summary figures."""
    apply_style("paper")
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "lines.linewidth": 1.65,
            "lines.markersize": 4.2,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        }
    )


def load_summary(path: Path) -> list[dict[str, str]]:
    """Load a simulator summary CSV."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def _json_map(row: dict[str, str], key: str) -> dict[str, float]:
    payload = json.loads(row[key])
    return {str(name): float(value) for name, value in payload.items()}


def _stage11_row(
    rows: list[dict[str, str]],
    *,
    scenario: str,
    policy: str,
) -> dict[str, str]:
    for row in rows:
        if row["scenario"] == scenario and row["policy"] == policy:
            return row
    raise KeyError(f"missing row for scenario={scenario!r}, policy={policy!r}")


def _quota_rows(rows: list[dict[str, str]], policy: str) -> list[dict[str, str]]:
    selected = [row for row in rows if row["policy"] == policy and row["public_scenario"] == "quota"]
    return sorted(selected, key=lambda row: _int(row, "subscription_count"))


def _best_quota_row(rows: list[dict[str, str]], policy: str) -> dict[str, str]:
    return min(_quota_rows(rows, policy), key=lambda row: _float(row, "total_cost_usd"))


def plot_stage11_p_sweep(rows: list[dict[str, str]], output_dir: Path) -> None:
    """Plot Stage 1.1 cost increase and flat latency across p."""
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.45), sharex=True)
    x = [P_VALUES[policy] for policy in P_POLICIES]

    for scenario in STAGE11_SCENARIOS:
        cost_values = [
            _float(_stage11_row(rows, scenario=scenario, policy=policy), "total_cost_usd")
            for policy in P_POLICIES
        ]
        p99_values = [
            _float(_stage11_row(rows, scenario=scenario, policy=policy), "p99_ms")
            for policy in P_POLICIES
        ]
        axes[0].plot(
            x,
            cost_values,
            marker="o",
            color=SCENARIO_COLORS[scenario],
            label=SCENARIO_LABELS[scenario],
        )
        axes[1].plot(
            x,
            p99_values,
            marker="o",
            color=SCENARIO_COLORS[scenario],
            label=SCENARIO_LABELS[scenario],
        )

    axes[0].set_ylabel("Total cost ($)")
    axes[0].set_title("Cost rises with p")
    axes[0].grid(True, alpha=0.22)
    axes[1].set_ylabel("P99 TTFT (ms)")
    axes[1].set_yscale("log")
    axes[1].set_title("Latency stays flat")
    axes[1].grid(True, which="both", alpha=0.18)
    for ax in axes:
        ax.set_xlabel("p")
        ax.set_xticks(x)
        ax.set_xticklabels(["0", ".25", ".50", ".75", "1"])
    axes[0].legend(frameon=False, ncols=2, loc="upper left")
    fig.suptitle("Stage 1.1: p changes spend, not latency", y=1.04, fontsize=10)
    save_figure(fig, output_dir, "stage1_1_p_sweep_cost_vs_latency", formats=["png", "pdf"])
    plt.close(fig)


def plot_stage11_provider_mix(rows: list[dict[str, str]], output_dir: Path) -> None:
    """Plot how p shifts provider mix for every latency family."""
    fig, axes = plt.subplots(2, 2, figsize=(6.7, 4.1), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    x = np.arange(len(P_POLICIES))

    for ax, scenario in zip(axes_flat, STAGE11_SCENARIOS, strict=True):
        bottom = np.zeros(len(P_POLICIES))
        for provider in PROVIDER_ORDER_STAGE11:
            values = []
            for policy in P_POLICIES:
                row = _stage11_row(rows, scenario=scenario, policy=policy)
                values.append(_json_map(row, "provider_mix").get(provider, 0.0))
            ax.bar(
                x,
                values,
                bottom=bottom,
                color=PROVIDER_COLORS[provider],
                label=PROVIDER_LABELS[provider],
                width=0.68,
            )
            bottom += np.asarray(values)
        ax.set_title(SCENARIO_LABELS[scenario])
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.18)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xticks(x)
        ax.set_xticklabels(["0", ".25", ".50", ".75", "1"])
    axes[1, 0].set_xlabel("p")
    axes[1, 1].set_xlabel("p")
    axes[0, 0].set_ylabel("Request fraction")
    axes[1, 0].set_ylabel("Request fraction")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Stage 1.1: p moves traffic to pricier providers", y=1.10, fontsize=10)
    save_figure(fig, output_dir, "stage1_1_provider_mix_by_p", formats=["png", "pdf"])
    plt.close(fig)


def plot_stage12_total_cost(rows: list[dict[str, str]], output_dir: Path) -> None:
    """Plot the Stage 1.2 q-sweep total-cost U-shape."""
    fig, ax = plt.subplots(figsize=(6.1, 3.0))
    for policy in POLICY_ORDER_STAGE12:
        policy_rows = _quota_rows(rows, policy)
        counts = [_int(row, "subscription_count") for row in policy_rows]
        totals = [_float(row, "total_cost_usd") for row in policy_rows]
        ax.plot(
            counts,
            totals,
            marker="o",
            color=POLICY_COLORS[policy],
            linestyle=POLICY_LINESTYLES.get(policy, "-"),
            label=POLICY_LABELS[policy],
            alpha=0.95,
        )
        best = _best_quota_row(rows, policy)
        ax.scatter(
            [_int(best, "subscription_count")],
            [_float(best, "total_cost_usd")],
            s=42,
            facecolor="white",
            edgecolor=POLICY_COLORS[policy],
            linewidth=1.2,
            zorder=5,
        )
    ax.set_xlabel("Subscriptions (q)")
    ax.set_ylabel("Total cost ($)")
    ax.set_title("Stage 1.2: total cost has an interior q optimum")
    ax.grid(True, alpha=0.22)
    _set_q_ticks(ax, rows)
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    save_figure(fig, output_dir, "stage1_2_q_sweep_total_cost_all_policies", formats=["png", "pdf"])
    plt.close(fig)


def plot_stage12_quota_fraction(rows: list[dict[str, str]], output_dir: Path) -> None:
    """Plot quota request fraction for the main sanity-check policies."""
    fig, ax = plt.subplots(figsize=(4.4, 2.65))
    for policy in MAIN_QUOTA_POLICIES:
        policy_rows = _quota_rows(rows, policy)
        counts = [_int(row, "subscription_count") for row in policy_rows]
        quota_fractions = [_json_map(row, "tier_mix").get("quota", 0.0) for row in policy_rows]
        ax.plot(
            counts,
            quota_fractions,
            marker="o",
            color=POLICY_COLORS[policy],
            linestyle=POLICY_LINESTYLES.get(policy, "-"),
            label=POLICY_LABELS[policy],
        )
    ax.set_xlabel("Subscriptions (q)")
    ax.set_ylabel("Quota request fraction")
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Effective cost uses quota more selectively")
    ax.grid(True, alpha=0.22)
    _set_q_ticks(ax, rows)
    ax.legend(frameon=False, ncols=2, loc="upper left")
    save_figure(fig, output_dir, "stage1_2_quota_fraction_main_policies", formats=["png", "pdf"])
    plt.close(fig)


def plot_stage12_p0_cost_decomposition(rows: list[dict[str, str]], output_dir: Path) -> None:
    """Plot p=0 API spillover cost versus fixed subscription cost."""
    policy_rows = _quota_rows(rows, "ablation_lp_only_p0")
    counts = np.asarray([_int(row, "subscription_count") for row in policy_rows])
    api_cost = np.asarray([_float(row, "api_cost_usd") for row in policy_rows])
    fixed_cost = np.asarray([_float(row, "subscription_fixed_cost_usd") for row in policy_rows])
    total_cost = api_cost + fixed_cost
    best = _best_quota_row(rows, "ablation_lp_only_p0")

    fig, ax = plt.subplots(figsize=(4.5, 2.75))
    ax.stackplot(
        counts,
        api_cost,
        fixed_cost,
        labels=("API spillover", "Fixed subscription"),
        colors=("#9ecae1", "#fdae6b"),
        alpha=0.9,
    )
    ax.plot(counts, total_cost, color="#54278f", marker="o", label="Total")
    ax.scatter(
        [_int(best, "subscription_count")],
        [_float(best, "total_cost_usd")],
        s=50,
        facecolor="white",
        edgecolor="#54278f",
        linewidth=1.3,
        zorder=5,
    )
    ax.annotate(
        f"q*={_int(best, 'subscription_count')}",
        xy=(_int(best, "subscription_count"), _float(best, "total_cost_usd")),
        xytext=(3.0, 28.0),
        textcoords="offset points",
        fontsize=7,
    )
    ax.set_xlabel("Subscriptions (q)")
    ax.set_ylabel("Cost ($)")
    ax.set_title("RW p=0 balances API spillover and fixed fee")
    ax.grid(True, alpha=0.22)
    _set_q_ticks(ax, rows)
    ax.legend(frameon=False, loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.20))
    save_figure(fig, output_dir, "stage1_2_p0_cost_decomposition", formats=["png", "pdf"])
    plt.close(fig)


def _set_q_ticks(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    counts = sorted({_int(row, "subscription_count") for row in rows if row["public_scenario"] == "quota"})
    preferred = [1, 4, 8, 12, 16, 20, 24, 32, 40]
    ticks = [count for count in preferred if count in counts]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(tick) for tick in ticks])


def make_summary_plots(
    *,
    stage11_dir: Path,
    stage12_dir: Path,
    output_dir: Path,
) -> None:
    """Generate summary figures for Stage 1.1 and Stage 1.2."""
    apply_summary_style()
    stage11_rows = load_summary(stage11_dir / "summary.csv")
    stage12_rows = load_summary(stage12_dir / "summary.csv")
    plot_stage11_p_sweep(stage11_rows, output_dir)
    plot_stage11_provider_mix(stage11_rows, output_dir)
    plot_stage12_total_cost(stage12_rows, output_dir)
    plot_stage12_quota_fraction(stage12_rows, output_dir)
    plot_stage12_p0_cost_decomposition(stage12_rows, output_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage11-dir",
        type=Path,
        default=Path("outputs/stage0_on_demand_provider_rng"),
        help="Stage 1.1 output directory containing summary.csv.",
    )
    parser.add_argument(
        "--stage12-dir",
        type=Path,
        default=Path("outputs/stage1_quota_chutes_sweep"),
        help="Stage 1.2 output directory containing summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cost_layer_summary_figures"),
        help="Directory to write summary figures.",
    )
    args = parser.parse_args(argv)
    make_summary_plots(
        stage11_dir=args.stage11_dir,
        stage12_dir=args.stage12_dir,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
