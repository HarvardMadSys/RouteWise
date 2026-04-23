"""Plot Pareto frontiers for Joint vs Two-layer.

Generates:
  1. Per-scenario scatter on (cost, SLO-violation) axes.
     - Two-layer variants: circles
     - Joint variants: stars
     - Connects variants of the same mechanism with a line.
  2. Combined grid figure with all scenarios.
  3. Summary table plot: win counts per mechanism.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MECHANISMS = ["cheapest", "fastest", "v2", "lp", "hedge"]

MECH_COLORS = {
    "cheapest": "#1f77b4",
    "fastest": "#ff7f0e",
    "v2": "#2ca02c",
    "lp": "#d62728",
    "hedge": "#9467bd",
}


def _mech_of(strategy_name: str) -> str:
    for m in MECHANISMS:
        if strategy_name.endswith("_" + m):
            return m
    return "unknown"


def _arch_of(strategy_name: str) -> str:
    if strategy_name.startswith("joint_"):
        return "joint"
    if strategy_name.startswith("two_layer_"):
        return "two_layer"
    return "unknown"


def plot_pareto(
    scenario_summary: dict,
    output_path: Path,
    use_p99: bool = False,
) -> None:
    """Plot one scenario's 5x2 Pareto scatter.

    X-axis: total cost USD.
    Y-axis: SLO violation rate (%) or P99 TTFT (ms) if use_p99.
    """
    results = scenario_summary["results"]
    fig, ax = plt.subplots(figsize=(8, 6))

    # Group by mechanism.
    by_mech: dict[str, dict[str, dict]] = {}
    for r in results:
        m = _mech_of(r["strategy"])
        a = _arch_of(r["strategy"])
        if m not in by_mech:
            by_mech[m] = {}
        by_mech[m][a] = r

    for mech in MECHANISMS:
        if mech not in by_mech:
            continue
        pair = by_mech[mech]
        color = MECH_COLORS[mech]
        for arch, marker, size in [
            ("two_layer", "o", 120),
            ("joint", "*", 280),
        ]:
            if arch not in pair:
                continue
            r = pair[arch]
            cost = r["cost"]["mean"]
            y = (
                r["p99_ttft_ms"]["mean"]
                if use_p99
                else r["slo_violation_rate"]["mean"] * 100.0
            )
            label = f"{arch} {mech}" if arch == "joint" else None
            ax.scatter(
                cost,
                y,
                s=size,
                c=color,
                marker=marker,
                edgecolor="black",
                linewidth=0.8,
                alpha=0.85,
                label=label,
            )
            # Annotate with mechanism name near joint variant only
            if arch == "joint":
                ax.annotate(
                    mech,
                    (cost, y),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=9,
                    color=color,
                )

        # Connect two_layer and joint for same mechanism with a dashed line.
        if "two_layer" in pair and "joint" in pair:
            x_tl = pair["two_layer"]["cost"]["mean"]
            x_j = pair["joint"]["cost"]["mean"]
            y_tl = (
                pair["two_layer"]["p99_ttft_ms"]["mean"]
                if use_p99
                else pair["two_layer"]["slo_violation_rate"]["mean"] * 100.0
            )
            y_j = (
                pair["joint"]["p99_ttft_ms"]["mean"]
                if use_p99
                else pair["joint"]["slo_violation_rate"]["mean"] * 100.0
            )
            ax.plot(
                [x_tl, x_j], [y_tl, y_j],
                linestyle=":",
                color=color,
                alpha=0.4,
                linewidth=1.2,
            )

    ax.set_xlabel("Total cost (USD)", fontsize=12)
    ax.set_ylabel(
        "P99 TTFT (ms)" if use_p99 else "SLO violation rate (%)",
        fontsize=12,
    )
    ax.set_title(
        f"{scenario_summary['scenario']}  |  SLO = {scenario_summary['slo_ms']:.0f} ms\n"
        f"{scenario_summary['n_requests']} req, {scenario_summary['n_seeds']} seeds",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3, linestyle="--")

    # Legend: circles + stars at top.
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="two-layer",
               markerfacecolor="gray", markersize=10, markeredgecolor="black"),
        Line2D([0], [0], marker="*", color="w", label="joint",
               markerfacecolor="gray", markersize=15, markeredgecolor="black"),
    ]
    for mech in MECHANISMS:
        if mech in by_mech:
            legend_elements.append(
                Line2D([0], [0], marker="s", color="w", label=mech,
                       markerfacecolor=MECH_COLORS[mech], markersize=10)
            )
    ax.legend(handles=legend_elements, loc="best", fontsize=9, ncol=2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_combined_grid(
    summaries: list[dict],
    output_path: Path,
) -> None:
    """All scenarios in a single grid figure."""
    n = len(summaries)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 5 * rows), squeeze=False)

    for idx, summary in enumerate(summaries):
        ax = axes[idx // cols][idx % cols]

        by_mech: dict[str, dict[str, dict]] = {}
        for r in summary["results"]:
            m = _mech_of(r["strategy"])
            a = _arch_of(r["strategy"])
            if m not in by_mech:
                by_mech[m] = {}
            by_mech[m][a] = r

        for mech in MECHANISMS:
            if mech not in by_mech:
                continue
            pair = by_mech[mech]
            color = MECH_COLORS[mech]
            for arch, marker, size in [("two_layer", "o", 80), ("joint", "*", 200)]:
                if arch not in pair:
                    continue
                r = pair[arch]
                cost = r["cost"]["mean"]
                y = r["slo_violation_rate"]["mean"] * 100.0
                ax.scatter(
                    cost, y, s=size, c=color, marker=marker,
                    edgecolor="black", linewidth=0.6, alpha=0.85,
                )
            if "two_layer" in pair and "joint" in pair:
                x1 = pair["two_layer"]["cost"]["mean"]
                x2 = pair["joint"]["cost"]["mean"]
                y1 = pair["two_layer"]["slo_violation_rate"]["mean"] * 100.0
                y2 = pair["joint"]["slo_violation_rate"]["mean"] * 100.0
                ax.plot([x1, x2], [y1, y2], ":", color=color, alpha=0.4, linewidth=1.0)

        ax.set_xlabel("Cost (USD)", fontsize=10)
        ax.set_ylabel("SLO violation (%)", fontsize=10)
        ax.set_title(
            f"{summary['scenario']} (SLO={summary['slo_ms']:.0f}ms, "
            f"n={summary['n_requests']})",
            fontsize=10,
        )
        ax.grid(True, alpha=0.3, linestyle="--")

    # Hide empty panes
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    # Shared legend.
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="two-layer",
               markerfacecolor="gray", markersize=10, markeredgecolor="black"),
        Line2D([0], [0], marker="*", color="w", label="joint",
               markerfacecolor="gray", markersize=15, markeredgecolor="black"),
    ]
    for mech in MECHANISMS:
        legend_elements.append(
            Line2D([0], [0], marker="s", color="w", label=mech,
                   markerfacecolor=MECH_COLORS[mech], markersize=10)
        )
    fig.legend(handles=legend_elements, loc="upper center",
               ncol=len(legend_elements), fontsize=10,
               bbox_to_anchor=(0.5, 1.02), frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dominance_matrix(
    summaries: list[dict],
    output_path: Path,
) -> None:
    """Heatmap: for each (scenario, mechanism) pair, is joint Pareto-dominant?

    +1 = joint strictly dominates (both axes better or equal, at least one better)
     0 = tie
    -1 = joint strictly worse
    """
    scenario_names = [s["scenario"] for s in summaries]
    mat = np.zeros((len(MECHANISMS), len(scenario_names)))

    for j, summary in enumerate(summaries):
        by = {}
        for r in summary["results"]:
            m = _mech_of(r["strategy"])
            a = _arch_of(r["strategy"])
            by.setdefault(m, {})[a] = r
        for i, mech in enumerate(MECHANISMS):
            if mech not in by or "joint" not in by[mech] or "two_layer" not in by[mech]:
                continue
            j_cost = by[mech]["joint"]["cost"]["mean"]
            t_cost = by[mech]["two_layer"]["cost"]["mean"]
            j_viol = by[mech]["joint"]["slo_violation_rate"]["mean"]
            t_viol = by[mech]["two_layer"]["slo_violation_rate"]["mean"]

            # Tolerance for "equal".
            cost_tol = max(0.001, 0.02 * max(j_cost, t_cost))
            viol_tol = 0.002  # 0.2%

            cost_better = j_cost < t_cost - cost_tol
            cost_equal = abs(j_cost - t_cost) <= cost_tol
            cost_worse = j_cost > t_cost + cost_tol
            viol_better = j_viol < t_viol - viol_tol
            viol_equal = abs(j_viol - t_viol) <= viol_tol
            viol_worse = j_viol > t_viol + viol_tol

            if (cost_better and not viol_worse) or (viol_better and not cost_worse):
                mat[i, j] = 1  # joint dominates
            elif cost_equal and viol_equal:
                mat[i, j] = 0
            elif (cost_worse and not viol_better) or (viol_worse and not cost_better):
                mat[i, j] = -1  # joint worse
            else:
                mat[i, j] = 0.5  # mixed / Pareto non-comparable (both differ)

    fig, ax = plt.subplots(figsize=(1.5 + 1.2 * len(scenario_names), 1.0 + 0.6 * len(MECHANISMS)))
    # Color scheme: green = joint wins, red = joint loses, gray = tie.
    cmap = plt.cm.RdYlGn
    im = ax.imshow(mat, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    # Annotations.
    labels = {1: "J>T", 0: "tie", 0.5: "mix", -1: "J<T"}
    for i in range(len(MECHANISMS)):
        for j in range(len(scenario_names)):
            v = mat[i, j]
            text = labels.get(v, "?")
            # pick contrasting text color
            color = "black" if -0.5 < v < 0.8 else "white"
            ax.text(j, i, text, ha="center", va="center", fontsize=10, color=color)

    ax.set_xticks(range(len(scenario_names)))
    ax.set_xticklabels(scenario_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(MECHANISMS)))
    ax.set_yticklabels(MECHANISMS, fontsize=10)
    ax.set_title("Joint vs Two-layer: Pareto dominance per (mechanism, scenario)", fontsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("experiment/results/synthetic"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find scenarios.
    summaries: list[dict] = []
    for p in sorted(args.input_dir.iterdir()):
        if not p.is_dir():
            continue
        summary_file = p / "summary.json"
        if not summary_file.exists():
            continue
        with open(summary_file) as f:
            summaries.append(json.load(f))

    if not summaries:
        raise SystemExit(f"No summaries found in {args.input_dir}")

    # Per-scenario plots.
    for s in summaries:
        plot_pareto(s, output_dir / f"pareto_{s['scenario']}_slo.png", use_p99=False)
        plot_pareto(s, output_dir / f"pareto_{s['scenario']}_p99.png", use_p99=True)

    # Combined grid.
    plot_combined_grid(summaries, output_dir / "all_scenarios_slo.png")
    # Dominance matrix.
    plot_dominance_matrix(summaries, output_dir / "dominance_matrix.png")

    print(f"Plots in {output_dir}")


if __name__ == "__main__":
    main()
