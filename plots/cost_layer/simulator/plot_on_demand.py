"""On-demand simulator figures for the cost-layer section.

Consumes a ``routewise simulator cost-layer`` output directory containing
``summary.csv`` and ``ttft_histograms.json``. The scenario is the cost-layer
sanity check: same latency distribution, different on-demand prices.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

from plots.helpers import save_figure
from plots.style import apply_style
from rwsim.metrics.histogram import TtftHistogram


SCENARIOS = (
    "cost_layer_uniform",
    "cost_layer_normal",
    "cost_layer_heavy_tail",
    "cost_layer_real_world",
)

SCENARIO_LABELS = {
    "cost_layer_uniform": "Uniform",
    "cost_layer_normal": "Normal",
    "cost_layer_heavy_tail": "Heavy-tail",
    "cost_layer_real_world": "Real-world",
}

SCENARIO_SLUGS = {
    "cost_layer_uniform": "uniform",
    "cost_layer_normal": "normal",
    "cost_layer_heavy_tail": "heavy_tail",
    "cost_layer_real_world": "real_world",
}

POLICIES = (
    "greedy_cost",
    "random",
    "offline",
    "ablation_lp_only_p0",
)

CDF_POLICIES = (
    "greedy_cost",
    "random",
    "ablation_lp_only_p0",
)

POLICY_LABELS = {
    "greedy_cost": "Greedy",
    "random": "Random",
    "offline": "Offline",
    "ablation_lp_only_p0": "RW p=0",
}

POLICY_COLORS = {
    "greedy_cost": "#1f77b4",
    "random": "#7f8c8d",
    "offline": "#555555",
    "ablation_lp_only_p0": "#9467bd",
}

POLICY_LINESTYLES = {
    "greedy_cost": "-",
    "random": "--",
    "ablation_lp_only_p0": ":",
}

PROVIDER_ORDER = (
    "api_cheap",
    "api_mid",
    "api_expensive",
)

PROVIDER_LABELS = {
    "api_cheap": "cheap",
    "api_mid": "mid",
    "api_expensive": "expensive",
}

PROVIDER_COLORS = {
    "api_cheap": "#4C78A8",
    "api_mid": "#F58518",
    "api_expensive": "#E45756",
}


def apply_on_demand_style() -> None:
    """Use compact paper-panel typography for standalone figure snippets."""
    apply_style("paper")
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "lines.linewidth": 1.6,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "savefig.pad_inches": 0.02,
            # Embed TrueType fonts so reviewers can ctrl-F text inside the PDF
            # and publishers do not reject path-encoded glyphs.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica",
                "Arial",
                "DejaVu Sans",
            ],
        }
    )


def _log_axis_human_ticks(
    ax: plt.Axes,
    *,
    axis: str = "x",
    ticks: tuple[float, ...] = (100.0, 200.0, 300.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0),
) -> None:
    """Replace matplotlib's ``2 × 10²`` style log ticks with plain numbers."""
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(FixedLocator(list(ticks)))
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    target.set_major_formatter(formatter)
    target.set_minor_formatter(NullFormatter())


def load_summary(input_dir: Path) -> list[dict[str, str]]:
    """Load section summary rows."""
    path = input_dir / "summary.csv"
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_histograms(input_dir: Path) -> list[dict[str, Any]]:
    """Load merged TTFT histograms."""
    with (input_dir / "ttft_histograms.json").open() as handle:
        return json.load(handle)


def summary_row(
    rows: list[dict[str, str]],
    *,
    scenario: str,
    policy: str,
) -> dict[str, str]:
    """Return one summary row."""
    for row in rows:
        if row["scenario"] == scenario and row["policy"] == policy:
            return row
    raise KeyError(f"missing summary row for scenario={scenario!r}, policy={policy!r}")


def histogram_for(
    rows: list[dict[str, Any]],
    *,
    scenario: str,
    policy: str,
) -> TtftHistogram:
    """Return one histogram as a ``TtftHistogram`` instance."""
    for row in rows:
        if row["scenario"] == scenario and row["policy"] == policy:
            payload = row["histogram"]
            return TtftHistogram(
                bin_edges_ms=np.asarray(payload["bin_edges_ms"], dtype=float),
                counts=np.asarray(payload["counts"], dtype=np.int64),
                sum_value=float(payload["mean_ms"]) * int(payload["n"]),
                # The CDF plot does not need exact std, but preserve the field
                # enough for TtftHistogram invariants and future reuse.
                sum_sq=(
                    float(payload["std_ms"]) ** 2 + float(payload["mean_ms"]) ** 2
                )
                * int(payload["n"]),
                n=int(payload["n"]),
                min_value=float(payload["min_ms"]),
                max_value=float(payload["max_ms"]),
            )
    raise KeyError(f"missing histogram for scenario={scenario!r}, policy={policy!r}")


def plot_total_cost(
    rows: list[dict[str, str]],
    output_dir: Path,
    *,
    scenario: str,
) -> None:
    """Plot total cost over the full workload for one scenario."""
    fig, ax = plt.subplots(figsize=(3.0, 1.85))
    y_pos = np.arange(len(POLICIES))
    costs = [
        float(summary_row(rows, scenario=scenario, policy=policy)["total_cost_usd"])
        for policy in POLICIES
    ]
    ax.barh(
        y_pos,
        costs,
        color=[POLICY_COLORS[policy] for policy in POLICIES],
        height=0.62,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([POLICY_LABELS[policy] for policy in POLICIES])
    ax.invert_yaxis()
    ax.set_xlabel("Total cost ($)")
    ax.grid(axis="x", alpha=0.22)
    xmax = max(costs) * 1.16
    ax.set_xlim(0.0, xmax)
    for y, cost in zip(y_pos, costs, strict=True):
        label = f"${cost / 1000:.1f}k" if cost >= 1000 else f"${cost:.0f}"
        ax.text(cost + xmax * 0.015, y, label, va="center", fontsize=7)
    save_figure(
        fig,
        output_dir,
        f"cost_layer_on_demand_{SCENARIO_SLUGS[scenario]}_total_cost",
        formats=["pdf"],
    )
    plt.close(fig)


def plot_provider_mix(
    rows: list[dict[str, str]],
    output_dir: Path,
    *,
    scenario: str,
) -> None:
    """Plot provider routing fractions for one scenario."""
    fig, ax = plt.subplots(figsize=(3.1, 2.05))
    y_pos = np.arange(len(POLICIES))
    bottom = np.zeros(len(POLICIES))
    for provider in PROVIDER_ORDER:
        values = []
        for policy in POLICIES:
                row = summary_row(rows, scenario=scenario, policy=policy)
                mix = json.loads(row["provider_mix"])
                values.append(float(mix.get(provider, 0.0)))
        ax.barh(
            y_pos,
            values,
            left=bottom,
            color=PROVIDER_COLORS[provider],
            label=PROVIDER_LABELS[provider],
            height=0.62,
        )
        bottom += np.asarray(values)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([POLICY_LABELS[policy] for policy in POLICIES])
    ax.invert_yaxis()
    ax.set_xlabel("Request fraction")
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.grid(axis="x", alpha=0.22)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        ncols=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        columnspacing=0.8,
        handlelength=1.3,
    )
    save_figure(
        fig,
        output_dir,
        f"cost_layer_on_demand_{SCENARIO_SLUGS[scenario]}_provider_mix",
        formats=["pdf"],
    )
    plt.close(fig)


def plot_ttft_cdf(
    summary_rows: list[dict[str, str]],
    histogram_rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    scenario: str,
) -> None:
    """Plot TTFT CDF sanity check for one scenario."""
    fig, ax = plt.subplots(figsize=(3.1, 2.15))

    for policy in CDF_POLICIES:
        histogram = histogram_for(histogram_rows, scenario=scenario, policy=policy)
        summary = summary_row(summary_rows, scenario=scenario, policy="greedy_cost")
        x_min = max(float(summary["p10_ms"]) * 0.6, 1.0)
        x_max = max(float(summary["p99_ms"]) * 1.6, x_min * 1.1)
        x_values = np.geomspace(
            x_min,
            x_max,
            512,
        )
        y_values = np.asarray([histogram.cdf(value) for value in x_values])
        ax.plot(
            x_values,
            y_values,
            label=POLICY_LABELS[policy],
            color=POLICY_COLORS[policy],
            linestyle=POLICY_LINESTYLES[policy],
            linewidth=1.8,
            alpha=0.95,
        )

    row = summary_row(summary_rows, scenario=scenario, policy="greedy_cost")
    p50_ms = float(row["p50_ms"])
    p99_ms = float(row["p99_ms"])
    ax.axvline(p50_ms, color="black", linestyle=":", linewidth=1.0, alpha=0.45)
    ax.axvline(p99_ms, color="black", linestyle="--", linewidth=1.0, alpha=0.45)
    ax.text(
        0.03,
        0.08,
        f"P50={p50_ms:.0f}ms\nP99={p99_ms:.0f}ms",
        transform=ax.transAxes,
        fontsize=7,
    )
    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.0, 1.01)
    _log_axis_human_ticks(ax, axis="x")
    ax.grid(True, which="both", alpha=0.18)
    ax.set_xlabel("TTFT (ms)")
    ax.set_ylabel("CDF")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        ncols=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        columnspacing=0.8,
        handlelength=1.5,
    )
    save_figure(
        fig,
        output_dir,
        f"cost_layer_on_demand_{SCENARIO_SLUGS[scenario]}_ttft_cdf",
        formats=["pdf"],
    )
    plt.close(fig)


def write_cost_table(
    rows: list[dict[str, str]],
    output_dir: Path,
    *,
    scenario: str,
) -> None:
    """Emit per-scenario cost table (CSV) accompanying the cost bar chart.

    Juncheng's metric list calls for "cost (bar AND table)". The bar chart
    shows the magnitude visually; this table is for paper text/appendix
    citation and lets the cost claim survive without re-reading the figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"cost_layer_on_demand_{SCENARIO_SLUGS[scenario]}_cost_table.csv"
    offline_cost = float(summary_row(rows, scenario=scenario, policy="offline")["total_cost_usd"])
    fieldnames = (
        "policy",
        "n_requests",
        "total_cost_usd",
        "mean_cost_usd",
        "vs_offline",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for policy in POLICIES:
            row = summary_row(rows, scenario=scenario, policy=policy)
            total = float(row["total_cost_usd"])
            writer.writerow(
                {
                    "policy": POLICY_LABELS[policy],
                    "n_requests": row["n_requests"],
                    "total_cost_usd": f"{total:.6f}",
                    "mean_cost_usd": f"{float(row['mean_cost_usd']):.9f}",
                    "vs_offline": (
                        f"{total / offline_cost:.4f}" if offline_cost > 0 else "n/a"
                    ),
                }
            )
    print(f"Saved: {path}")


def write_latency_percentile_table(
    rows: list[dict[str, str]],
    output_dir: Path,
    *,
    scenario: str,
) -> None:
    """Emit per-scenario latency percentile table covering Juncheng's full set.

    Captures mean / P10 / P25 / P50 / P75 / P90 / P99. Juncheng explicitly
    asked us to gather the wider percentile range, not just headline P50/P99.

    SLO violation rate is intentionally NOT included here. The cost layer is
    a controlled experiment over cost only — latency is held identical across
    providers by construction, so SLO violation is a property of the chosen
    distribution, not of the routing policy. The simulator still records
    `slo_violation_rate` in `summary.csv` for latency-layer / hedging /
    end-to-end sections to consume; cost-layer paper artifacts should not
    surface it.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"cost_layer_on_demand_{SCENARIO_SLUGS[scenario]}_latency_table.csv"
    fieldnames = (
        "policy",
        "mean_ttft_ms",
        "p10_ms",
        "p25_ms",
        "p50_ms",
        "p75_ms",
        "p90_ms",
        "p99_ms",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for policy in POLICIES:
            row = summary_row(rows, scenario=scenario, policy=policy)
            writer.writerow(
                {
                    "policy": POLICY_LABELS[policy],
                    "mean_ttft_ms": f"{float(row['mean_ttft_ms']):.3f}",
                    "p10_ms": f"{float(row['p10_ms']):.3f}",
                    "p25_ms": f"{float(row['p25_ms']):.3f}",
                    "p50_ms": f"{float(row['p50_ms']):.3f}",
                    "p75_ms": f"{float(row['p75_ms']):.3f}",
                    "p90_ms": f"{float(row['p90_ms']):.3f}",
                    "p99_ms": f"{float(row['p99_ms']):.3f}",
                }
            )
    print(f"Saved: {path}")


def make_on_demand_plots(input_dir: Path, output_dir: Path) -> None:
    """Generate all cost-layer on-demand simulator figures and tables.

    Per Juncheng's Notion metric list, every scenario emits all four
    artifacts: cost (bar + table), provider fraction (stacked bar),
    TTFT distribution (CDF), and latency percentile table covering
    mean / P10 / P25 / P50 / P75 / P90 / P99. Final paper figure
    selection is editorial and happens later.
    """
    apply_on_demand_style()
    summary_rows = load_summary(input_dir)
    histogram_rows = load_histograms(input_dir)
    for scenario in SCENARIOS:
        plot_total_cost(summary_rows, output_dir, scenario=scenario)
        write_cost_table(summary_rows, output_dir, scenario=scenario)
        plot_provider_mix(summary_rows, output_dir, scenario=scenario)
        plot_ttft_cdf(summary_rows, histogram_rows, output_dir, scenario=scenario)
        write_latency_percentile_table(summary_rows, output_dir, scenario=scenario)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Simulator output directory containing summary.csv and ttft_histograms.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure output directory. Defaults to <input-dir>/figures.",
    )
    args = parser.parse_args(argv)

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir / "figures"
    make_on_demand_plots(input_dir, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
