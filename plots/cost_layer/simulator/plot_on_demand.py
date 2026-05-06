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
    "greedy_cost": "Greedy-cost",
    "random": "Random",
    "offline": "Offline",
    "ablation_lp_only_p0": "RouteWise p=0",
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
    "api_cheap": "#2ca02c",
    "api_mid": "#ff7f0e",
    "api_expensive": "#d62728",
}


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


def plot_total_cost(rows: list[dict[str, str]], output_dir: Path) -> None:
    """Plot total cost over the full workload."""
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    x_pos = np.arange(len(SCENARIOS))
    width = 0.18

    for idx, policy in enumerate(POLICIES):
        costs = [
            float(summary_row(rows, scenario=scenario, policy=policy)["total_cost_usd"])
            for scenario in SCENARIOS
        ]
        ax.bar(
            x_pos + (idx - 1.5) * width,
            costs,
            width,
            label=POLICY_LABELS[policy],
            color=POLICY_COLORS[policy],
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([SCENARIO_LABELS[scenario] for scenario in SCENARIOS])
    ax.set_ylabel("Total cost over 30-day trace ($)")
    ax.set_title("Cost layer 1.1: same latency, different on-demand prices")
    ax.legend(ncols=2, frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    save_figure(
        fig,
        output_dir,
        "cost_layer_1_1_total_cost",
        formats=["png", "pdf"],
    )
    plt.close(fig)


def plot_provider_mix(rows: list[dict[str, str]], output_dir: Path) -> None:
    """Plot provider routing fractions."""
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.3), sharey=True)

    for ax, scenario in zip(axes, SCENARIOS, strict=True):
        x_pos = np.arange(len(POLICIES))
        bottom = np.zeros(len(POLICIES))
        for provider in PROVIDER_ORDER:
            values = []
            for policy in POLICIES:
                row = summary_row(rows, scenario=scenario, policy=policy)
                mix = json.loads(row["provider_mix"])
                values.append(float(mix.get(provider, 0.0)))
            ax.bar(
                x_pos,
                values,
                bottom=bottom,
                color=PROVIDER_COLORS[provider],
                label=PROVIDER_LABELS[provider],
            )
            bottom += np.asarray(values)

        ax.set_title(SCENARIO_LABELS[scenario])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [POLICY_LABELS[policy].replace(" ", "\n") for policy in POLICIES],
            fontsize=8,
        )
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("Fraction of requests")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncols=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
    )
    save_figure(
        fig,
        output_dir,
        "cost_layer_1_1_provider_mix",
        formats=["png", "pdf"],
    )
    plt.close(fig)


def plot_ttft_cdf(
    summary_rows: list[dict[str, str]],
    histogram_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Plot TTFT CDF sanity check from histogram artifacts."""
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.8), sharey=True)

    for ax, scenario in zip(axes.ravel(), SCENARIOS, strict=True):
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
                linewidth=2.0,
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
            fontsize=9,
        )
        ax.set_title(SCENARIO_LABELS[scenario], fontsize=13, pad=4)
        ax.set_xscale("log")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.0, 1.01)
        ax.grid(True, which="both", alpha=0.18)
        ax.tick_params(axis="both", labelsize=9)

    for ax in axes[1, :]:
        ax.set_xlabel("TTFT (ms, log scale)", fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel("CDF", fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncols=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        fontsize=10,
    )
    fig.suptitle(
        "Cost layer 1.1 TTFT distribution sanity check",
        y=1.06,
        fontsize=14,
    )
    fig.subplots_adjust(top=0.82, hspace=0.32, wspace=0.18)
    save_figure(
        fig,
        output_dir,
        "cost_layer_1_1_ttft_cdf",
        formats=["png", "pdf"],
    )
    plt.close(fig)


def make_on_demand_plots(input_dir: Path, output_dir: Path) -> None:
    """Generate all cost-layer on-demand simulator figures."""
    apply_style("paper")
    summary_rows = load_summary(input_dir)
    histogram_rows = load_histograms(input_dir)
    plot_total_cost(summary_rows, output_dir)
    plot_provider_mix(summary_rows, output_dir)
    plot_ttft_cdf(summary_rows, histogram_rows, output_dir)


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
