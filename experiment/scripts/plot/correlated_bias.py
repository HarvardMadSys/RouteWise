#!/usr/bin/env python3
"""Plot input-correlated bias robustness results.

Produces one figure per (dataset, plan) config, overlaying the three
bias modes (long_underestimate, short_overestimate, tail_underestimate).
Each figure is a standalone PNG, no subplots.

Outputs:
    correlated_bias_{dataset}_{plan}_vs_oracle.png
    correlated_bias_{dataset}_{plan}_vs_optimal.png

Usage:
    python experiment/scripts/plot/correlated_bias.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

REFERENCE_PD_EMA_CR = 1.1807
REFERENCE_GREEDY_CR = 1.2986

MODE_LABELS = {
    "long_underestimate": "Long input under-est",
    "short_overestimate": "Short input over-est",
    "tail_underestimate": "Top-value under-est",
}
MODE_COLORS = {
    "long_underestimate": "#e34a33",
    "short_overestimate": "#2c7fb8",
    "tail_underestimate": "#756bb1",
}
MODE_MARKERS = {
    "long_underestimate": "o",
    "short_overestimate": "s",
    "tail_underestimate": "^",
}


def _set_bias_xaxis(ax) -> None:
    ax.set_xscale("log")
    ax.set_xticks([0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
    ax.set_xticklabels(["0.3x", "0.5x", "0.7x", "1.0x", "1.5x", "2.0x"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.xaxis.set_minor_formatter(plt.NullFormatter())


def plot_vs_oracle(plan_res: dict, output_path: Path, dataset: str, plan: str) -> None:
    """Overlay the 3 correlated-bias modes on one (vs Oracle) figure."""
    fig, ax = plt.subplots(figsize=(5.8, 4.3))
    for mode, entries in plan_res["sweeps"].items():
        xs = [e["bias_factor"] for e in entries]
        ys = [e["relative_cost_vs_oracle"] for e in entries]
        # Find biased fraction (constant across entries for a mode).
        biased_frac = next(
            (e.get("biased_fraction") for e in entries if e.get("biased_fraction")),
            None,
        )
        frac_str = f" ({biased_frac * 100:.0f}% biased)" if biased_frac else ""
        ax.plot(
            xs,
            ys,
            marker=MODE_MARKERS[mode],
            linewidth=2,
            markersize=7,
            color=MODE_COLORS[mode],
            label=MODE_LABELS[mode] + frac_str,
            zorder=3,
        )
    ax.axhline(
        1.0,
        linestyle=":",
        color="black",
        alpha=0.4,
        linewidth=1.2,
        label="PD-Oracle (perfect prediction)",
    )
    _set_bias_xaxis(ax)
    ax.set_xlabel("Bias factor applied to biased class  (predicted = true x factor)")
    ax.set_ylabel("Relative Cost  (vs PD-Oracle)")
    ax.set_title(
        f"Input-correlated bias  ({dataset.title()}, {plan} plan, Q={plan_res['quota']})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_vs_optimal(plan_res: dict, output_path: Path, dataset: str, plan: str) -> None:
    """Overlay the 3 correlated-bias modes on one (vs Optimal) figure."""
    fig, ax = plt.subplots(figsize=(5.8, 4.3))
    for mode, entries in plan_res["sweeps"].items():
        xs = [e["bias_factor"] for e in entries]
        ys = [e["relative_cost_vs_optimal"] for e in entries]
        biased_frac = next(
            (e.get("biased_fraction") for e in entries if e.get("biased_fraction")),
            None,
        )
        frac_str = f" ({biased_frac * 100:.0f}% biased)" if biased_frac else ""
        ax.plot(
            xs,
            ys,
            marker=MODE_MARKERS[mode],
            linewidth=2,
            markersize=7,
            color=MODE_COLORS[mode],
            label=MODE_LABELS[mode] + frac_str,
            zorder=3,
        )
    ax.axhline(
        REFERENCE_PD_EMA_CR,
        linestyle="--",
        color="#31a354",
        alpha=0.7,
        linewidth=1.3,
        label=f"PD-EMA ({REFERENCE_PD_EMA_CR:.2f}x)",
    )
    ax.axhline(
        REFERENCE_GREEDY_CR,
        linestyle="--",
        color="#e34a33",
        alpha=0.7,
        linewidth=1.3,
        label=f"Greedy ({REFERENCE_GREEDY_CR:.2f}x)",
    )
    ax.axhline(1.0, linestyle=":", color="black", alpha=0.4, linewidth=1.2)
    _set_bias_xaxis(ax)
    ax.set_xlabel("Bias factor applied to biased class")
    ax.set_ylabel("Relative Cost  (vs Offline Optimal)")
    ax.set_title(
        f"Input-correlated bias  ({dataset.title()}, {plan} plan, Q={plan_res['quota']})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot input-correlated bias results"
    )
    parser.add_argument(
        "--results",
        type=str,
        default="experiment/results/misprediction/correlated_bias_results.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiment/results/misprediction",
    )
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset, ds_res in data["datasets"].items():
        for plan, plan_res in ds_res["plans"].items():
            plot_vs_oracle(
                plan_res,
                output_dir / f"correlated_bias_{dataset}_{plan.lower()}_vs_oracle.png",
                dataset,
                plan,
            )
            plot_vs_optimal(
                plan_res,
                output_dir / f"correlated_bias_{dataset}_{plan.lower()}_vs_optimal.png",
                dataset,
                plan,
            )


if __name__ == "__main__":
    main()
