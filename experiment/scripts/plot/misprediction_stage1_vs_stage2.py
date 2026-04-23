#!/usr/bin/env python3
"""Compare Stage 1 vs Stage 2 misprediction robustness.

Produces one PNG per quota plan, overlaying Stage 1 (S_Q only) and
Stage 2 (S_Q + S_C, C=2) bias curves. Each PNG is a standalone figure.

Outputs:
    misprediction_stage1_vs_stage2_{base,plus,pro}.png

Also produces a summary figure listing worst-case cost per config.

Usage:
    python experiment/scripts/plot/misprediction_stage1_vs_stage2.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _set_bias_xaxis(ax) -> None:
    ax.set_xscale("log")
    ax.set_xticks([0.5, 0.667, 0.8, 1.0, 1.25, 1.5, 2.0])
    ax.set_xticklabels(["0.5x", "0.67x", "0.8x", "1.0x", "1.25x", "1.5x", "2.0x"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.xaxis.set_minor_formatter(plt.NullFormatter())


def extract_bias_curve(plan_res: dict, vs: str = "oracle") -> tuple[list[float], list[float]]:
    """Extract (bias_factor, relative_cost) from a Stage-1 plan_res."""
    xs, ys = [], []
    for e in plan_res["bias_sweep"]:
        xs.append(e["bias_factor"])
        key = "relative_cost_vs_oracle" if vs == "oracle" else "relative_cost_vs_optimal"
        ys.append(e[key])
    return xs, ys


def extract_stage2_curve(stage2_res: dict) -> tuple[list[float], list[float]]:
    """Extract (bias_factor, relative_cost_vs_oracle) from a Stage-2 run."""
    xs, ys = [], []
    for e in stage2_res["bias_sweep"]:
        xs.append(e["bias_factor"])
        ys.append(e["relative_cost_vs_oracle"])
    return xs, ys


def plot_per_plan_comparison(
    plan_name: str,
    stage1_res: dict,
    stage2_res: dict,
    output_path: Path,
) -> None:
    """Overlay Stage 1 vs Stage 2 bias curves for one plan."""
    stage1_x, stage1_y = extract_bias_curve(stage1_res, vs="oracle")
    stage2_x, stage2_y = extract_stage2_curve(stage2_res)

    fig, ax = plt.subplots(figsize=(5.8, 4.3))
    ax.plot(
        stage1_x,
        stage1_y,
        marker="o",
        linewidth=2,
        markersize=7,
        color="#2c7fb8",
        label="Stage 1  (S_Q only)",
        zorder=3,
    )
    ax.plot(
        stage2_x,
        stage2_y,
        marker="s",
        linewidth=2,
        markersize=7,
        color="#e34a33",
        label=f"Stage 2  (S_Q + S_C, C={stage2_res['concurrency_limit']})",
        zorder=3,
    )
    ax.axhline(
        1.0,
        linestyle=":",
        color="black",
        alpha=0.4,
        linewidth=1.2,
        label="PD-Oracle",
    )
    _set_bias_xaxis(ax)
    ax.set_xlabel("Systematic bias factor")
    ax.set_ylabel("Relative Cost  (vs PD-Oracle)")
    q = stage1_res["quota"]
    ax.set_title(
        f"Stage 1 vs Stage 2 bias sensitivity  "
        f"(FreeInference, {plan_name} plan, Q={q})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_stage2_regimes(
    stage2_data: dict[str, dict],
    output_path: Path,
) -> None:
    """Overlay 3 plans' Stage 2 bias curves."""
    colors = {"Base": "#e34a33", "Plus": "#756bb1", "Pro": "#2c7fb8"}
    markers = {"Base": "o", "Plus": "s", "Pro": "^"}

    fig, ax = plt.subplots(figsize=(5.8, 4.3))
    for plan in ["Base", "Plus", "Pro"]:
        if plan not in stage2_data:
            continue
        res = stage2_data[plan]
        xs, ys = extract_stage2_curve(res)
        q = res["daily_quota"]
        ax.plot(
            xs,
            ys,
            marker=markers[plan],
            linewidth=2,
            markersize=7,
            color=colors[plan],
            label=f"{plan} (Q={q})",
            zorder=3,
        )
    ax.axhline(1.0, linestyle=":", color="black", alpha=0.4, linewidth=1.2)
    _set_bias_xaxis(ax)
    ax.set_xlabel("Systematic bias factor")
    ax.set_ylabel("Relative Cost  (vs PD-Oracle)")
    any_c = next(iter(stage2_data.values()))["concurrency_limit"]
    ax.set_title(
        f"Stage 2 bias sensitivity across quota plans  "
        f"(FreeInference, C={any_c})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, title="Quota plan")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage1-results",
        type=str,
        default="experiment/results/misprediction/misprediction_results.json",
    )
    parser.add_argument(
        "--stage2-dir",
        type=str,
        default="experiment/results/misprediction",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiment/results/misprediction",
    )
    args = parser.parse_args()

    with open(args.stage1_results) as f:
        stage1_all = json.load(f)
    stage1_fi = stage1_all["datasets"]["freeinference"]["plans"]

    stage2_data: dict[str, dict] = {}
    stage2_dir = Path(args.stage2_dir)
    for plan in ["base", "plus", "pro"]:
        path = stage2_dir / f"misprediction_stage2_{plan}_c2.json"
        if path.exists():
            with open(path) as f:
                stage2_data[plan.capitalize()] = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-plan comparison figures.
    for plan in ["Base", "Plus", "Pro"]:
        if plan not in stage2_data:
            continue
        s1 = stage1_fi.get(plan)
        if s1 is None:
            continue
        s2 = stage2_data[plan]
        plot_per_plan_comparison(
            plan,
            s1,
            s2,
            output_dir / f"misprediction_stage1_vs_stage2_{plan.lower()}.png",
        )

    # Stage 2 regime overlay.
    plot_stage2_regimes(
        stage2_data,
        output_dir / "misprediction_stage2_regimes.png",
    )


if __name__ == "__main__":
    main()
