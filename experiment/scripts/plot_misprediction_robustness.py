"""Plot misprediction robustness for the NSDI paper.

Reads experiment/results/misprediction/misprediction_results.json and emits a
single 2-panel figure (BurstGPT | FreeInference) showing relative cost vs.
injected multiplicative bias.  Bias factor 1.0 = oracle.  x-axis is shown in
percentage terms: (bias_factor - 1) * 100.

Output: images/misprediction_robustness.{pdf,png}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


BIAS_TICKS = [-50, -33, -20, -10, 0, 10, 20, 50, 100]
NSDI_BIAS_TICKS = [-50, -20, -10, 0, 10, 20, 50]

PLAN_STYLE = {
    "Base": dict(color="#1f77b4", marker="o", label="Base (Q=300)"),
    "Plus": dict(color="#ff7f0e", marker="s", label="Plus (Q=2000)"),
    "Pro": dict(color="#2ca02c", marker="^", label="Pro (Q=5000)"),
}


def load(results_path: Path) -> dict:
    with results_path.open() as fh:
        return json.load(fh)


def filter_unique_bias(entries: list[dict]) -> list[dict]:
    """Keep only noise_std == 0 and unique bias factors."""
    seen = set()
    out = []
    for e in entries:
        if e.get("noise_std", 0.0) != 0.0:
            continue
        b = e["bias_factor"]
        if b in seen:
            continue
        seen.add(b)
        out.append(e)
    out.sort(key=lambda x: x["bias_factor"])
    return out


def plot_dataset(ax, ds_block: dict, include_extended: bool) -> None:
    ticks = BIAS_TICKS if include_extended else NSDI_BIAS_TICKS
    tick_set = {t / 100.0 + 1.0 for t in ticks}
    for plan, style in PLAN_STYLE.items():
        plan_block = ds_block["plans"].get(plan)
        if plan_block is None:
            continue
        entries = filter_unique_bias(plan_block["bias_sweep"])
        xs, ys = [], []
        for e in entries:
            b = e["bias_factor"]
            if b not in tick_set:
                continue
            xs.append((b - 1.0) * 100.0)
            ys.append(e["relative_cost_vs_optimal"])
        ax.plot(xs, ys, linewidth=1.8, markersize=6, **style)
    ax.axvline(0.0, color="#888888", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Injected bias in oracle predictor (%)")
    ax.set_ylabel(r"Relative cost vs.\ offline optimal ($\times$)")
    ax.set_xticks(ticks)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(
            "/Users/realtmxi/Desktop/NSDI2027_RouteWise/experiment/results/"
            "misprediction/misprediction_results.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/realtmxi/Desktop/6991d665791ce21ba05287b8/images"),
    )
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()

    data = load(args.results)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # Two separate figures, one per dataset, per project convention
    # (feedback_plot_style.md: each figure gets its own PNG).
    for ds_name, title in [
        ("burstgpt", "BurstGPT (1.4M requests)"),
        ("freeinference", "FreeInference (371K requests)"),
    ]:
        fig, ax = plt.subplots(figsize=(3.3, 2.4))
        plot_dataset(ax, data["datasets"][ds_name], args.extended)
        ax.set_title(title)
        ax.legend(loc="best", frameon=True)
        fig.tight_layout()
        stem = f"misprediction_{ds_name}"
        fig.savefig(args.out_dir / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(args.out_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {args.out_dir / stem}.pdf and .png")


if __name__ == "__main__":
    main()
