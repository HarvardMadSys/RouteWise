#!/usr/bin/env python3
"""Plot oracle experiment results.

Generates a CR comparison chart across strategies for oracle vs predicted.
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np

# ICML-style configuration
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    }
)

COLORS = {
    "prediction": "#ED7D31",  # Orange - prediction gap
}


def load_results(results_path: str) -> dict:
    """Load oracle experiment results."""
    with open(results_path) as f:
        return json.load(f)


def plot_cr_comparison(results: dict, output_dir: Path) -> None:
    """Plot CR comparison across all strategies for Pro plan.

    Args:
        results: Oracle experiment results
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    datasets = ["burstgpt", "freeinference"]
    titles = ["BurstGPT (Q=5000)", "FreeInference (Q=5000)"]

    strategy_order = [
        "Optimal",
        "PD-Oracle",
        "PD-EMA",
        "PD-Hist",
        "LA-Oracle",
        "LA-EMA",
        "LA-Hist",
        "Greedy",
    ]
    strategy_colors = {
        "Optimal": "#70AD47",
        "PD-Oracle": "#4472C4",
        "PD-EMA": "#5B9BD5",
        "PD-Hist": "#9DC3E6",
        "LA-Oracle": "#C55A11",
        "LA-EMA": "#ED7D31",
        "LA-Hist": "#F4B183",
        "Greedy": "#A5A5A5",
    }

    for idx, (ds_name, title) in enumerate(zip(datasets, titles, strict=False)):
        ax = axes[idx]

        if ds_name not in results or "Pro" not in results[ds_name]["plans"]:
            ax.set_title(f"{title}\n(no data)")
            continue

        pro = results[ds_name]["plans"]["Pro"]["analysis"]["strategies"]

        crs = []
        colors = []
        labels = []
        for s in strategy_order:
            if s in pro:
                crs.append(pro[s]["cr"])
                colors.append(strategy_colors[s])
                labels.append(s)

        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, crs, color=colors, edgecolor="white", linewidth=0.5)

        # Set x-axis to focus on the interesting range
        cr_min = min(crs)
        cr_max = max(crs)
        x_start = max(0.95, cr_min - 0.02)
        x_end = cr_max + 0.06
        ax.set_xlim(x_start, x_end)

        # Add value labels
        for bar, cr in zip(bars, crs, strict=False):
            ax.text(
                bar.get_width() + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{cr:.4f}x",
                va="center",
                fontsize=8,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Competitive Ratio (lower = better)")
        ax.set_title(title)
        ax.axvline(x=1.0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.invert_yaxis()

        # Highlight oracle gap
        if "PD-Oracle" in pro and "PD-EMA" in pro:
            oracle_cr = pro["PD-Oracle"]["cr"]
            ema_cr = pro["PD-EMA"]["cr"]
            ax.axvspan(oracle_cr, ema_cr, alpha=0.15, color=COLORS["prediction"])

    plt.suptitle(
        "Competitive Ratio Comparison: Oracle vs Predicted (Pro Plan)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    output_path = output_dir / "oracle_cr_comparison.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Generate oracle experiment plots."""
    results_path = project_root / "experiment/results/oracle/oracle_experiment_results.json"
    output_dir = project_root / "experiment/results/oracle"

    if not results_path.exists():
        print(f"Results not found: {results_path}")
        print("Run experiment/scripts/run_oracle_experiment.py first.")
        sys.exit(1)

    results = load_results(str(results_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_cr_comparison(results, output_dir)

    print("\nDone! Plots saved to:", output_dir)


if __name__ == "__main__":
    main()
