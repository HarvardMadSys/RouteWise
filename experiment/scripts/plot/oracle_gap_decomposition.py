#!/usr/bin/env python3
"""Plot oracle gap decomposition for slide.

Generates a stacked bar chart showing Online Gap vs Prediction Gap
across all dataset/plan combinations.
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


def main():
    results_path = project_root / "experiment/results/oracle/oracle_experiment_results.json"
    with open(results_path) as f:
        results = json.load(f)

    # Extract gap decomposition data (PD strategy)
    labels = []
    online_gaps = []
    pred_gaps = []

    settings = [
        ("burstgpt", "Base", "Burst-Base"),
        ("burstgpt", "Plus", "Burst-Plus"),
        ("burstgpt", "Pro", "Burst-Pro"),
        ("freeinference", "Base", "FreeInf-Base"),
        ("freeinference", "Plus", "FreeInf-Plus"),
        ("freeinference", "Pro", "FreeInf-Pro"),
    ]

    for ds, plan, label in settings:
        analysis = results[ds]["plans"][plan]["analysis"]
        strategies = analysis["strategies"]

        ema_savings = strategies["PD-EMA"]["savings_pct"]
        oracle_savings = strategies["PD-Oracle"]["savings_pct"]
        opt_savings = strategies["Optimal"]["savings_pct"]

        online_gap = opt_savings - oracle_savings
        pred_gap = oracle_savings - ema_savings

        labels.append(label)
        online_gaps.append(online_gap)
        pred_gaps.append(max(pred_gap, 0))  # Clip negative to 0 for visualization

    # Create figure
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    y = np.arange(len(labels))
    bar_height = 0.55

    # Stacked horizontal bars
    bars_online = ax.barh(
        y, online_gaps, bar_height,
        label="Online Gap (unknown future)",
        color="#4472C4", edgecolor="white", linewidth=0.5,
    )
    bars_pred = ax.barh(
        y, pred_gaps, bar_height, left=online_gaps,
        label="Prediction Gap (unknown length)",
        color="#ED7D31", edgecolor="white", linewidth=0.5,
    )

    # Add value annotations
    for i in range(len(labels)):
        total = online_gaps[i] + pred_gaps[i]
        # Online gap label
        if online_gaps[i] > 1.5:
            ax.text(
                online_gaps[i] / 2, y[i],
                f"{online_gaps[i]:.1f}%",
                ha="center", va="center", fontsize=7.5, color="white", fontweight="bold",
            )
        # Prediction gap label (outside if too small)
        if pred_gaps[i] > 0.8:
            ax.text(
                online_gaps[i] + pred_gaps[i] / 2, y[i],
                f"{pred_gaps[i]:.1f}%",
                ha="center", va="center", fontsize=7.5, color="white", fontweight="bold",
            )
        else:
            ax.text(
                total + 0.3, y[i],
                f"{pred_gaps[i]:.1f}%",
                ha="left", va="center", fontsize=7.5, color="#ED7D31", fontweight="bold",
            )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Gap vs Optimal (% savings)")
    ax.invert_yaxis()
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(0, max(online_gaps[i] + pred_gaps[i] for i in range(len(labels))) + 2.5)

    plt.tight_layout()

    # Save to slide/images
    output_path = project_root / "slide/images/oracle_gap_decomposition.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")

    # Also save to experiment results
    output_path2 = project_root / "experiment/results/oracle/oracle_gap_decomposition.png"
    fig.savefig(output_path2, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path2}")

    plt.close()


if __name__ == "__main__":
    main()
