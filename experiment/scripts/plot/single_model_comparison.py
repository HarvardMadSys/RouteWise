#!/usr/bin/env python3
"""Plot single-model comparison results for slide.

Shows how multiplier (model size) affects S_C subscription value.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np

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
    # Q=0 results (pure S_C vs API, no quota tier)
    # From gpu2 Gurobi run with featherless_scale, deepseek-r1
    q0_data = {
        1: {"ilp": 243.70, "daily": 367.77, "conc": 243.70},
        2: {"ilp": 282.59, "daily": 367.77, "conc": 282.59},
        4: {"ilp": 335.37, "daily": 367.77, "conc": 335.37},
    }

    # Q=5000 results (with quota tier)
    q5000_data = {
        1: {"ilp": 287.96, "daily": 163.19, "conc": 243.70},
        2: {"ilp": 293.29, "daily": 163.19, "conc": 282.59},
        4: {"ilp": 312.94, "daily": 163.19, "conc": 335.37},
    }

    multipliers = [1, 2, 4]
    eff_c = [8, 4, 2]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- Left: Q=0 (S_C value analysis) ---
    ax = axes[0]
    x = np.arange(len(multipliers))
    width = 0.25

    api_only = [q0_data[m]["daily"] for m in multipliers]
    sc_only = [q0_data[m]["conc"] for m in multipliers]

    bars_api = ax.bar(x - width / 2, api_only, width, label="API-Only", color="#A5A5A5", edgecolor="white")
    bars_sc = ax.bar(x + width / 2, sc_only, width, label="S_C + API (ILP)", color="#4472C4", edgecolor="white")

    # Add savings annotations
    for i, m in enumerate(multipliers):
        saving = api_only[i] - sc_only[i]
        pct = saving / api_only[i] * 100
        ax.annotate(
            f"${saving:.0f}\n({pct:.0f}%)",
            xy=(x[i] + width / 2, sc_only[i]),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center", va="top",
            fontsize=8, color="#4472C4", fontweight="bold",
        )

    ax.set_xlabel("Multiplier (Effective Concurrency)")
    ax.set_ylabel("Total Cost ($)")
    ax.set_title("Without Quota (Q=0): S_C Value")
    ax.set_xticks(x)
    ax.set_xticklabels([f"mult={m}\n(C_eff={c})" for m, c in zip(multipliers, eff_c)])
    ax.legend(loc="upper left")
    ax.set_ylim(0, 430)

    # --- Right: Q=5000 (joint optimization) ---
    ax = axes[1]

    ilp_vals = [q5000_data[m]["ilp"] for m in multipliers]
    daily_vals = [q5000_data[m]["daily"] for m in multipliers]
    conc_vals = [q5000_data[m]["conc"] for m in multipliers]

    bars_daily = ax.bar(x - width, daily_vals, width, label="S_Q Only", color="#70AD47", edgecolor="white")
    bars_ilp = ax.bar(x, ilp_vals, width, label="ILP (S_Q+S_C)", color="#4472C4", edgecolor="white")
    bars_conc = ax.bar(x + width, conc_vals, width, label="S_C Only", color="#ED7D31", edgecolor="white")

    # Add cost labels
    for i, m in enumerate(multipliers):
        ax.text(x[i] - width, daily_vals[i] + 5, f"${daily_vals[i]:.0f}", ha="center", va="bottom", fontsize=7.5)
        ax.text(x[i], ilp_vals[i] + 5, f"${ilp_vals[i]:.0f}", ha="center", va="bottom", fontsize=7.5)
        ax.text(x[i] + width, conc_vals[i] + 5, f"${conc_vals[i]:.0f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xlabel("Multiplier (Effective Concurrency)")
    ax.set_ylabel("Total Cost ($)")
    ax.set_title("With Quota (Q=5000): S_Q Dominates")
    ax.set_xticks(x)
    ax.set_xticklabels([f"mult={m}\n(C_eff={c})" for m, c in zip(multipliers, eff_c)])
    ax.legend(loc="upper left")
    ax.set_ylim(0, 430)

    plt.tight_layout()

    output_path = project_root / "slide/images/single_model_comparison.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
