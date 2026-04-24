"""Run the LP vs V2 phase diagram sweep and produce the heatmap figure.

Usage:
    source ../.venv/bin/activate
    routewise suite phase_diagram

Output:
    outputs/phase_diagram/
        cells.json            # per-cell aggregated metrics
        raw.json              # per-seed metrics
        heatmap_p99.png       # main figure: V2 advantage over LP on P99
        heatmap_slo.png       # secondary: SLO violation winner
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.synthetic_latency.phase_diagram import (
    P50_SPREADS,
    TAIL_RATIOS,
    aggregate_cells,
    compute_winner_matrix,
    run_grid,
)


# Production anchors: overlay real-deployment regimes on the synthetic map.
# Numbers derived from our OpenRouter runs:
# Qwen3-235B-A22B-2507: WandB P50=257/P99=1068 dominant (99%), DeepInfra P50=619.
#   → Effective P50 spread ≈ 619/257 ≈ 2.4, tail ratio of fastest ≈ 1068/257 ≈ 4.2
# DeepSeek-V3.2: 12 providers, provider mix-heavy, moderate regime.
# Llama-3.3-70B: Friendli P50=301 / P99=1282, Novita P50=590.
#   → P50 spread ≈ 590/301 ≈ 2.0, tail ratio of fastest ≈ 1282/301 ≈ 4.3
PRODUCTION_ANCHORS = [
    ("Qwen3-235B-2507", 2.4, 4.2),
    ("Llama-3.3-70B",   2.0, 4.3),
    # DeepSeek-V3.2 had 12 providers; harder to reduce to (spread, tail).
    # Skip for now; mention in caption.
]


OUT = _ROOT / "outputs" / "phase_diagram"


def plot_heatmap_p99(agg, p50_spreads, tail_ratios, out_path: Path) -> None:
    """Heatmap: V2 advantage over LP on P99. Positive = V2 wins."""
    winner, magnitude = compute_winner_matrix(agg, p50_spreads, tail_ratios)

    fig, ax = plt.subplots(figsize=(11, 8))
    vmax = max(0.5, float(np.nanmax(np.abs(magnitude))))
    im = ax.imshow(
        magnitude,
        cmap="RdBu_r",
        aspect="auto",
        origin="lower",
        vmin=-vmax, vmax=vmax,
        extent=(-0.5, len(p50_spreads) - 0.5, -0.5, len(tail_ratios) - 0.5),
    )

    ax.set_xticks(range(len(p50_spreads)))
    ax.set_xticklabels([f"{v:.1f}x" for v in p50_spreads])
    ax.set_yticks(range(len(tail_ratios)))
    ax.set_yticklabels([f"{v:.1f}x" for v in tail_ratios])
    ax.set_xlabel("P50 spread (slowest provider P50 / fastest provider P50)", fontsize=11)
    ax.set_ylabel("Tail asymmetry of fastest provider (P99 / P50)", fontsize=11)
    ax.set_title(
        "LP vs V2 (both + Explorer): where does each win on P99?\n"
        "Red: V2 concentrates on fastest-P50 → lower P99 when that provider has controllable tail.\n"
        "Blue: LP diversifies via CDF constraint → lower P99 in narrow-spread / light-tail regime.\n"
        "White (≈): both algorithms collapse to the same primary provider (degenerate).",
        fontsize=10,
    )

    # Identify degenerate region (LP and V2 pick same answer).
    for i, sp in enumerate(p50_spreads):
        for j, tr in enumerate(tail_ratios):
            lp = agg.get((sp, tr, "lp_explorer"))
            v2 = agg.get((sp, tr, "v2_explorer"))
            if lp is None or v2 is None:
                continue
            lp_p99 = lp["p99_ms"]
            v2_p99 = v2["p99_ms"]
            rel_delta = abs(lp_p99 - v2_p99) / min(lp_p99, v2_p99) if min(lp_p99, v2_p99) > 0 else 0
            if rel_delta < 0.02:
                # Degenerate cell: both algorithms agree. Keep text minimal.
                label = f"both {lp_p99:.0f}ms"
                color = "gray"
                ax.text(i, j, label, ha="center", va="center",
                        fontsize=8, color=color, style="italic")
            else:
                winner_tag = "V2 wins" if v2_p99 < lp_p99 else "LP wins"
                delta_pct = rel_delta * 100
                color = "white" if abs(magnitude[j, i]) > 0.5 else "black"
                ax.text(
                    i, j,
                    f"{winner_tag} ({delta_pct:+.0f}%)\nLP:{lp_p99:.0f}ms\nV2:{v2_p99:.0f}ms",
                    ha="center", va="center", fontsize=8,
                    color=color, fontweight="bold" if delta_pct > 20 else "normal",
                )

    # Overlay production anchors with explanatory label.
    for name, sp, tr in PRODUCTION_ANCHORS:
        i = float(np.argmin([abs(s - sp) for s in p50_spreads]))
        j = float(np.argmin([abs(t - tr) for t in tail_ratios]))
        ax.scatter([i], [j], marker="D", s=160, edgecolor="black",
                   facecolor="yellow", zorder=5, linewidths=1.5)
        ax.annotate(
            name,
            xy=(i, j),
            xytext=(i + 0.35, j + 0.35),
            fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="yellow", alpha=0.9, ec="black"),
            arrowprops=dict(arrowstyle="-", color="black", lw=0.8),
            zorder=6,
        )

    # Add explanatory box in bottom-right of plot.
    explanation = (
        "Note: In cells where LP ≈ V2 (white), both algorithms\n"
        "route 100% to the fastest-P50 provider because it\n"
        "dominates. The meaningful regime is the left column\n"
        "(narrow P50 spread), where V2 concentrates on the\n"
        "fastest-but-tail-heavy provider while LP diversifies.\n"
        "\n"
        "Real deployments (yellow diamonds) fall in the narrow\n"
        "spread/moderate tail corner — the regime where V2's\n"
        "concentration benefit trades against LP's safety."
    )
    ax.text(
        1.02, 0.02,
        explanation,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.95),
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.12)
    cbar.set_label(
        "V2 advantage on P99: (LP.P99 − V2.P99) / min(LP, V2)\n(positive = V2 faster)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_heatmap_slo(agg, p50_spreads, tail_ratios, out_path: Path) -> None:
    """Secondary: SLO violation rate at SLO=2s, for both policies side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, strategy, title in [
        (axes[0], "lp_explorer", "LP + Explorer: SLO@2s violation rate"),
        (axes[1], "v2_explorer", "V2 + Explorer: SLO@2s violation rate"),
    ]:
        n_sp = len(p50_spreads)
        n_tr = len(tail_ratios)
        mat = np.full((n_tr, n_sp), np.nan)
        for i, sp in enumerate(p50_spreads):
            for j, tr in enumerate(tail_ratios):
                v = agg.get((sp, tr, strategy))
                if v is not None:
                    mat[j, i] = v["slo_violation_rate_2s"]

        im = ax.imshow(
            mat,
            cmap="YlOrRd",
            aspect="auto",
            origin="lower",
            vmin=0, vmax=max(0.3, float(np.nanmax(mat))),
        )
        ax.set_xticks(range(n_sp))
        ax.set_xticklabels([f"{v:.1f}x" for v in p50_spreads])
        ax.set_yticks(range(n_tr))
        ax.set_yticklabels([f"{v:.1f}x" for v in tail_ratios])
        ax.set_xlabel("P50 spread")
        ax.set_ylabel("Tail asymmetry of fastest")
        ax.set_title(title)

        for i, sp in enumerate(p50_spreads):
            for j, tr in enumerate(tail_ratios):
                v = agg.get((sp, tr, strategy))
                if v is None:
                    continue
                val = v["slo_violation_rate_2s"] * 100
                ax.text(i, j, f"{val:.1f}%", ha="center", va="center",
                        fontsize=8, color="white" if val > 15 else "black")

        fig.colorbar(im, ax=ax, shrink=0.7, label="SLO violation fraction")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Running phase diagram: {len(P50_SPREADS)} x {len(TAIL_RATIOS)} = "
          f"{len(P50_SPREADS) * len(TAIL_RATIOS)} cells, 2 strategies, 3 seeds.")
    print(f"Expected total runs: {len(P50_SPREADS) * len(TAIL_RATIOS) * 2 * 3}")

    results = run_grid()

    # Dump raw per-seed metrics.
    raw_path = OUT / "raw.json"
    with raw_path.open("w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"saved raw {raw_path}")

    # Aggregate and dump.
    agg = aggregate_cells(results)
    cells_path = OUT / "cells.json"
    with cells_path.open("w") as f:
        json.dump(
            {f"{k[0]:.3f}_{k[1]:.3f}_{k[2]}": v for k, v in agg.items()},
            f,
            indent=2,
        )
    print(f"saved agg {cells_path}")

    # Plot.
    plot_heatmap_p99(agg, P50_SPREADS, TAIL_RATIOS, OUT / "heatmap_p99.png")
    plot_heatmap_slo(agg, P50_SPREADS, TAIL_RATIOS, OUT / "heatmap_slo.png")

    print(f"\nDone. Outputs in {OUT}/")


if __name__ == "__main__":
    main()
