"""Run the Llama-like phase diagram (varying per-provider tail structure)."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiment.scripts.simulate.synthetic.phase_diagram_v2 import (
    P50_SPREAD_AB,
    TAIL_RATIO_A,
    aggregate_cells,
    compute_winner_matrix,
    run_grid,
)


# Production anchors translated to the new axes.
# Qwen3-2507: WandB P50=257/P99=1068, DeepInfra P50=619 (stable-ish).
#   → P50 spread A→B ≈ 619/257 ≈ 2.4 (but here "B" must be stable, and
#     DeepInfra's P99=5493 so actually not stable; this anchor doesn't fit
#     cleanly — skip).
# Llama-3.3-70B: Friendli P50=301 P99=1282 (A); Novita P50=590 P99=1024 (B
#   stable); Parasail P50=466 (middle). Use A→B as Friendli→Novita.
#   spread_AB ≈ 590/301 ≈ 1.96, tail_A ≈ 1282/301 ≈ 4.26
PRODUCTION_ANCHORS = [
    ("Llama-3.3-70B", 2.0, 4.3),
]

OUT = _ROOT / "results" / "phase_diagram_v2"


def plot_heatmap(agg, p50_spreads, tail_ratios, out_path: Path) -> None:
    magnitude = compute_winner_matrix(agg, p50_spreads, tail_ratios)

    fig, ax = plt.subplots(figsize=(11, 8))
    vmax = max(0.5, float(np.nanmax(np.abs(magnitude))))
    im = ax.imshow(
        magnitude, cmap="RdBu_r", aspect="auto", origin="lower",
        vmin=-vmax, vmax=vmax,
        extent=(-0.5, len(p50_spreads) - 0.5, -0.5, len(tail_ratios) - 0.5),
    )

    ax.set_xticks(range(len(p50_spreads)))
    ax.set_xticklabels([f"{v:.1f}x" for v in p50_spreads])
    ax.set_yticks(range(len(tail_ratios)))
    ax.set_yticklabels([f"{v:.1f}x" for v in tail_ratios])
    ax.set_xlabel("P50 spread A→B (B's P50 / A's P50); B is the stable backup", fontsize=11)
    ax.set_ylabel("Tail asymmetry of A (fastest provider): P99_A / P50_A", fontsize=11)
    ax.set_title(
        "LP vs V2 (both + Explorer) on Llama-like provider structure\n"
        "A: fastest P50 with variable tail. B: slower but stable (P99 = 1.3*P50).\n"
        "Red = V2 wins on P99.  Blue = LP wins on P99.  White = tie.",
        fontsize=10,
    )

    for i, sp in enumerate(p50_spreads):
        for j, tr in enumerate(tail_ratios):
            lp = agg.get((sp, tr, "lp_explorer"))
            v2 = agg.get((sp, tr, "v2_explorer"))
            if lp is None or v2 is None:
                continue
            lp_p99 = lp["p99_ms"]
            v2_p99 = v2["p99_ms"]
            rel = (lp_p99 - v2_p99) / min(lp_p99, v2_p99) if min(lp_p99, v2_p99) > 0 else 0
            if abs(rel) < 0.03:
                ax.text(i, j, f"≈\n{lp_p99:.0f}ms", ha="center", va="center",
                        fontsize=8, color="gray", style="italic")
            else:
                tag = "V2 wins" if v2_p99 < lp_p99 else "LP wins"
                pct = abs(rel) * 100
                color = "white" if abs(magnitude[j, i]) > 0.5 else "black"
                bold = "bold" if pct > 25 else "normal"
                ax.text(
                    i, j,
                    f"{tag} ({pct:.0f}%)\nLP:{lp_p99:.0f}\nV2:{v2_p99:.0f}",
                    ha="center", va="center", fontsize=8,
                    color=color, fontweight=bold,
                )

    # Production anchor.
    for name, sp, tr in PRODUCTION_ANCHORS:
        i = int(np.argmin([abs(s - sp) for s in p50_spreads]))
        j = int(np.argmin([abs(t - tr) for t in tail_ratios]))
        ax.scatter([i], [j], marker="D", s=180, edgecolor="black",
                   facecolor="yellow", zorder=5, linewidths=1.5)
        ax.annotate(
            f"{name}\n(production)",
            xy=(i, j),
            xytext=(i + 0.35, j + 0.35),
            fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="yellow", alpha=0.9, ec="black"),
            arrowprops=dict(arrowstyle="-", color="black", lw=0.8),
            zorder=6,
        )

    explanation = (
        "Provider structure per cell:\n"
        "  A: P50 = 200ms, tail varies along Y (P99 = P50·tail_ratio)\n"
        "  B: P50 = 200·spread, P99 = P50·1.3  ← stable backup\n"
        "  C: P50 = 600ms, P99 = 1200ms       ← slow fallback\n"
        "All providers have equal cost. V2 picks A 100% by P50 rank.\n"
        "LP diversifies across A/B/C via CDF constraint.\n"
        "\n"
        "When A has mild tail (bottom rows): V2 collapses to A, low P99.\n"
        "When A has heavy tail (top rows): V2 pays that tail;\n"
        "LP diversifies to B and wins on P99 (but both use Explorer,\n"
        "so V2's hedging rate adapts)."
    )
    ax.text(
        1.02, 0.02, explanation, transform=ax.transAxes, fontsize=8,
        verticalalignment="bottom", horizontalalignment="left",
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Running v2 phase diagram: {len(P50_SPREAD_AB)}x{len(TAIL_RATIO_A)} cells, 2 strat x 3 seeds")

    results = run_grid()

    raw_path = OUT / "raw.json"
    with raw_path.open("w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"saved raw {raw_path}")

    agg = aggregate_cells(results)
    cells_path = OUT / "cells.json"
    with cells_path.open("w") as f:
        json.dump(
            {f"{k[0]:.3f}_{k[1]:.3f}_{k[2]}": v for k, v in agg.items()},
            f, indent=2,
        )
    print(f"saved agg {cells_path}")

    plot_heatmap(agg, P50_SPREAD_AB, TAIL_RATIO_A, OUT / "heatmap_v2.png")


if __name__ == "__main__":
    main()
