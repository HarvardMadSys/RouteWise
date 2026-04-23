"""Re-export step5 sanity-check plots as vector PDFs for the NSDI paper.

Reads the existing summary.json emitted by run_sanity_check.py and re-plots
the three response curves (A-share, cost, P99) with matplotlib, saving as
PDF instead of PNG. PNG remains the default output for development; PDF is
only produced here for paper inclusion where vector graphics are required.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


# Focused strategy set for paper (skip explorer_no_probe / oracle variants
# to keep the plot legible). lp_* = our LP-mix family, v2_* = legacy V2.
PAPER_STRATEGIES = [
    "cheapest_fixed",
    "fastest_fixed",
    "lp_mix",
    "lp_hedge",
    "v2_only",
    "v2_p50_hedge",
]

STRATEGY_LABELS = {
    "cheapest_fixed": "Cheapest fixed",
    "fastest_fixed": "Fastest fixed",
    "lp_mix": "LP-mix (ours)",
    "lp_hedge": "LP-mix + hedge (ours)",
    "v2_only": "V2 tier-first",
    "v2_p50_hedge": "V2 + hedge",
}

STRATEGY_COLORS = {
    "cheapest_fixed": "#888888",
    "fastest_fixed": "#BBBBBB",
    "lp_mix": "#1f77b4",
    "lp_hedge": "#2ca02c",
    "v2_only": "#d62728",
    "v2_p50_hedge": "#ff7f0e",
}


def load_step(summary_path: Path) -> list[dict]:
    data = json.loads(summary_path.read_text())
    return data["scenarios"]


def extract_x(scenarios: list[dict]) -> list[float]:
    return [s["providers"]["A"]["p50_ms"] for s in scenarios]


def extract_y(scenarios: list[dict], strat: str, field: str) -> list[float]:
    out = []
    for s in scenarios:
        entry = s["strategies"].get(strat)
        if entry is None:
            out.append(float("nan"))
            continue
        if field == "a_share":
            out.append(entry["provider_fractions"].get("A", 0.0))
        elif field == "cost":
            out.append(entry["mean_cost_usd"])
        elif field == "p99":
            out.append(entry["p99_ms"])
        else:
            raise ValueError(f"Unknown field: {field}")
    return out


def render(scenarios: list[dict], field: str, ylabel: str, out_path: Path) -> None:
    x = extract_x(scenarios)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for strat in PAPER_STRATEGIES:
        y = extract_y(scenarios, strat, field)
        ax.plot(
            x,
            y,
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=STRATEGY_LABELS[strat],
            color=STRATEGY_COLORS[strat],
        )
    ax.set_xlabel("Provider A P50 latency (ms)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, frameon=False)
    # Highlight the V2 band boundary at 110ms for step5 interpretation.
    ax.axvline(x=110, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(
        112, ax.get_ylim()[1] * 0.95,
        "V2 band boundary",
        fontsize=8, color="black", alpha=0.7,
    )
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    repo = Path("/Users/realtmxi/Desktop/NSDI2027_RouteWise")
    summary = repo / "routewise-simulator-joint" / "results" / "sanity_check" / "step5_latency_sweep" / "summary.json"
    scenarios = load_step(summary)
    # The paper repo lives at ~/Desktop/6991d665791ce21ba05287b8 (separate
    # from the NSDI2027_RouteWise working tree that hosts the simulator).
    out_dir = Path("/Users/realtmxi/Desktop/6991d665791ce21ba05287b8/images")
    out_dir.mkdir(parents=True, exist_ok=True)

    render(scenarios, "a_share", "Share of traffic to A", out_dir / "sanity_step5_a_share.pdf")
    render(scenarios, "cost", "Mean cost per request (USD)", out_dir / "sanity_step5_cost.pdf")
    render(scenarios, "p99", "P99 TTFT (ms)", out_dir / "sanity_step5_p99.pdf")


if __name__ == "__main__":
    main()
