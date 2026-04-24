"""Run Pareto frontier sweep across scenarios and plot cost vs P99.

Usage:
    source ../.venv/bin/activate
    python run_pareto.py

Output:
    results/pareto/
        {scenario}/
            points.json     # all (cost, P99, SLO, hedge_rate) points
            pareto.png      # cost vs P99 scatter with frontier
            pareto_slo.png  # cost vs SLO_violation scatter
"""

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

from legacy.experiment.scripts.simulate.synthetic.pareto import (
    sweep_scenario,
    pareto_front,
    ParetoPoint,
)
from legacy.experiment.scripts.simulate.synthetic.scenarios import make_scenarios
from legacy.experiment.scripts.simulate.synthetic.workload import generate_workload


# Focus on scenarios where Pareto trade-off is meaningful.
# S1 and S5 have near-zero SLO violation for all strategies, so Pareto is flat.
SCENARIOS = ["s2_tradeoff", "s3_tail", "s4_shift"]

# Sweep grid. Keep cost_ratios dense, bands coarse.
COST_RATIOS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
P50_BANDS = [0.05, 0.10, 0.30]

OUT = _ROOT / "results" / "pareto"


FAMILY_STYLE = {
    "baseline":    {"color": "#333333", "marker": "s", "size": 120, "zorder": 5},
    "v2_hedge":    {"color": "#aec7e8", "marker": "o", "size": 60,  "zorder": 2},
    "v2_explorer": {"color": "#1f77b4", "marker": "o", "size": 80,  "zorder": 3},
    "lp_hedge":    {"color": "#ff9896", "marker": "^", "size": 60,  "zorder": 2},
    "lp_explorer": {"color": "#d62728", "marker": "^", "size": 80,  "zorder": 3},
}


def plot_pareto(points: list[ParetoPoint], scenario_name: str, out_dir: Path) -> None:
    """Cost vs P99 with Pareto frontier highlighted."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, y_key, y_label in [
        (axes[0], "p99_ms", "P99 TTFT (ms)"),
        (axes[1], "slo_violation_rate", "SLO@2s violation rate"),
    ]:
        # scatter by family
        for family in ["v2_hedge", "v2_explorer", "lp_hedge", "lp_explorer", "baseline"]:
            fp = [p for p in points if p.family == family]
            if not fp:
                continue
            xs = [p.mean_cost for p in fp]
            ys = [getattr(p, y_key) for p in fp]
            style = FAMILY_STYLE[family]
            ax.scatter(
                xs, ys,
                c=style["color"], marker=style["marker"],
                s=style["size"], alpha=0.75, zorder=style["zorder"],
                label=family,
                edgecolors="black" if family == "baseline" else "none",
                linewidths=0.5,
            )

        # Annotate baselines
        for p in [pp for pp in points if pp.family == "baseline"]:
            ax.annotate(
                p.label,
                (p.mean_cost, getattr(p, y_key)),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color="#333333",
            )

        # Pareto frontier (across all RouteWise points, including Explorer).
        rw_points = [p for p in points if p.family in
                     {"v2_hedge", "v2_explorer", "lp_hedge", "lp_explorer"}]
        front = pareto_front(rw_points, x_key="mean_cost", y_key=y_key)
        if len(front) >= 2:
            fx = [p.mean_cost for p in front]
            fy = [getattr(p, y_key) for p in front]
            ax.plot(fx, fy, "--", color="#2ca02c", linewidth=2, zorder=4,
                    label="RouteWise Pareto")

        ax.set_xlabel("Mean cost per request (USD)")
        ax.set_ylabel(y_label)
        ax.set_title(f"{scenario_name}: cost vs {y_label}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    plt.suptitle(
        f"Pareto Frontier ({scenario_name}): "
        f"RouteWise spans cost-latency trade-offs baselines cannot",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    out_path = out_dir / "pareto.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def main() -> None:
    all_scenarios = make_scenarios()
    scenarios = {k: all_scenarios[k] for k in SCENARIOS if k in all_scenarios}

    for sc_id, sc in scenarios.items():
        print(f"\n=== {sc_id} ===")
        out_dir = OUT / sc_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # Generate workload once, shared across all strategies.
        requests = generate_workload(
            n_requests=sc.n_requests,
            duration_seconds=sc.duration_seconds,
            seed=0,
            start_time=0.0,
            arrival_process=sc.arrival_process,
        )
        print(f"  generated {len(requests)} requests")

        # Run sweep.
        points = sweep_scenario(
            sc, requests,
            seed=42,
            cost_ratios=COST_RATIOS,
            p50_bands=P50_BANDS,
        )

        # Dump points.
        payload = [asdict(p) for p in points]
        with (out_dir / "points.json").open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"  {len(points)} points")

        # Summary to stdout.
        print(f"  {'family':<10s} {'label':<36s} {'cost':>10s} {'p99(ms)':>10s} {'slo%':>7s} {'hedge%':>7s}")
        for p in sorted(points, key=lambda q: q.mean_cost):
            print(
                f"  {p.family:<10s} {p.label:<36s} "
                f"{p.mean_cost:>10.3e} {p.p99_ms:>10.1f} "
                f"{p.slo_violation_rate*100:>6.2f}% {p.hedge_rate*100:>6.2f}%"
            )

        # Plot.
        plot_pareto(points, sc_id, out_dir)

    print(f"\nDone. Output in {OUT}/")


if __name__ == "__main__":
    main()
