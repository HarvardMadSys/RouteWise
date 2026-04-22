"""Alpha-weighted joint LP sweep on the tiered scenarios.

Extends the joint router's effective cost with a latency penalty term
(Juncheng 2026-04-21 proposal):

    c_eff[j] = marginal_cost[j] + psi(z_j) + lambda(u_j) + alpha * p50_j

At alpha = 0 the router behaves as the original joint_ucb_hedge. As alpha
grows the objective rewards faster providers, giving the operator a
single knob to trade cost for latency across all tiers simultaneously.

Output
------
results/alpha_joint/{scenario}/
    summary.json              all alpha points for this scenario
    pareto_cost_vs_p99.png    cost / P99 Pareto curve across alpha
    tier_mix_vs_alpha.png     how tier fractions shift as alpha grows

Usage (from routewise-simulator-joint/ root, venv active):
    python run_joint_alpha_sweep.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiment.scripts.simulate.synthetic.tiered.scenarios import (  # noqa: E402
    make_tiered_scenarios,
)
from experiment.scripts.simulate.synthetic.tiered.scenarios_calibrated import (  # noqa: E402
    make_calibrated_scenarios,
)
from experiment.scripts.simulate.synthetic.tiered.scenarios_mm25 import (  # noqa: E402
    make_mm25_scenarios,
)
from experiment.scripts.simulate.synthetic.tiered.strategies import (  # noqa: E402
    _run_joint_ucb,
)
from experiment.scripts.simulate.synthetic.workload import (  # noqa: E402
    generate_workload,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Alpha in USD per ms. Effective-cost units are USD per request, and P50 is
# in ms, so alpha * p50 has natural units USD. The crossover point depends
# on the envelope U (max API cost / request) which is ~6e-4 for the S_A
# providers in the current scenarios; a reasonable sweep brackets U/1000.
ALPHAS = [
    0.0,
    1e-8,
    1e-7,
    3e-7,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
]

SEEDS = [42, 43, 44]
OUTPUT_ROOT = _ROOT / "results" / "alpha_joint_mm25"
SCENARIOS = [
    "s6m_featherless_saturation",
    "s7m_quota_depletion",
    "s8m_multi_sq_hierarchy",
]


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _avg(runs, fn) -> float:
    return float(np.mean([fn(r) for r in runs]))


def summarize_runs(alpha: float, runs, scenario) -> dict:
    entry: dict = {"alpha": alpha}
    for slo in scenario.slo_thresholds_ms:
        entry[f"slo_violation_rate_{int(slo)}ms"] = _avg(
            runs, lambda r, s=slo: r.slo_violation_rate(s)
        )
    entry["mean_cost_usd"] = _avg(runs, lambda r: r.mean_cost_usd())
    entry["p50_ms"] = _avg(runs, lambda r: r.p50_ms())
    entry["p99_ms"] = _avg(runs, lambda r: r.p99_ms())
    entry["hedge_rate"] = _avg(runs, lambda r: float(np.mean(r.hedge_triggered)))

    # Tier mix averaged across seeds.
    tier_lists: dict[str, list[float]] = {}
    for r in runs:
        for tname, frac in r.tier_fractions().items():
            tier_lists.setdefault(tname, []).append(frac)
    entry["tier_fractions"] = {
        t: float(np.mean(fracs)) for t, fracs in sorted(tier_lists.items())
    }

    # Per-provider mix (helps see e.g. which S_A provider is selected).
    prov_lists: dict[str, list[float]] = {}
    for r in runs:
        names = set(r.provider)
        total = len(r.provider)
        for pname in names:
            frac = r.provider.count(pname) / total if total else 0.0
            prov_lists.setdefault(pname, []).append(frac)
    entry["provider_fractions"] = {
        p: float(np.mean(fracs)) for p, fracs in sorted(prov_lists.items())
    }
    return entry


# ---------------------------------------------------------------------------
# Plotting (one PNG per figure, no subplots)
# ---------------------------------------------------------------------------


def plot_pareto(scenario_id: str, summaries: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    alphas = [s["alpha"] for s in summaries]
    costs = [s["mean_cost_usd"] for s in summaries]
    p99s = [s["p99_ms"] for s in summaries]
    p50s = [s["p50_ms"] for s in summaries]

    # Figure 1: cost vs P99
    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
    ax.plot(costs, p99s, marker="o", linewidth=1.8, color="tab:green")
    for a, c, p in zip(alphas, costs, p99s):
        ax.annotate(
            f"α=0" if a == 0 else f"α={a:.0e}",
            xy=(c, p),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.75,
        )
    ax.set_xlabel("Mean cost per request (USD)", fontsize=12)
    ax.set_ylabel("P99 TTFT (ms)", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_cost_vs_p99.png")
    plt.close(fig)

    # Figure 2: cost vs P50
    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
    ax.plot(costs, p50s, marker="s", linewidth=1.8, color="tab:blue")
    for a, c, p in zip(alphas, costs, p50s):
        ax.annotate(
            f"α=0" if a == 0 else f"α={a:.0e}",
            xy=(c, p),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.75,
        )
    ax.set_xlabel("Mean cost per request (USD)", fontsize=12)
    ax.set_ylabel("P50 TTFT (ms)", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_cost_vs_p50.png")
    plt.close(fig)

    # Figure 3: tier mix vs alpha
    all_tiers = sorted({t for s in summaries for t in s["tier_fractions"]})
    alphas_plot = [max(a, 1e-10) for a in alphas]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
    for tname in all_tiers:
        fracs = [s["tier_fractions"].get(tname, 0.0) for s in summaries]
        ax.plot(alphas_plot, fracs, marker="o", linewidth=1.8, label=tname)
    ax.set_xscale("log")
    ax.set_xlabel("alpha (USD/ms, latency weight)", fontsize=12)
    ax.set_ylabel("Tier traffic share", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "tier_mix_vs_alpha.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    scenarios = make_mm25_scenarios()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for scenario_id in SCENARIOS:
        scenario = scenarios[scenario_id]
        print(f"\n{'=' * 60}")
        print(f"Scenario: {scenario_id}")
        print(f"  {scenario.description}")

        out_dir = OUTPUT_ROOT / scenario_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # Share the same workload across all alpha values for apples-to-apples.
        requests = generate_workload(
            n_requests=scenario.n_requests,
            duration_seconds=scenario.duration_seconds,
            seed=0,
            start_time=0.0,
            arrival_process=scenario.arrival_process,
        )

        summaries: list[dict] = []
        for alpha in ALPHAS:
            t0 = time.perf_counter()
            runs = []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                run = _run_joint_ucb(
                    scenario,
                    requests,
                    rng,
                    use_hedge=True,
                    strategy_name=f"joint_ucb_hedge_alpha_{alpha:.0e}",
                    latency_alpha=alpha,
                )
                runs.append(run)
            elapsed = time.perf_counter() - t0

            summary = summarize_runs(alpha, runs, scenario)
            summaries.append(summary)

            tier_str = ", ".join(
                f"{k}={v:.0%}" for k, v in summary["tier_fractions"].items()
            )
            print(
                f"  alpha={alpha:.0e}  cost={summary['mean_cost_usd']:.3e}  "
                f"P50={summary['p50_ms']:.0f}ms  P99={summary['p99_ms']:.0f}ms  "
                f"hedge={summary['hedge_rate']:.1%}  [{tier_str}]  ({elapsed:.1f}s)"
            )

        with open(out_dir / "summary.json", "w") as f:
            json.dump({"scenario": scenario_id, "alphas": summaries}, f, indent=2)
        plot_pareto(scenario_id, summaries, out_dir)

        print(f"  Saved summary + 3 PNGs to {out_dir}/")

    print(f"\nDone. Results in {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
