"""Run stress-test scenarios (ST1-ST3) for the joint router.

Usage:
    python run_stress_tests.py

Output:
    results/stress/
        st1_multi_s_a/
            summary.json
            provider_mix.png
            slo_cost_pareto.png
        st2_s_q_degradation/
            summary.json
            tier_over_time.png     <- special plot showing S_Q->S_A switchover
            slo_cost_pareto.png
        st3_multi_day_rollover/
            summary.json
            quota_over_time.png    <- shows z oscillation across days
            slo_cost_pareto.png
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from legacy.experiment.scripts.simulate.synthetic.tiered import (
    TIERED_STRATEGIES,
    StrategyRun,
    run_tiered_scenario,
)
from legacy.experiment.scripts.simulate.synthetic.tiered.plots import (
    plot_provider_mix,
    plot_slo_cost_pareto,
)
from legacy.experiment.scripts.simulate.synthetic.tiered.stress_scenarios import (
    make_stress_scenarios,
)
from legacy.experiment.scripts.simulate.synthetic.workload import generate_workload


SEEDS = [42, 43, 44]
OUTPUT_ROOT = _ROOT / "results" / "stress"

# Focus: two_layer as baseline + current joint router variants.
FOCUS_STRATEGIES = ["two_layer", "joint_nohedge", "joint_hedge"]


def _avg(runs: list[StrategyRun], fn) -> float:
    return float(np.mean([fn(r) for r in runs]))


def build_summary(scenario, results: dict[str, list[StrategyRun]]) -> dict:
    summary: dict = {}
    for strat, runs in results.items():
        entry: dict = {}
        for slo in scenario.slo_thresholds_ms:
            key = f"slo_violation_rate_{int(slo)}ms"
            entry[key] = _avg(runs, lambda r, s=slo: r.slo_violation_rate(s))
        entry["mean_cost_usd"] = _avg(runs, lambda r: r.mean_cost_usd())
        entry["p50_ms"] = _avg(runs, lambda r: r.p50_ms())
        entry["p99_ms"] = _avg(runs, lambda r: r.p99_ms())
        entry["hedge_rate"] = float(
            np.mean([float(np.mean(r.hedge_triggered)) for r in runs])
        )

        frac_lists: dict[str, list[float]] = {}
        for r in runs:
            for tier, frac in r.tier_fractions().items():
                frac_lists.setdefault(tier, []).append(frac)
        entry["tier_fractions"] = {
            t: float(np.mean(fs)) for t, fs in sorted(frac_lists.items())
        }
        summary[strat] = entry
    return summary


def plot_tier_over_time(
    scenario_name: str,
    results: dict[str, list[StrategyRun]],
    output_path: Path,
    focus_strategies: list[str],
) -> None:
    """Stacked area plot of tier selection fractions over simulated time."""
    fig, axes = plt.subplots(
        len(focus_strategies), 1,
        figsize=(10, 2.8 * len(focus_strategies)),
        sharex=True,
    )
    if len(focus_strategies) == 1:
        axes = [axes]

    tier_colors = {"quota": "#2ca02c", "concurrency": "#9467bd", "api": "#1f77b4"}
    tier_order = ["quota", "concurrency", "api"]

    for ax, strat in zip(axes, focus_strategies):
        if strat not in results:
            continue
        r = results[strat][0]
        if len(r.timestamp) == 0:
            continue
        mids, fracs = r.tier_fractions_over_time(window_sec=300.0)

        # Build stacked arrays in tier_order.
        arr = np.array([
            fracs.get(t, np.zeros(len(mids))) for t in tier_order
        ])

        times_min = np.array(mids) / 60.0
        ax.stackplot(
            times_min,
            arr,
            labels=tier_order,
            colors=[tier_colors[t] for t in tier_order],
            alpha=0.85,
        )
        ax.set_ylabel("fraction")
        ax.set_title(f"{strat}")
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        ax.set_ylim(0, 1.0)

    axes[-1].set_xlabel("Simulated time (min)")
    fig.suptitle(f"{scenario_name}: tier selection over time", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_quota_over_time(
    scenario_name: str,
    results: dict[str, list[StrategyRun]],
    output_path: Path,
    focus_strategies: list[str],
) -> None:
    """Plot quota fraction used (z) over time for ST3 (multi-day rollover)."""
    fig, ax = plt.subplots(figsize=(10, 3.6))
    colors = {
        "two_layer": "#1f77b4",
        "joint_nohedge": "#2ca02c",
        "joint_hedge": "#17becf",
    }

    for strat in focus_strategies:
        if strat not in results:
            continue
        r = results[strat][0]
        if len(r.quota_fraction_used) == 0:
            continue
        t_hours = (r.timestamp - r.timestamp[0]) / 3600.0
        ax.plot(
            t_hours, r.quota_fraction_used,
            label=strat,
            color=colors.get(strat, "gray"),
            linewidth=1.4,
            alpha=0.85,
        )

    # Mark day boundaries.
    for day in range(1, 4):
        ax.axvline(day * 24, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

    ax.set_xlabel("Simulated time (hours)")
    ax.set_ylabel("Quota fraction used (z)")
    ax.set_title(f"{scenario_name}: S_Q quota usage across days")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    scenarios = make_stress_scenarios()

    for scenario_id, scenario in scenarios.items():
        print(f"\n{'=' * 60}")
        print(f"Scenario: {scenario_id}")
        print(f"  {scenario.description}")
        print(f"  Providers: {scenario.provider_names}")
        print(f"  Requests: {scenario.n_requests}, "
              f"Duration: {scenario.duration_seconds:.0f}s")

        out_dir = OUTPUT_ROOT / scenario_id
        out_dir.mkdir(parents=True, exist_ok=True)

        requests = generate_workload(
            n_requests=scenario.n_requests,
            duration_seconds=scenario.duration_seconds,
            seed=0,
            start_time=0.0,
            arrival_process=scenario.arrival_process,
        )
        print(f"  Generated {len(requests)} requests")

        t0 = time.perf_counter()
        results = run_tiered_scenario(
            scenario, requests, seeds=SEEDS, strategies=FOCUS_STRATEGIES,
        )
        elapsed = time.perf_counter() - t0
        print(f"  All strategies done in {elapsed:.1f}s")

        for strategy in FOCUS_STRATEGIES:
            runs = results[strategy]
            viol = _avg(
                runs, lambda r: r.slo_violation_rate(scenario.primary_slo_ms)
            )
            cost = _avg(runs, lambda r: r.mean_cost_usd())
            tiers = runs[0].tier_fractions()
            tier_str = ", ".join(f"{k}={v:.0%}" for k, v in sorted(tiers.items()))
            p50 = _avg(runs, lambda r: r.p50_ms())
            p99 = _avg(runs, lambda r: r.p99_ms())
            print(
                f"  [{strategy:<20s}]  "
                f"SLO({scenario.primary_slo_ms:.0f}ms)={viol:.1%}  "
                f"cost={cost:.2e}  P50={p50:.0f}ms  P99={p99:.0f}ms  "
                f"tiers=[{tier_str}]"
            )

        summary = build_summary(scenario, results)
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Plots
        plot_slo_cost_pareto(
            scenario_id, scenario.primary_slo_ms,
            results, out_dir / "slo_cost_pareto.png",
        )
        plot_provider_mix(scenario_id, results, out_dir / "provider_mix.png")

        # Scenario-specific plots
        if scenario_id == "st2_s_q_degradation":
            plot_tier_over_time(
                scenario_id, results, out_dir / "tier_over_time.png",
                FOCUS_STRATEGIES,
            )
        if scenario_id == "st3_multi_day_rollover":
            plot_quota_over_time(
                scenario_id, results, out_dir / "quota_over_time.png",
                FOCUS_STRATEGIES,
            )

        print(f"  Results saved to {out_dir}/")

    print(f"\nDone. Results in {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
