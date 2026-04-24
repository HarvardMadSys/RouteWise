"""Orchestrator: run tiered scenarios (S6-S8) for joint vs two-layer comparison.

Usage (from project root, with venv active):
    python scripts/experiments/run_joint.py

Output layout:
    outputs/joint/
        s6_slow_q_trap/
            summary.json
        s7_quota_depletion/
            summary.json
        s8_concurrency_saturation/
            summary.json

Each summary.json has:
    {
      "<strategy>": {
        "slo_violation_rate_<n>ms": float,
        "mean_cost_usd": float,
        "p50_ms": float,
        "p99_ms": float,
        "hedge_rate": float,
        "tier_fractions": {"api": ..., "quota": ..., "concurrency": ...}
      },
      ...
    }
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.tiered_capacity import load_all_world_scenarios
from experiments.tiered_capacity.plots import (
    make_scenario_plots,
    plot_summary_across_scenarios,
)
from rwsim.runner import TIERED_STRATEGIES, run_registered_strategy
from rwsim.world import StrategyRun, generate_workload

SEEDS = [42, 43, 44]
OUTPUT_ROOT = _ROOT / "outputs" / "joint"


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
        entry["hedge_rate"] = float(np.mean([float(np.mean(r.hedge_triggered)) for r in runs]))

        # Tier fractions averaged across seeds.
        frac_lists: dict[str, list[float]] = {}
        for r in runs:
            for tier, frac in r.tier_fractions().items():
                frac_lists.setdefault(tier, []).append(frac)
        entry["tier_fractions"] = {
            t: float(np.mean(fs)) for t, fs in sorted(frac_lists.items())
        }
        summary[strat] = entry
    return summary


def main() -> None:
    scenarios = {scenario.name: scenario for scenario in load_all_world_scenarios()}

    all_results: dict[str, dict[str, list[StrategyRun]]] = {}
    slo_map: dict[str, float] = {}

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
        results = {
            strategy: [
                run_registered_strategy(scenario, requests, strategy, seed=seed)
                for seed in SEEDS
            ]
            for strategy in TIERED_STRATEGIES
        }
        elapsed = time.perf_counter() - t0
        print(f"  All strategies done in {elapsed:.1f}s")

        for strategy in TIERED_STRATEGIES:
            runs = results[strategy]
            viol = _avg(runs, lambda r: r.slo_violation_rate(scenario.primary_slo_ms))
            cost = _avg(runs, lambda r: r.mean_cost_usd())
            tiers = runs[0].tier_fractions()
            tier_str = ", ".join(f"{k}={v:.0%}" for k, v in sorted(tiers.items()))
            print(
                f"  [{strategy:<20s}]  "
                f"SLO({scenario.primary_slo_ms:.0f}ms)={viol:.1%}  "
                f"cost={cost:.2e}  tiers=[{tier_str}]"
            )

        summary = build_summary(scenario, results)
        summary_path = out_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved {summary_path}")

        make_scenario_plots(
            scenario_name=scenario_id,
            primary_slo_ms=scenario.primary_slo_ms,
            results=results,
            output_dir=out_dir,
        )
        print(f"  Plots saved to {out_dir}/")

        all_results[scenario_id] = results
        slo_map[scenario_id] = scenario.primary_slo_ms

    summary_fig = OUTPUT_ROOT / "summary.png"
    plot_summary_across_scenarios(all_results, slo_map, summary_fig)
    print(f"\nSummary figure: {summary_fig}")

    print(f"\nDone. Results in {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
