"""Run stress-test scenarios (ST1-ST3) for paper policy presets.

Usage:
    routewise suite stress

Output:
    outputs/stress/
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

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.simulation import make_stress_scenarios
from experiments.simulation.lp_budget_eval import generate_scenario_workload
from rwsim.metrics import SimulationRun
from rwsim.runner import POLICIES, run_policy


SEEDS = [42, 43, 44]
OUTPUT_ROOT = _ROOT / "outputs" / "stress"

FOCUS_POLICIES = ["greedy_cost", "ablation_lp_only", "ablation_lp_hedging", "routewise"]


def _avg(runs: list[SimulationRun], fn) -> float:
    return float(np.mean([fn(r) for r in runs]))


def build_summary(scenario, results: dict[str, list[SimulationRun]]) -> dict:
    summary: dict = {}
    for policy_name, runs in results.items():
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
        summary[policy_name] = entry
    return summary


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

        requests = generate_scenario_workload(
            scenario,
            seed=0,
            dataset_name="burstgpt",
        )
        print(f"  Loaded {len(requests)} trace requests")

        t0 = time.perf_counter()
        results = {
            policy_name: [
                run_policy(scenario, requests, policy_name, seed=seed)
                for seed in SEEDS
            ]
            for policy_name in FOCUS_POLICIES
        }
        elapsed = time.perf_counter() - t0
        print(f"  All policies done in {elapsed:.1f}s")

        for policy_name in FOCUS_POLICIES:
            runs = results[policy_name]
            viol = _avg(
                runs, lambda r: r.slo_violation_rate(scenario.primary_slo_ms)
            )
            cost = _avg(runs, lambda r: r.mean_cost_usd())
            tiers = runs[0].tier_fractions()
            tier_str = ", ".join(f"{k}={v:.0%}" for k, v in sorted(tiers.items()))
            p50 = _avg(runs, lambda r: r.p50_ms())
            p99 = _avg(runs, lambda r: r.p99_ms())
            print(
                f"  [{policy_name:<20s}]  "
                f"SLO({scenario.primary_slo_ms:.0f}ms)={viol:.1%}  "
                f"cost={cost:.2e}  P50={p50:.0f}ms  P99={p99:.0f}ms  "
                f"tiers=[{tier_str}]"
            )

        summary = build_summary(scenario, results)
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"  Results saved to {out_dir}/")

    print(f"\nDone. Results in {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
