"""Run baseline strategies (two_layer + joint variants without alpha) on
MiniMax-M2.5 calibrated scenarios, so the alpha sweep has an apples-to-
apples baseline to compare against.

Usage:
    python run_joint_mm25_baselines.py

Output:
    results/alpha_joint_mm25/{scenario}/baselines.json
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

from experiment.scripts.simulate.synthetic.tiered import (  # noqa: E402
    TIERED_STRATEGIES,
    run_tiered_scenario,
)
from experiment.scripts.simulate.synthetic.tiered.scenarios_mm25 import (  # noqa: E402
    make_mm25_scenarios,
)
from experiment.scripts.simulate.synthetic.workload import (  # noqa: E402
    generate_workload,
)


OUTPUT_ROOT = _ROOT / "results" / "alpha_joint_mm25"
SEEDS = [42, 43, 44]


def _avg(runs, fn) -> float:
    return float(np.mean([fn(r) for r in runs]))


def summarize(runs, scenario) -> dict:
    entry = {}
    for slo in scenario.slo_thresholds_ms:
        entry[f"slo_violation_rate_{int(slo)}ms"] = _avg(
            runs, lambda r, s=slo: r.slo_violation_rate(s)
        )
    entry["mean_cost_usd"] = _avg(runs, lambda r: r.mean_cost_usd())
    entry["p50_ms"] = _avg(runs, lambda r: r.p50_ms())
    entry["p99_ms"] = _avg(runs, lambda r: r.p99_ms())
    entry["hedge_rate"] = _avg(
        runs, lambda r: float(np.mean(r.hedge_triggered))
    )
    tier_lists = {}
    for r in runs:
        for t, f in r.tier_fractions().items():
            tier_lists.setdefault(t, []).append(f)
    entry["tier_fractions"] = {
        t: float(np.mean(fs)) for t, fs in sorted(tier_lists.items())
    }
    prov_lists = {}
    for r in runs:
        names = set(r.provider)
        total = len(r.provider)
        for pn in names:
            prov_lists.setdefault(pn, []).append(r.provider.count(pn) / total)
    entry["provider_fractions"] = {
        p: float(np.mean(fs)) for p, fs in sorted(prov_lists.items())
    }
    return entry


def main() -> None:
    scenarios = make_mm25_scenarios()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for scenario_id, scenario in scenarios.items():
        print(f"\n=== {scenario_id} ===")
        requests = generate_workload(
            n_requests=scenario.n_requests,
            duration_seconds=scenario.duration_seconds,
            seed=0,
            start_time=0.0,
            arrival_process=scenario.arrival_process,
        )

        results = {}
        for strat in TIERED_STRATEGIES:
            t0 = time.perf_counter()
            runs = run_tiered_scenario(scenario, requests, seeds=SEEDS, strategies=[strat])[strat]
            elapsed = time.perf_counter() - t0
            s = summarize(runs, scenario)
            results[strat] = s
            tier_s = ", ".join(f"{k}={v:.0%}" for k, v in s["tier_fractions"].items())
            print(
                f"  {strat:<24s} cost={s['mean_cost_usd']:.2e}  "
                f"P50={s['p50_ms']:.0f}ms  P99={s['p99_ms']:.0f}ms  "
                f"viol@SLO={s.get(f'slo_violation_rate_{int(scenario.primary_slo_ms)}ms', 0):.1%}  "
                f"[{tier_s}]  ({elapsed:.1f}s)"
            )

        out_dir = OUTPUT_ROOT / scenario_id
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "baselines.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"  saved baselines.json")


if __name__ == "__main__":
    main()
