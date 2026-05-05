"""Run paper policy presets on MiniMax-M2.5 calibrated scenarios.

Usage:
    routewise suite mm25_baselines

Output:
    outputs/mm25_baselines/{scenario}/baselines.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.simulation import make_mm25_scenarios  # noqa: E402
from experiments.simulation.lp_budget_eval import generate_scenario_workload  # noqa: E402
from rwsim.runner import POLICIES, run_policy  # noqa: E402


DEFAULT_OUTPUT_ROOT = _ROOT / "outputs" / "mm25_baselines"
SEEDS = [42, 43, 44]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MM25 calibrated baseline scenarios."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Directory where baseline outputs should be written. Defaults to "
            "outputs/mm25_baselines under the worktree root."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="burstgpt",
        help="Trace workload dataset to replay. Default: burstgpt.",
    )
    return parser.parse_args()


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
    args = _parse_args()
    scenarios = make_mm25_scenarios()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    for scenario_id, scenario in scenarios.items():
        print(f"\n=== {scenario_id} ===")
        requests = generate_scenario_workload(
            scenario,
            seed=0,
            dataset_name=args.dataset,
        )

        results = {}
        for policy_name in POLICIES:
            t0 = time.perf_counter()
            runs = [
                run_policy(scenario, requests, policy_name, seed=seed)
                for seed in SEEDS
            ]
            elapsed = time.perf_counter() - t0
            s = summarize(runs, scenario)
            results[policy_name] = s
            tier_s = ", ".join(f"{k}={v:.0%}" for k, v in s["tier_fractions"].items())
            print(
                f"  {policy_name:<24s} cost={s['mean_cost_usd']:.2e}  "
                f"P50={s['p50_ms']:.0f}ms  P99={s['p99_ms']:.0f}ms  "
                f"viol@SLO={s.get(f'slo_violation_rate_{int(scenario.primary_slo_ms)}ms', 0):.1%}  "
                f"[{tier_s}]  ({elapsed:.1f}s)"
            )

        out_dir = output_root / scenario_id
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "baselines.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"  saved baselines.json")


if __name__ == "__main__":
    main()
