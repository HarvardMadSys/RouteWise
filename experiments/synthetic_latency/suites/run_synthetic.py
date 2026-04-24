"""Orchestrator: run all synthetic scenarios and save results.

Usage (from project root, with venv active):
    routewise suite synthetic

Output layout:
    outputs/synthetic/
        s1_dominant/
            summary.json
            slo_violation.png
            cost_comparison.png
            provider_selection.png
            latency_cdf.png
        s2_tradeoff/
            ...
        ...

Each summary.json has the structure:
    {
      "<strategy>": {
        "slo_violation_rate_1000ms": 0.05,
        "slo_violation_rate_2000ms": 0.02,
        "slo_violation_rate_3000ms": 0.01,
        "slo_violation_rate_5000ms": 0.005,
        "mean_cost_usd": 1.5e-5,
        "p50_ms": 120.0,
        "p99_ms": 800.0,
        "hedge_rate": 0.12,
        "provider_fractions": {"A": 0.1, "B": 0.9}
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

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.synthetic_latency import load_all_world_scenarios
from experiments.synthetic_latency.plots import make_plots
from rwsim.runner import LATENCY_STRATEGIES as STRATEGIES, run_registered_strategy
from rwsim.world import ScenarioConfig, StrategyRun, generate_workload

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEEDS = [42, 43, 44]          # seeds for latency sampling; averaged for summary
OUTPUT_ROOT = _ROOT / "outputs" / "synthetic"


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def _avg(runs: list[StrategyRun], fn) -> float:
    return float(np.mean([fn(r) for r in runs]))


def build_summary(
    scenario: ScenarioConfig,
    results: dict[str, list[StrategyRun]],
) -> dict:
    summary: dict = {}
    for strat, runs in results.items():
        entry: dict = {}
        for slo in scenario.slo_thresholds_ms:
            key = f"slo_violation_rate_{int(slo)}ms"
            entry[key] = _avg(runs, lambda r, s=slo: r.slo_violation_rate(s))
        entry["mean_cost_usd"] = _avg(runs, lambda r: r.mean_cost_usd())
        entry["p50_ms"] = _avg(runs, lambda r: r.p50_ms())
        entry["p99_ms"] = _avg(runs, lambda r: r.p99_ms())
        entry["hedge_rate"] = _avg(runs, lambda r: r.hedge_rate())
        # Provider fractions averaged across seeds.
        frac_lists: dict[str, list[float]] = {}
        for r in runs:
            for pname, frac in r.provider_fractions().items():
                frac_lists.setdefault(pname, []).append(frac)
        entry["provider_fractions"] = {
            p: float(np.mean(fracs)) for p, fracs in sorted(frac_lists.items())
        }
        summary[strat] = entry
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    scenarios = {scenario.name: scenario for scenario in load_all_world_scenarios()}

    for scenario_id, scenario in scenarios.items():
        print(f"\n{'=' * 60}")
        print(f"Scenario: {scenario_id}")
        print(f"  {scenario.description}")
        print(f"  Providers: {scenario.provider_names}")
        print(f"  Requests: {scenario.n_requests}, Duration: {scenario.duration_seconds:.0f}s")

        out_dir = OUTPUT_ROOT / scenario_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # Generate workload (shared across strategies and seeds).
        requests = generate_workload(
            n_requests=scenario.n_requests,
            duration_seconds=scenario.duration_seconds,
            seed=0,
            start_time=0.0,
            arrival_process=scenario.arrival_process,
        )
        print(f"  Generated {len(requests)} requests")

        # Run all strategies × seeds.
        results: dict[str, list[StrategyRun]] = {s: [] for s in STRATEGIES}
        for strategy in STRATEGIES:
            t0 = time.perf_counter()
            for seed in SEEDS:
                run = run_registered_strategy(scenario, requests, strategy, seed=seed)
                results[strategy].append(run)
            elapsed = time.perf_counter() - t0

            # Quick per-strategy summary to stdout.
            runs = results[strategy]
            viol_2s = _avg(runs, lambda r: r.slo_violation_rate(2000.0))
            cost = _avg(runs, lambda r: r.mean_cost_usd())
            pfrac = runs[0].provider_fractions()
            pfrac_str = ", ".join(f"{k}={v:.0%}" for k, v in sorted(pfrac.items()))
            print(
                f"  [{strategy:<20s}]  "
                f"SLO(2s)={viol_2s:.1%}  cost={cost:.2e}  "
                f"providers=[{pfrac_str}]  ({elapsed:.1f}s)"
            )

        # Write summary JSON.
        summary = build_summary(scenario, results)
        summary_path = out_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved {summary_path}")

        # Generate plots.
        make_plots(
            scenario_name=scenario_id,
            results=results,
            output_dir=out_dir,
            slo_thresholds_ms=scenario.slo_thresholds_ms,
            primary_slo_ms=scenario.primary_slo_ms,
        )
        print(f"  Plots saved to {out_dir}/")

    print(f"\nDone. Results in {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
