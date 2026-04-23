"""Run the full Joint-vs-Two-layer Pareto experiment on all scenarios.

Usage:
    python -m experiment.synthetic.runner \
        --output-dir experiment/results/synthetic \
        --n-seeds 5
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from experiment.synthetic.scenarios import Scenario, all_scenarios
from experiment.synthetic.simulator import SimResult, simulate
from experiment.synthetic.strategies import build_all_strategies


def run_scenario(scenario: Scenario, seeds: list[int]) -> list[SimResult]:
    """Run every strategy on `scenario` for each seed. Returns flat list."""
    results: list[SimResult] = []
    for seed in seeds:
        strategies = build_all_strategies(
            providers=scenario.providers,
            slo_ms=scenario.slo_ms,
        )
        for strat in strategies:
            res = simulate(
                strategy=strat,
                requests=scenario.requests,
                providers=scenario.providers,
                slo_ms=scenario.slo_ms,
                seed=seed,
                scenario_name=scenario.name,
            )
            results.append(res)
    return results


def aggregate(
    results: list[SimResult], slo_ms: float
) -> list[dict]:
    """Group by (scenario, strategy), compute mean and 95% CI across seeds."""
    summaries_by_key: dict[tuple[str, str], list[dict]] = {}
    for r in results:
        s = r.summary(slo_ms)
        key = (s["scenario"], s["strategy"])
        summaries_by_key.setdefault(key, []).append(s)

    agg: list[dict] = []
    for (scenario, strategy), summaries in summaries_by_key.items():
        n_seeds = len(summaries)
        costs = np.array([s["total_cost"] for s in summaries])
        p50s = np.array([s["p50_ttft_ms"] for s in summaries])
        p99s = np.array([s["p99_ttft_ms"] for s in summaries])
        viols = np.array([s["slo_violation_rate"] for s in summaries])
        hedges = np.array([s["hedge_rate"] for s in summaries])

        def ci95(x):
            mean = float(x.mean())
            sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
            se = sd / max(1, np.sqrt(len(x)))
            return {"mean": mean, "se": float(se), "min": float(x.min()), "max": float(x.max())}

        agg.append(
            {
                "scenario": scenario,
                "strategy": strategy,
                "n_seeds": n_seeds,
                "n_requests": summaries[0]["n"],
                "cost": ci95(costs),
                "p50_ttft_ms": ci95(p50s),
                "p99_ttft_ms": ci95(p99s),
                "slo_violation_rate": ci95(viols),
                "hedge_rate": ci95(hedges),
                # Provider distribution: use first seed for readability.
                "provider_distribution_seed0": summaries[0].get("provider_distribution", {}),
            }
        )
    return agg


def _save_outcomes_csv(results: list[SimResult], path: Path) -> None:
    import csv

    rows = []
    for r in results:
        for o in r.outcomes:
            rows.append(
                {
                    "scenario": r.scenario_name,
                    "strategy": r.strategy_name,
                    "request_id": o.request_id,
                    "timestamp": o.timestamp,
                    "primary": o.primary,
                    "backup": o.backup,
                    "actual_provider": o.actual_provider,
                    "hedged": int(o.hedged),
                    "ttft_ms": o.ttft_ms,
                    "primary_ttft_ms": o.primary_ttft_ms,
                    "backup_ttft_ms": o.backup_ttft_ms,
                    "cost_usd": o.cost_usd,
                    "slo_violated": int(o.slo_violated),
                }
            )
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment/results/synthetic"),
    )
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Scenario names to run (default: all)",
    )
    parser.add_argument("--skip-csv", action="store_true", help="Skip writing per-request CSV")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.n_seeds))

    scenarios = all_scenarios(seed=0)
    if args.scenarios:
        scenarios = [s for s in scenarios if s.name in args.scenarios]
        if not scenarios:
            raise SystemExit(f"No matching scenarios: {args.scenarios}")

    all_results: list[SimResult] = []
    for scenario in scenarios:
        print(f"\n=== {scenario.name} ===")
        print(f"  {scenario.description}")
        print(
            f"  Providers: {[(p.name, p.tier) for p in scenario.providers]}"
        )
        print(
            f"  N requests: {len(scenario.requests)}, SLO: {scenario.slo_ms}ms, "
            f"seeds: {len(seeds)}"
        )
        results = run_scenario(scenario, seeds=seeds)
        all_results.extend(results)

        # Per-scenario aggregation
        agg = aggregate(results, slo_ms=scenario.slo_ms)

        # Sort by strategy name for readability
        agg_sorted = sorted(agg, key=lambda x: x["strategy"])
        for a in agg_sorted:
            viol = a["slo_violation_rate"]["mean"] * 100
            cost = a["cost"]["mean"]
            p99 = a["p99_ttft_ms"]["mean"]
            hedge = a["hedge_rate"]["mean"] * 100
            print(
                f"  {a['strategy']:<25s}  cost=${cost:>8.4f}  "
                f"p99={p99:>7.1f}ms  slo_viol={viol:>5.2f}%  "
                f"hedge={hedge:>5.1f}%"
            )

        # Save aggregate and raw outcomes
        scenario_dir = args.output_dir / scenario.name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        with open(scenario_dir / "summary.json", "w") as f:
            json.dump(
                {
                    "scenario": scenario.name,
                    "description": scenario.description,
                    "providers": [
                        {
                            "name": p.name,
                            "tier": p.tier,
                            "price_per_m_output": p.price_per_m_output,
                            "daily_quota": p.daily_quota,
                            "concurrency_limit": p.concurrency_limit,
                            "ttft_mu": p.ttft_mu,
                            "ttft_sigma": p.ttft_sigma,
                        }
                        for p in scenario.providers
                    ],
                    "slo_ms": scenario.slo_ms,
                    "n_requests": len(scenario.requests),
                    "n_seeds": len(seeds),
                    "results": agg_sorted,
                },
                f,
                indent=2,
            )
        if not args.skip_csv:
            _save_outcomes_csv(results, scenario_dir / "outcomes.csv")

    # Global aggregate
    all_agg = aggregate(all_results, slo_ms=2000)  # slo_ms placeholder; per-scenario is authoritative
    with open(args.output_dir / "all_summary.json", "w") as f:
        json.dump(sorted(all_agg, key=lambda x: (x["scenario"], x["strategy"])), f, indent=2)

    print(f"\n[done] Results in {args.output_dir}")


if __name__ == "__main__":
    main()
