#!/usr/bin/env python3
"""
Compare Stage 2 experiment results with different S_C monthly fees.

Tests S_C at $0 (local GPU), $25 (Featherless Premium), and $75 (Featherless Scale).
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def update_config(config_path: Path, sc_monthly_fee: float, sc_concurrency: int) -> None:
    """Update the S_C monthly fee and concurrency in config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["subscriptions"]["featherless"]["monthly_fee"] = sc_monthly_fee
    config["subscriptions"]["featherless"]["concurrency_limit"] = sc_concurrency

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def run_experiment(
    data: str,
    delta: float,
    solver: str,
    limit: int | None,
    latency_slo: int,
    concurrency: int,
) -> dict:
    """Run the stage 2 experiment and return results."""
    cmd = [
        sys.executable, "-m", "experiment.scripts.run_stage2_experiments",
        "--data", data,
        "--delta", str(delta),
        "--solver", solver,
        "--latency-slo", str(latency_slo),
        "--concurrency", str(concurrency),
        "--no-cache",  # Force recalculation with new S_C config
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse the summary from output
    output = result.stdout + result.stderr

    # Read results from file
    results_path = Path(f"experiment/results/stage2/stage2_results_{data}.json")
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Compare S_C monthly fee impact")
    parser.add_argument("--data", default="freeinference", choices=["freeinference", "rednote"])
    parser.add_argument("--delta", type=float, default=1.0, help="Time discretization (default: 1s)")
    parser.add_argument("--solver", default="gurobi", choices=["cbc", "gurobi"])
    parser.add_argument("--limit", type=int, default=None, help="Limit requests for faster testing")
    parser.add_argument("--latency-slo", type=int, default=0)
    args = parser.parse_args()

    config_path = Path("config/experiment.yaml")

    # Backup original config
    with open(config_path, "r") as f:
        original_config = f.read()

    # Test configurations: (monthly_fee, concurrency_limit, label)
    test_configs = [
        (0.0, 8, "Local GPU ($0/mo, C=8)"),
        (25.0, 4, "Featherless Premium ($25/mo, C=4)"),
        (75.0, 8, "Featherless Scale ($75/mo, C=8)"),
    ]

    all_results = {}

    try:
        for sc_fee, sc_concurrency, label in test_configs:
            print(f"\n{'='*60}")
            print(f"Testing: {label}")
            print(f"{'='*60}")

            # Update config
            update_config(config_path, sc_fee, sc_concurrency)

            # Clear cache for this config (different S_C params = different cache)
            # The cache key includes pricing hash, so we don't need to clear manually

            # Run experiment
            results = run_experiment(
                data=args.data,
                delta=args.delta,
                solver=args.solver,
                limit=args.limit,
                latency_slo=args.latency_slo,
                concurrency=sc_concurrency,
            )

            all_results[label] = results

            # Print summary
            if results:
                print(f"\nResults for {label}:")
                for strategy, result_data in results.items():
                    cost = result_data.get("costs", {}).get("total", "N/A")
                    print(f"  {strategy}: ${cost:.2f}" if isinstance(cost, (int, float)) else f"  {strategy}: {cost}")

    finally:
        # Restore original config
        with open(config_path, "w") as f:
            f.write(original_config)
        print("\n[Config restored to original]")

    # Final comparison table
    print("\n" + "="*70)
    print("COMPARISON SUMMARY (Offline Strategies)")
    print("="*70)
    print(f"{'S_C Configuration':<35} {'ILP-Optimal':>12} {'Daily-Only':>12} {'Conc-Only':>12}")
    print("-"*70)

    for label, results in all_results.items():
        ilp = results.get("ilp_optimal", {}).get("costs", {}).get("total", "-")
        daily = results.get("daily_quota_only", {}).get("costs", {}).get("total", "-")
        conc = results.get("concurrency_only", {}).get("costs", {}).get("total", "-")

        ilp_str = f"${ilp:.2f}" if isinstance(ilp, (int, float)) else str(ilp)
        daily_str = f"${daily:.2f}" if isinstance(daily, (int, float)) else str(daily)
        conc_str = f"${conc:.2f}" if isinstance(conc, (int, float)) else str(conc)

        print(f"{label:<35} {ilp_str:>12} {daily_str:>12} {conc_str:>12}")

    print("="*70)

    # Save comparison results
    output_path = Path(f"experiment/results/stage2/sc_fee_comparison_{args.data}.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
