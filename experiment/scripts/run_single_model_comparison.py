#!/usr/bin/env python3
"""
Compare Stage 2 results with different single-model configurations.

Converts all requests to a single model to isolate the routing optimization value.
Tests different models with varying API prices and S_C multipliers.
"""

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiment.scripts.run_stage2_experiments import (
    create_dual_config,
    load_experiment_config,
    run_single_experiment,
    setup_logging,
)
from experiment.strategies.stage2_baselines import (
    ConcurrencyOnlyStrategy,
    DailyQuotaOnlyStrategy,
)
from experiment.strategies.stage2_optimal import ILPOptimalStrategy


def load_and_convert_requests(dataset_name: str, target_model: str, limit: int | None = None):
    """Load dataset and convert all requests to target model."""
    cache_path = Path(f"experiment/cache/dataset/{dataset_name}.pkl")

    with open(cache_path, "rb") as f:
        data = pickle.load(f)

    requests = data if isinstance(data, list) else data.get("requests", data)

    if limit:
        requests = requests[:limit]

    # Convert all requests to target model
    converted = []
    for r in requests:
        r_copy = copy.copy(r)
        r_copy.model = target_model
        converted.append(r_copy)

    return converted


def run_single_model_experiment(
    requests: list,
    model_name: str,
    config: dict,
    delta: float,
    daily_quota: int,
    concurrency: int,
    solver: str,
    sc_monthly_fee: float,
) -> dict:
    """Run experiment for a single model configuration."""
    results = {}

    # ILP Optimal
    results["ilp_optimal"] = run_single_experiment(
        requests,
        ILPOptimalStrategy,
        config,
        "ILP-Optimal",
        delta=delta,
        daily_quota=daily_quota,
        concurrency_limit=concurrency,
        solver=solver,
        dataset_name=f"singlemodel_{model_name}",
        use_cache=False,
        latency_slo=0,
    )

    # Daily-Quota-Only
    results["daily_quota_only"] = run_single_experiment(
        requests,
        DailyQuotaOnlyStrategy,
        config,
        "Daily-Quota-Only",
        daily_quota=daily_quota,
    )

    # Concurrency-Only
    results["concurrency_only"] = run_single_experiment(
        requests,
        ConcurrencyOnlyStrategy,
        config,
        "Concurrency-Only",
        concurrency_limit=concurrency,
        delta=delta,
        dataset_name=f"singlemodel_{model_name}",
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare single-model configurations")
    parser.add_argument("--data", default="freeinference", choices=["freeinference", "rednote"])
    parser.add_argument("--limit", type=int, default=None, help="Limit requests")
    parser.add_argument("--delta", type=float, default=1.0, help="Time discretization")
    parser.add_argument("--solver", default="gurobi", choices=["cbc", "gurobi"])
    parser.add_argument("--daily-quota", type=int, default=5000)
    parser.add_argument(
        "--sc-config",
        type=str,
        choices=["local_gpu", "featherless_premium", "featherless_scale"],
        default="local_gpu",
        help="S_C configuration from experiment.yaml",
    )
    args = parser.parse_args()

    setup_logging()

    # Load config
    exp_config = load_experiment_config()
    model_pricing = exp_config.get("model_pricing", {})
    subscriptions = exp_config.get("subscriptions", {})

    # Fixed model, varying multiplier to isolate effect on S_C value
    model_name = "deepseek-r1"
    test_multipliers = [
        (1, "Small model (mult=1, C_eff=C)"),
        (2, "Medium model (mult=2, C_eff=C/2)"),
        (4, "Large model (mult=4, C_eff=C/4)"),
    ]

    all_results = {}

    # Read S_C config from experiment.yaml
    sc_config = subscriptions.get(args.sc_config, {})
    sc_monthly_fee = sc_config.get("monthly_fee", 0.0)
    sc_concurrency = sc_config.get("concurrency_limit", 8)
    sq_monthly_fee = subscriptions.get("chutes", {}).get("monthly_fee", 20.0)

    # If daily_quota=0, don't charge S_Q subscription (no quota tier)
    if args.daily_quota == 0:
        sq_monthly_fee = 0.0

    print(f"S_Q (Chutes): ${sq_monthly_fee}/mo, Q={args.daily_quota}/day")
    print(f"S_C ({args.sc_config}): ${sc_monthly_fee}/mo, C={sc_concurrency}")
    print(f"Fixed model: {model_name} (input=${model_pricing.get(model_name, {}).get('input', 'N/A')}, "
          f"output=${model_pricing.get(model_name, {}).get('output', 'N/A')})")

    # Load and convert requests once (same model for all runs)
    requests = load_and_convert_requests(args.data, model_name, args.limit)
    print(f"Converted {len(requests)} requests to {model_name}")

    for multiplier, desc in test_multipliers:
        eff_c = sc_concurrency // multiplier
        print(f"\n{'='*70}")
        print(f"Testing: multiplier={multiplier} -> C_eff={eff_c}")
        print(f"  {desc}")
        print(f"{'='*70}")

        # Create config
        config = create_dual_config(
            daily_quota=args.daily_quota,
            concurrency_limit=sc_concurrency,
            sq_monthly_fee=sq_monthly_fee,
            sc_monthly_fee=sc_monthly_fee,
            model_pricing=model_pricing,
            subscriptions=subscriptions,
        )

        # Build active_sc_subscription with the test multiplier
        active_sc = copy.deepcopy(sc_config)
        active_sc["supported_models"] = {model_name: {"multiplier": multiplier}}
        config["active_sc_subscription"] = active_sc

        # Run experiment
        results = run_single_model_experiment(
            requests=requests,
            model_name=model_name,
            config=config,
            delta=args.delta,
            daily_quota=args.daily_quota,
            concurrency=sc_concurrency,
            solver=args.solver,
            sc_monthly_fee=sc_monthly_fee,
        )

        key = f"mult{multiplier}"
        all_results[key] = {
            "model": model_name,
            "description": desc,
            "multiplier": multiplier,
            "effective_concurrency": eff_c,
            "results": results,
        }

        # Print summary
        ilp = results["ilp_optimal"]["costs"]["total"]
        daily = results["daily_quota_only"]["costs"]["total"]
        conc = results["concurrency_only"]["costs"]["total"]
        print(f"\nResults:")
        print(f"  ILP-Optimal:    ${ilp:.2f}")
        print(f"  Daily-Only:     ${daily:.2f}")
        print(f"  Concurrency-Only: ${conc:.2f}")

        # Calculate savings
        best = min(ilp, daily, conc)
        if ilp == best:
            print(f"  -> ILP is best!")
        if daily < ilp:
            print(f"  -> ILP saves ${daily - ilp:.2f} vs Daily-Only ({(daily-ilp)/daily*100:.1f}%)")

    # Final comparison
    print("\n" + "="*90)
    print(f"SINGLE-MODEL COMPARISON SUMMARY")
    print(f"Model: {model_name}, S_Q=${sq_monthly_fee}/mo, S_C=${sc_monthly_fee}/mo, C={sc_concurrency}")
    print("="*90)
    print(f"{'Multiplier':>10} {'Eff_C':>5} {'ILP':>10} {'Daily':>10} {'Conc':>10} {'Best':>10}")
    print("-"*90)

    for key, data in all_results.items():
        mult = data["multiplier"]
        eff_c = data["effective_concurrency"]
        ilp = data["results"]["ilp_optimal"]["costs"]["total"]
        daily = data["results"]["daily_quota_only"]["costs"]["total"]
        conc = data["results"]["concurrency_only"]["costs"]["total"]
        best = min(ilp, daily, conc)
        best_name = "ILP" if ilp == best else ("Daily" if daily == best else "Conc")

        print(f"{mult:>10} {eff_c:>5} ${ilp:>9.2f} ${daily:>9.2f} ${conc:>9.2f} {best_name:>10}")

    print("="*90)

    # Save results
    output_path = Path(f"experiment/results/stage2/single_model_comparison_{args.data}.json")

    # Clean for JSON serialization
    def clean_result(obj):
        if isinstance(obj, dict):
            return {k: clean_result(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_result(v) for v in obj]
        elif callable(obj):
            return str(obj)
        else:
            return obj

    with open(output_path, "w") as f:
        json.dump(clean_result(all_results), f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
