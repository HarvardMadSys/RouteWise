#!/usr/bin/env python3
"""Oracle experiment: Output length prediction sensitivity analysis.

This script measures the impact of prediction error on end-to-end routing cost
by comparing strategies using predicted vs true output lengths.

Gap decomposition:
    Total Gap = Cost(PD-EMA) - Cost(Optimal)
              = [Cost(PD-EMA) - Cost(PD-Oracle)]  <-- prediction error impact
              + [Cost(PD-Oracle) - Cost(Optimal)]  <-- online decision cost

If the prediction gap is small, simple heuristics (EMA/Histogram) are sufficient.
If the prediction gap is large, better prediction methods are needed.

Usage:
    python experiment/scripts/run_oracle_experiment.py
    python experiment/scripts/run_oracle_experiment.py --data burstgpt
    python experiment/scripts/run_oracle_experiment.py --data freeinference
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiment.config import ExperimentConfig
from experiment.cost import CostCalculator
from experiment.data.loader import DataLoader
from experiment.predictors import (
    EMAOutputPredictor,
    HistogramOutputPredictor,
    OracleOutputPredictor,
)
from experiment.quota import QuotaManager
from experiment.simulator import OfflineSimulator
from experiment.strategies.all_api import AllAPIStrategy
from experiment.strategies.online import (
    GreedyOnlineStrategy,
    LAPDConfig,
    LearningAugmentedPrimalDualStrategy,
    PrimalDualOnlineStrategy,
)
from experiment.strategies.stage1_optimal import OptimalStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Dataset configurations
DATASETS = {
    "burstgpt": {
        "path": "data/BurstGPT_1.csv",
        "model_override": "deepseek-r1",
        "multimodel": False,
        "description": "BurstGPT (single-model, 1.4M requests)",
    },
    "freeinference": {
        "path": "data/freeinference_logs.csv",
        "model_override": None,
        "multimodel": True,
        "description": "FreeInference (multi-model, 371K requests)",
    },
}

# Quota plans to test (matching existing experiments)
QUOTA_PLANS = {
    "Base": {"monthly_fee": 3.0, "daily_quota": 300},
    "Plus": {"monthly_fee": 10.0, "daily_quota": 2000},
    "Pro": {"monthly_fee": 20.0, "daily_quota": 5000},
}


def run_strategy(
    requests: list,
    strategy,
    config_dict: dict,
    name: str,
) -> dict:
    """Run a single strategy and return results.

    Args:
        requests: List of requests
        strategy: Strategy instance
        config_dict: Configuration dictionary
        name: Strategy name for logging

    Returns:
        Result dictionary
    """
    logger.info(f"  Running {name}...")
    simulator = OfflineSimulator(requests, strategy, config_dict)
    result = simulator.run()
    return result.to_dict()


def run_oracle_experiment_for_quota(
    requests: list,
    config: ExperimentConfig,
    daily_quota: int,
    model_override: str | None = None,
) -> dict:
    """Run oracle experiment for a single quota level.

    Args:
        requests: List of requests
        config: Experiment configuration
        daily_quota: Daily quota Q
        model_override: Model name for pricing override

    Returns:
        Results dictionary with all strategies
    """
    config_dict = config.to_dict()

    # Calculate L/U bounds
    cost_calculator = CostCalculator(config_dict["providers"], config_dict.get("model_pricing"))
    costs = []
    for r in requests[:10000]:
        try:
            c = cost_calculator.calculate_cost_by_model(r)
        except Exception:
            try:
                _, c = cost_calculator.get_cheapest_api_provider(r)
            except Exception:
                c = 0.0
        costs.append(c)
    costs.sort()
    L = costs[int(len(costs) * 0.05)] if costs else 0.0001
    U = costs[int(len(costs) * 0.95)] if costs else 0.01
    logger.info(f"  Cost bounds: L={L:.6f}, U={U:.6f}")

    results = {}

    # --- Baselines ---

    # All-API
    quota_mgr = QuotaManager(daily_quota)
    strategy = AllAPIStrategy(cost_calculator, quota_mgr, config_dict)
    results["All-API"] = run_strategy(requests, strategy, config_dict, "All-API")

    # Greedy
    quota_mgr = QuotaManager(daily_quota)
    strategy = GreedyOnlineStrategy(
        cost_calculator,
        quota_mgr,
        config_dict,
        daily_quota=daily_quota,
        concurrency_limit=0,
    )
    results["Greedy"] = run_strategy(requests, strategy, config_dict, "Greedy")

    # Offline Optimal
    quota_mgr = QuotaManager(daily_quota)
    strategy = OptimalStrategy(cost_calculator, quota_mgr, config_dict)
    results["Optimal"] = run_strategy(requests, strategy, config_dict, "Optimal")

    # --- PD variants (threshold on median value) ---

    # PD-Oracle: PD + true output length
    quota_mgr = QuotaManager(daily_quota)
    strategy = PrimalDualOnlineStrategy(
        cost_calculator,
        quota_mgr,
        config_dict,
        daily_quota=daily_quota,
        sq_min_value=L,
        sq_max_value=U,
        concurrency_limit=0,
        output_predictor=OracleOutputPredictor(),
    )
    results["PD-Oracle"] = run_strategy(requests, strategy, config_dict, "PD-Oracle")

    # PD-EMA: PD + EMA predictor (current default)
    quota_mgr = QuotaManager(daily_quota)
    strategy = PrimalDualOnlineStrategy(
        cost_calculator,
        quota_mgr,
        config_dict,
        daily_quota=daily_quota,
        sq_min_value=L,
        sq_max_value=U,
        concurrency_limit=0,
    )
    results["PD-EMA"] = run_strategy(requests, strategy, config_dict, "PD-EMA")

    # PD-Histogram: PD + Histogram predictor
    quota_mgr = QuotaManager(daily_quota)
    strategy = PrimalDualOnlineStrategy(
        cost_calculator,
        quota_mgr,
        config_dict,
        daily_quota=daily_quota,
        sq_min_value=L,
        sq_max_value=U,
        concurrency_limit=0,
        output_predictor=HistogramOutputPredictor(),
    )
    results["PD-Hist"] = run_strategy(requests, strategy, config_dict, "PD-Histogram")

    # --- LA variants (threshold on LCB / P10) ---

    # LA-Oracle: LA + true output length
    quota_mgr = QuotaManager(daily_quota)
    lapd_config = LAPDConfig(quantile_for_lcb=0.10)
    strategy = LearningAugmentedPrimalDualStrategy(
        cost_calculator,
        quota_mgr,
        config_dict,
        daily_quota=daily_quota,
        sq_min_value=L,
        sq_max_value=U,
        lapd_config=lapd_config,
        output_predictor=OracleOutputPredictor(),
    )
    results["LA-Oracle"] = run_strategy(requests, strategy, config_dict, "LA-Oracle")

    # LA-EMA: LA + EMA predictor
    quota_mgr = QuotaManager(daily_quota)
    lapd_config = LAPDConfig(quantile_for_lcb=0.10)
    strategy = LearningAugmentedPrimalDualStrategy(
        cost_calculator,
        quota_mgr,
        config_dict,
        daily_quota=daily_quota,
        sq_min_value=L,
        sq_max_value=U,
        lapd_config=lapd_config,
        output_predictor=EMAOutputPredictor(),
    )
    results["LA-EMA"] = run_strategy(requests, strategy, config_dict, "LA-EMA")

    # LA-Histogram: LA + Histogram predictor (default for LA)
    quota_mgr = QuotaManager(daily_quota)
    lapd_config = LAPDConfig(quantile_for_lcb=0.10)
    strategy = LearningAugmentedPrimalDualStrategy(
        cost_calculator,
        quota_mgr,
        config_dict,
        daily_quota=daily_quota,
        sq_min_value=L,
        sq_max_value=U,
        lapd_config=lapd_config,
    )
    results["LA-Hist"] = run_strategy(requests, strategy, config_dict, "LA-Histogram")

    return results


def analyze_results(results: dict, plan_name: str) -> dict:
    """Analyze results and compute gap decomposition.

    Args:
        results: Dictionary of strategy -> result
        plan_name: Name of the quota plan

    Returns:
        Analysis dictionary
    """
    optimal_cost = results["Optimal"]["costs"]["total"]
    all_api_cost = results["All-API"]["costs"]["total"]

    analysis = {"plan": plan_name, "strategies": {}}

    for name, res in results.items():
        total_cost = res["costs"]["total"]
        cr = total_cost / optimal_cost if optimal_cost > 0 else float("inf")
        savings_pct = (all_api_cost - total_cost) / all_api_cost * 100 if all_api_cost > 0 else 0
        analysis["strategies"][name] = {
            "cost": round(total_cost, 4),
            "cr": round(cr, 4),
            "savings_pct": round(savings_pct, 2),
        }

    # Gap decomposition for PD
    pd_oracle_cr = analysis["strategies"]["PD-Oracle"]["cr"]
    pd_ema_cr = analysis["strategies"]["PD-EMA"]["cr"]
    pd_hist_cr = analysis["strategies"]["PD-Hist"]["cr"]
    optimal_cr = 1.0

    analysis["pd_gap_decomposition"] = {
        "total_gap": round(pd_ema_cr - optimal_cr, 4),
        "online_gap": round(pd_oracle_cr - optimal_cr, 4),
        "prediction_gap_ema": round(pd_ema_cr - pd_oracle_cr, 4),
        "prediction_gap_hist": round(pd_hist_cr - pd_oracle_cr, 4),
        "prediction_pct_of_total_ema": round(
            (pd_ema_cr - pd_oracle_cr) / (pd_ema_cr - optimal_cr) * 100, 1
        )
        if pd_ema_cr > optimal_cr
        else 0.0,
    }

    # Gap decomposition for LA
    la_oracle_cr = analysis["strategies"]["LA-Oracle"]["cr"]
    la_ema_cr = analysis["strategies"]["LA-EMA"]["cr"]
    la_hist_cr = analysis["strategies"]["LA-Hist"]["cr"]

    analysis["la_gap_decomposition"] = {
        "total_gap": round(la_ema_cr - optimal_cr, 4),
        "online_gap": round(la_oracle_cr - optimal_cr, 4),
        "prediction_gap_ema": round(la_ema_cr - la_oracle_cr, 4),
        "prediction_gap_hist": round(la_hist_cr - la_oracle_cr, 4),
        "prediction_pct_of_total_ema": round(
            (la_ema_cr - la_oracle_cr) / (la_ema_cr - optimal_cr) * 100, 1
        )
        if la_ema_cr > optimal_cr
        else 0.0,
    }

    return analysis


def print_results_table(analysis: dict, dataset_name: str) -> None:
    """Print formatted results table.

    Args:
        analysis: Analysis dictionary from analyze_results
        dataset_name: Name of the dataset
    """
    plan = analysis["plan"]
    strategies = analysis["strategies"]
    pd_gap = analysis["pd_gap_decomposition"]
    la_gap = analysis["la_gap_decomposition"]

    print(f"\n{'=' * 72}")
    print(f"  {dataset_name} | Plan: {plan}")
    print(f"{'=' * 72}")
    print(f"  {'Strategy':<16} {'Cost':>10} {'CR':>8} {'Savings':>10}")
    print(f"  {'-' * 60}")

    for name in [
        "Optimal",
        "PD-Oracle",
        "PD-EMA",
        "PD-Hist",
        "LA-Oracle",
        "LA-EMA",
        "LA-Hist",
        "Greedy",
        "All-API",
    ]:
        s = strategies[name]
        print(f"  {name:<16} ${s['cost']:>9.2f} {s['cr']:>7.4f}x {s['savings_pct']:>8.1f}%")

    print("\n  PD Gap Decomposition:")
    print(f"    Total gap (PD-EMA - Optimal):     {pd_gap['total_gap']:.4f}")
    print(f"    Online gap (PD-Oracle - Optimal):  {pd_gap['online_gap']:.4f}")
    print(
        f"    Prediction gap (PD-EMA - Oracle):  {pd_gap['prediction_gap_ema']:.4f}"
        f"  ({pd_gap['prediction_pct_of_total_ema']:.1f}% of total)"
    )
    print(f"    Prediction gap (PD-Hist - Oracle): {pd_gap['prediction_gap_hist']:.4f}")

    print("\n  LA Gap Decomposition:")
    print(f"    Total gap (LA-EMA - Optimal):      {la_gap['total_gap']:.4f}")
    print(f"    Online gap (LA-Oracle - Optimal):   {la_gap['online_gap']:.4f}")
    print(
        f"    Prediction gap (LA-EMA - Oracle):   {la_gap['prediction_gap_ema']:.4f}"
        f"  ({la_gap['prediction_pct_of_total_ema']:.1f}% of total)"
    )
    print(f"    Prediction gap (LA-Hist - Oracle):  {la_gap['prediction_gap_hist']:.4f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Oracle experiment: prediction sensitivity analysis"
    )
    parser.add_argument(
        "--data",
        type=str,
        choices=[*list(DATASETS.keys()), "all"],
        default="all",
        help="Dataset to use",
    )
    parser.add_argument(
        "--plan",
        type=str,
        choices=[*list(QUOTA_PLANS.keys()), "all"],
        default="all",
        help="Quota plan to test",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment.yaml",
        help="Path to experiment config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiment/results/oracle",
        help="Output directory for results",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select datasets
    dataset_names = list(DATASETS.keys()) if args.data == "all" else [args.data]

    # Select plans
    plan_names = list(QUOTA_PLANS.keys()) if args.plan == "all" else [args.plan]

    all_results = {}

    for dataset_name in dataset_names:
        ds = DATASETS[dataset_name]
        logger.info(f"\n{'#' * 60}")
        logger.info(f"Dataset: {ds['description']}")
        logger.info(f"{'#' * 60}")

        # Check dataset exists
        if not Path(ds["path"]).exists():
            logger.warning(f"Dataset not found: {ds['path']}, skipping")
            continue

        # Load config and data
        config = ExperimentConfig(args.config)
        config_dict = config.to_dict()

        # Handle model pricing
        if not ds["multimodel"]:
            config_dict.pop("model_pricing", None)
        if ds["model_override"]:
            model_pricing = config_dict.get("model_pricing", {})
            if ds["model_override"] in model_pricing:
                model_pricing["default"] = model_pricing[ds["model_override"]]
                config_dict["model_pricing"] = model_pricing

        loader = DataLoader(config_dict)
        requests = loader.load(ds["path"], model_override=ds["model_override"])
        stats = loader.get_statistics(requests)
        logger.info(f"Loaded {stats['total_requests']} requests over {stats['num_days']} days")

        dataset_results = {}

        for plan_name in plan_names:
            plan = QUOTA_PLANS[plan_name]
            daily_quota = plan["daily_quota"]

            logger.info(f"\n--- Plan: {plan_name} (Q={daily_quota}) ---")

            # Update subscription provider quota
            sub_provider = config.get_subscription_provider()
            original_quota = sub_provider.daily_quota
            sub_provider_key = None
            for key, p in config_dict["providers"].items():
                if hasattr(p, "is_subscription") and p.is_subscription():
                    sub_provider_key = key
                    break

            if sub_provider_key:
                old_p = config_dict["providers"][sub_provider_key]
                config_dict["providers"][sub_provider_key] = type(old_p)(
                    name=old_p.name,
                    type=old_p.type,
                    monthly_fee=plan["monthly_fee"],
                    daily_quota=daily_quota,
                )

            results = run_oracle_experiment_for_quota(
                requests, config, daily_quota, ds["model_override"]
            )
            analysis = analyze_results(results, plan_name)
            print_results_table(analysis, ds["description"])

            dataset_results[plan_name] = {
                "config": plan,
                "raw_results": {
                    name: {
                        "cost": res["costs"]["total"],
                        "api_cost": res["costs"]["api"],
                        "subscription_cost": res["costs"]["subscription"],
                        "quota_utilization": res["quota_utilization"],
                    }
                    for name, res in results.items()
                },
                "analysis": analysis,
            }

            # Restore original quota
            if sub_provider_key:
                old_p = config_dict["providers"][sub_provider_key]
                config_dict["providers"][sub_provider_key] = type(old_p)(
                    name=old_p.name,
                    type=old_p.type,
                    monthly_fee=old_p.monthly_fee,
                    daily_quota=original_quota,
                )

        all_results[dataset_name] = {
            "description": ds["description"],
            "num_requests": stats["total_requests"],
            "num_days": stats["num_days"],
            "plans": dataset_results,
        }

    # Save all results
    output_file = output_dir / "oracle_experiment_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {output_file}")

    # Print final summary
    print("\n" + "=" * 72)
    print("  SUMMARY: Prediction Gap Analysis")
    print("=" * 72)
    for _dataset_name, ds_results in all_results.items():
        print(f"\n  {ds_results['description']}:")
        for plan_name, plan_results in ds_results["plans"].items():
            pd_gap = plan_results["analysis"]["pd_gap_decomposition"]
            print(
                f"    {plan_name:>5s} (Q={QUOTA_PLANS[plan_name]['daily_quota']:>5d}): "
                f"Total gap={pd_gap['total_gap']:.4f}, "
                f"Online={pd_gap['online_gap']:.4f}, "
                f"Prediction={pd_gap['prediction_gap_ema']:.4f} "
                f"({pd_gap['prediction_pct_of_total_ema']:.1f}%)"
            )


if __name__ == "__main__":
    main()
