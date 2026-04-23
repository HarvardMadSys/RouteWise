#!/usr/bin/env python3
"""Run Stage 1 experiments for ICML submission.

This script runs all Stage 1 experiments:
- Fig 1: Cost comparison (All-API, Greedy, Optimal)
- Fig 2: Parameter sensitivity (Q sweep)
- Fig 3: Competitive ratio analysis

Usage:
    python experiment/scripts/run_stage1_experiments.py --data sharegpt
    python experiment/scripts/run_stage1_experiments.py --data community
    python experiment/scripts/run_stage1_experiments.py --data both
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
from experiment.quota import QuotaManager
from experiment.simulator import OfflineSimulator
from experiment.strategies.all_api import AllAPIStrategy
from experiment.strategies.greedy import GreedyStrategy
from experiment.strategies.stage1_optimal import OptimalStrategy

# Default config path
DEFAULT_CONFIG_PATH = "config/experiment.yaml"


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(
    config_path: str = DEFAULT_CONFIG_PATH,
    daily_quota: int | None = None,
    multimodel: bool = False,
    target_model: str | None = None,
) -> dict:
    """Load experiment configuration from YAML file.

    Args:
        config_path: Path to experiment.yaml
        daily_quota: Override daily quota (optional)
        multimodel: Enable multi-model pricing from config
        target_model: Target model name for pricing (maps "default" to this model)

    Returns:
        Configuration dictionary
    """
    # Load from YAML
    exp_config = ExperimentConfig(config_path)
    config = exp_config.to_dict()

    # Override daily quota if specified
    if daily_quota is not None:
        sub_provider = config["simulation"].get("default_subscription", "chutes-subscription")
        if sub_provider in config["providers"]:
            # Create new ProviderConfig with updated quota
            old_provider = config["providers"][sub_provider]
            config["providers"][sub_provider] = type(old_provider)(
                name=old_provider.name,
                type=old_provider.type,
                monthly_fee=old_provider.monthly_fee,
                daily_quota=daily_quota,
            )

    # Handle model pricing
    if target_model:
        # Use specified model's pricing, map "default" to it
        model_pricing = config.get("model_pricing", {})
        if target_model in model_pricing:
            # Map "default" to the target model's pricing
            model_pricing["default"] = model_pricing[target_model]
            config["model_pricing"] = model_pricing
        else:
            raise ValueError(
                f"Model '{target_model}' not found in model_pricing. "
                f"Available models: {list(model_pricing.keys())}"
            )
    elif not multimodel:
        # Remove model_pricing if not using multi-model
        config.pop("model_pricing", None)

    return config


def run_single_experiment(
    requests: list,
    strategy_class,
    config: dict,
    strategy_name: str,
) -> dict:
    """Run a single experiment with given strategy.

    Args:
        requests: List of requests
        strategy_class: Strategy class to use
        config: Configuration dictionary
        strategy_name: Name for logging

    Returns:
        Result dictionary
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running {strategy_name}...")

    # Create components
    calculator = CostCalculator(config["providers"], config.get("model_pricing"))
    sub_provider = config["simulation"].get("default_subscription", "chutes-subscription")
    quota_manager = QuotaManager(
        daily_quota=config["providers"][sub_provider].daily_quota,
        num_subscriptions=config["simulation"]["num_subscriptions"],
    )

    # Create strategy
    strategy = strategy_class(calculator, quota_manager, config)

    # Run simulation
    simulator = OfflineSimulator(requests, strategy, config)
    result = simulator.run()

    return result.to_dict()


def run_fig1_experiment(requests: list, config: dict) -> dict:
    """Run Fig 1 experiment: Cost comparison.

    Args:
        requests: List of requests
        config: Configuration dictionary

    Returns:
        Results for all three strategies
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Fig 1: Cost Comparison (All-API, Greedy, Optimal)")
    logger.info("=" * 60)

    # Check if multi-model pricing is enabled
    if config.get("model_pricing"):
        logger.info("Using multi-model pricing")

    results = {}

    # All-API baseline
    results["all_api"] = run_single_experiment(requests, AllAPIStrategy, config, "All-API")

    # Greedy online
    results["greedy"] = run_single_experiment(requests, GreedyStrategy, config, "Greedy")

    # Optimal offline
    results["optimal"] = run_single_experiment(requests, OptimalStrategy, config, "Optimal")

    # Calculate savings
    all_api_cost = results["all_api"]["costs"]["total"]
    for _name, result in results.items():
        total_cost = result["costs"]["total"]
        savings = (all_api_cost - total_cost) / all_api_cost * 100 if all_api_cost > 0 else 0
        result["savings_vs_all_api"] = round(savings, 2)

    return results


def run_fig2_experiment(
    requests: list,
    base_config: dict,
    multimodel: bool = False,
    target_model: str | None = None,
) -> dict:
    """Run Fig 2 experiment: Parameter sensitivity (Q sweep).

    Args:
        requests: List of requests
        base_config: Base configuration dictionary
        multimodel: Enable multi-model pricing
        target_model: Target model for pricing

    Returns:
        Results for different Q values
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Fig 2: Parameter Sensitivity (Q sweep)")
    logger.info("=" * 60)

    # Q values to test
    q_values = [1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000]

    results = {"q_values": q_values, "all_api": [], "greedy": [], "optimal": []}

    for q in q_values:
        logger.info(f"\n--- Q = {q} ---")

        # Update config with new Q
        config = load_config(daily_quota=q, multimodel=multimodel, target_model=target_model)

        # Run all three strategies
        all_api_result = run_single_experiment(requests, AllAPIStrategy, config, f"All-API (Q={q})")
        greedy_result = run_single_experiment(requests, GreedyStrategy, config, f"Greedy (Q={q})")
        optimal_result = run_single_experiment(
            requests, OptimalStrategy, config, f"Optimal (Q={q})"
        )

        results["all_api"].append(all_api_result["costs"]["total"])
        results["greedy"].append(greedy_result["costs"]["total"])
        results["optimal"].append(optimal_result["costs"]["total"])

    return results


def run_fig3_experiment(fig2_results: dict) -> dict:
    """Run Fig 3 experiment: Competitive ratio analysis.

    Args:
        fig2_results: Results from Fig 2 experiment

    Returns:
        Competitive ratio data
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Fig 3: Competitive Ratio Analysis")
    logger.info("=" * 60)

    q_values = fig2_results["q_values"]
    greedy_costs = fig2_results["greedy"]
    optimal_costs = fig2_results["optimal"]

    competitive_ratios = []
    for i, q in enumerate(q_values):
        ratio = greedy_costs[i] / optimal_costs[i] if optimal_costs[i] > 0 else 1.0
        competitive_ratios.append(round(ratio, 4))
        logger.info(f"Q={q}: Greedy/Optimal = {ratio:.4f}")

    return {
        "q_values": q_values,
        "competitive_ratios": competitive_ratios,
        "avg_ratio": round(sum(competitive_ratios) / len(competitive_ratios), 4),
        "max_ratio": round(max(competitive_ratios), 4),
        "min_ratio": round(min(competitive_ratios), 4),
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Stage 1 experiments")
    parser.add_argument(
        "--data",
        type=str,
        choices=["sharegpt", "freeinference", "rednote", "all"],
        default="sharegpt",
        help="Dataset to use: sharegpt (single-model), freeinference/rednote (multi-model), all",
    )
    parser.add_argument(
        "--multimodel",
        action="store_true",
        help="Enable multi-model pricing (uses each request's model field)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Target model for pricing (e.g., 'llama-3.3-70b-instruct', 'deepseek-chat'). "
        "Use this for single-model datasets like sharegpt.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiment/results/stage1",
        help="Output directory for results",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Logging level",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define datasets
    datasets = {}
    if args.data in ["sharegpt", "all"]:
        datasets["sharegpt"] = "data/sharegpt_burstgpt/converted.csv"
    if args.data in ["freeinference", "all"]:
        datasets["freeinference"] = "data/freeinference_logs.csv"
    if args.data in ["rednote", "all"]:
        datasets["rednote"] = "data/rednote_logs.csv"

    # Run experiments for each dataset
    for dataset_name, dataset_path in datasets.items():
        logger.info(f"\n{'#' * 60}")
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"{'#' * 60}")

        # Determine pricing mode
        use_multimodel = args.multimodel or dataset_name in ["freeinference", "rednote"]
        target_model = args.model

        if target_model:
            logger.info(f"Using model pricing: {target_model}")
        elif use_multimodel:
            logger.info("Multi-model pricing ENABLED (per-request model)")

        # Load data
        config = load_config(multimodel=use_multimodel, target_model=target_model)
        loader = DataLoader(config)

        try:
            requests = loader.load(dataset_path)
        except Exception as e:
            logger.error(f"Failed to load {dataset_path}: {e}")
            continue

        logger.info(f"Loaded {len(requests):,} requests")

        # Run experiments
        fig1_results = run_fig1_experiment(requests, config)
        fig2_results = run_fig2_experiment(requests, config, use_multimodel, target_model)
        fig3_results = run_fig3_experiment(fig2_results)

        # Combine results (convert any non-serializable objects)
        def clean_result(obj):
            if isinstance(obj, dict):
                return {k: clean_result(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_result(v) for v in obj]
            elif callable(obj):
                return str(obj)
            else:
                return obj

        all_results = clean_result(
            {
                "dataset": dataset_name,
                "num_requests": len(requests),
                "pricing_model": target_model or ("multimodel" if use_multimodel else "default"),
                "fig1_cost_comparison": fig1_results,
                "fig2_parameter_sensitivity": fig2_results,
                "fig3_competitive_ratio": fig3_results,
            }
        )

        # Save results (include model name in filename if specified)
        if target_model:
            output_file = output_dir / f"{dataset_name}_{target_model}_results.json"
        else:
            output_file = output_dir / f"{dataset_name}_results.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"\nResults saved to {output_file}")

        # Print summary
        print("\n" + "=" * 60)
        print(f"SUMMARY: {dataset_name}")
        print("=" * 60)
        print("\nFig 1 - Cost Comparison:")
        for strategy, result in fig1_results.items():
            print(
                f"  {strategy}: ${result['costs']['total']:.2f} "
                f"(savings: {result.get('savings_vs_all_api', 0):.1f}%)"
            )

        print("\nFig 3 - Competitive Ratio:")
        print(f"  Avg: {fig3_results['avg_ratio']:.4f}")
        print(f"  Max: {fig3_results['max_ratio']:.4f}")
        print(f"  Min: {fig3_results['min_ratio']:.4f}")


if __name__ == "__main__":
    main()
