#!/usr/bin/env python3
"""Run Stage 2 (Dual Subscription) experiments.

This script runs the "Phase 2" experiments involving two types of subscriptions:
1. Daily Quota Subscription (S_Q): Fixed daily quota (e.g., Chutes)
2. Concurrency Subscription (S_C): Fixed concurrency limit (e.g., Featherless)

All strategies are OFFLINE (have perfect future knowledge):
- ILP Optimal: Jointly optimizes assignment and scheduling
- Daily-Quota-Only (B2): Offline optimal for S_Q only
- Concurrency-Only (B3): Offline optimal for S_C only
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiment.cache import (
    load_cached_dataset,
    save_dataset_cache,
)
from experiment.config import ExperimentConfig
from experiment.cost import CostCalculator
from experiment.data.loader import DataLoader
from experiment.data.schema import ProviderConfig, ProviderType
from experiment.quota import QuotaManager
from experiment.simulator import OfflineSimulator, SimulationResult
from experiment.strategies.stage2_baselines import (
    ConcurrencyOnlyStrategy,
    DailyQuotaOnlyStrategy,
)
from experiment.strategies.stage2_optimal import ILPOptimalStrategy

DEFAULT_CONFIG_PATH = "config/experiment.yaml"


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def load_experiment_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load full config from experiment.yaml.

    Args:
        config_path: Path to experiment.yaml

    Returns:
        Full configuration dictionary including model_pricing and subscriptions
    """
    exp_config = ExperimentConfig(config_path)
    return exp_config.to_dict()


def create_dual_config(
    daily_quota: int = 5000,
    concurrency_limit: int = 4,
    sq_monthly_fee: float = 20.0,  # S_Q fee (Chutes $20/mo)
    sc_monthly_fee: float = 0.0,  # S_C fee (Local GPU, sunk cost)
    model_pricing: dict | None = None,
    subscriptions: dict | None = None,
) -> dict:
    """Create experiment configuration for dual subscription.

    Args:
        daily_quota: Daily quota for S_Q
        concurrency_limit: Concurrency limit for S_C (total slots)
        sq_monthly_fee: Monthly fee for S_Q
        sc_monthly_fee: Monthly fee for S_C
        model_pricing: Per-model API pricing from experiment.yaml
        subscriptions: Subscription configs with model compatibility

    Returns:
        Configuration dictionary
    """
    providers = {
        # S_Q: Daily Quota Subscription
        "daily-quota-sub": ProviderConfig(
            name="Daily Quota Subscription",
            type=ProviderType.SUBSCRIPTION,
            monthly_fee=sq_monthly_fee,
            daily_quota=daily_quota,
        ),
        # S_C: Concurrency Subscription
        "concurrency-sub": ProviderConfig(
            name="Concurrency Subscription",
            type=ProviderType.SUBSCRIPTION,
            monthly_fee=sc_monthly_fee,
            daily_quota=0,  # No daily limit, only concurrency
        ),
        # API Fallback
        "openai-chatgpt": ProviderConfig(
            name="OpenAI ChatGPT",
            type=ProviderType.API,
            input_price_per_1k=0.0015,
            output_price_per_1k=0.002,
        ),
    }

    config = {
        "providers": providers,
        "simulation": {
            "num_subscriptions": 1,  # Base count, logic will handle dual types
            "default_subscription": "daily-quota-sub",
            "subscription_provider": "daily-quota-sub",  # Required by OfflineSimulator
            "default_api_fallback": "openai-chatgpt",
            # Custom params for dual sub
            "sq_count": 1,
            "sc_count": 1,
        },
        "dataset": {},
        "output": {},
    }

    # Add model_pricing if provided
    if model_pricing:
        config["model_pricing"] = model_pricing

    # Add subscriptions config for model compatibility
    if subscriptions:
        config["subscriptions"] = subscriptions

    return config


class DualSubscriptionSimulator(OfflineSimulator):
    """Simulator that handles cost calculation for two subscription types."""

    def run(self) -> SimulationResult:
        """Run simulation with dual subscription cost calculation."""
        # Run standard simulation first
        result = super().run()

        # Recalculate subscription costs based on STRATEGY TYPE
        # Not all strategies use both subscriptions!

        num_days = result.num_days
        if num_days == 0 and self.requests:
            num_days = self.requests[-1].day - self.requests[0].day + 1

        # Get provider configs
        sq_provider = self.config["providers"]["daily-quota-sub"]
        sc_provider = self.config["providers"]["concurrency-sub"]
        sq_count = self.config["simulation"].get("sq_count", 1)
        sc_count = self.config["simulation"].get("sc_count", 1)

        # Calculate potential costs
        sq_cost = sq_provider.monthly_fee * sq_count * (num_days / 30.0)
        sc_cost = sc_provider.monthly_fee * sc_count * (num_days / 30.0)

        # Determine which costs apply based on strategy name
        strategy_name = self.strategy.name.lower()

        applied_sq_cost = 0.0
        applied_sc_cost = 0.0

        if "daily-quota-only" in strategy_name:
            # B2: Only uses S_Q
            applied_sq_cost = sq_cost
            applied_sc_cost = 0.0
        elif "concurrency-only" in strategy_name:
            # B3: Only uses S_C
            applied_sq_cost = 0.0
            applied_sc_cost = sc_cost
        else:
            # ILP-Optimal, Greedy-Online use BOTH
            applied_sq_cost = sq_cost
            applied_sc_cost = sc_cost

        total_sub_cost = applied_sq_cost + applied_sc_cost

        # Update result
        result.subscription_cost = total_sub_cost
        result.total_cost = total_sub_cost + result.api_cost

        # Log corrected costs
        logger = logging.getLogger(__name__)
        logger.info(
            f"Corrected Cost for {self.strategy.name}: ${total_sub_cost:.2f} "
            f"(S_Q: ${applied_sq_cost:.2f}, S_C: ${applied_sc_cost:.2f})"
        )
        logger.info(f"Total Cost: ${result.total_cost:.2f}")

        return result


def run_single_experiment(
    requests: list, strategy_class, config: dict, strategy_name: str, **strategy_kwargs
) -> dict:
    """Run a single experiment."""
    logger = logging.getLogger(__name__)
    logger.info(f"Running {strategy_name}...")

    # Create components (with model_pricing support)
    calculator = CostCalculator(config["providers"], config.get("model_pricing"))

    # QuotaManager primarily tracks Daily Quota (S_Q)
    # Concurrency logic is handled inside strategies or via multiple managers if we refactored
    # For now, we pass the S_Q parameters to the standard QuotaManager
    quota_manager = QuotaManager(
        daily_quota=config["providers"]["daily-quota-sub"].daily_quota,
        num_subscriptions=config["simulation"]["sq_count"],
    )

    # Create strategy
    # Pass specific limits if strategy needs them
    strategy = strategy_class(calculator, quota_manager, config, **strategy_kwargs)

    # Run simulation
    simulator = DualSubscriptionSimulator(requests, strategy, config)
    result = simulator.run()

    return result.to_dict()


def main():
    """Run Stage 2 dual subscription experiments."""
    parser = argparse.ArgumentParser(description="Run Stage 2 (Dual Subscription) experiments")
    parser.add_argument(
        "--data",
        type=str,
        choices=["sharegpt", "freeinference", "rednote"],
        default="freeinference",
        help="Dataset to use: sharegpt (single-model), freeinference/rednote (multi-model)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiment/results/stage2",
        help="Output directory",
    )
    parser.add_argument("--daily-quota", type=int, default=5000, help="Daily quota for S_Q")
    parser.add_argument(
        "--sc-config",
        type=str,
        choices=["local_gpu", "featherless_premium", "featherless_scale"],
        default="local_gpu",
        help="S_C configuration: local_gpu ($0, C=8), featherless_premium ($25, C=4), featherless_scale ($75, C=8)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=None, help="Override concurrency limit (default: from --sc-config)"
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of requests")
    parser.add_argument(
        "--delta",
        type=float,
        default=300.0,
        help="Time slot size in seconds for ILP (default: 300s = 5min)",
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=["cbc", "gurobi"],
        default="gurobi",
        help="ILP solver to use (default: gurobi)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable dataset caching (force reload from CSV)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Max parallel workers for ILP solving (default: auto = cpu_count)",
    )
    parser.add_argument(
        "--latency-slo",
        type=int,
        default=0,
        help="Latency SLO in slots: max queueing delay (0=zero-wait/loss system, -1=unlimited)",
    )

    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine if multi-model pricing should be used
    use_multimodel = args.data in ["freeinference", "rednote"]

    # Load config from experiment.yaml
    exp_config = load_experiment_config()
    model_pricing = exp_config.get("model_pricing") if use_multimodel else None
    subscriptions = exp_config.get("subscriptions")  # Model compatibility config

    if use_multimodel:
        logger.info("Multi-model pricing ENABLED (loaded from experiment.yaml)")
    if subscriptions:
        logger.info(f"Model compatibility loaded: {len(subscriptions)} subscription types")

    # Read S_Q and S_C configuration
    sq_monthly_fee = 20.0  # Default Chutes
    sc_monthly_fee = 0.0   # Default Local GPU
    sc_concurrency = 8     # Default concurrency
    if subscriptions:
        # S_Q config (Chutes)
        chutes_config = subscriptions.get("chutes", {})
        sq_monthly_fee = chutes_config.get("monthly_fee", 20.0)

        # S_C config (from --sc-config parameter)
        sc_config = subscriptions.get(args.sc_config, {})
        sc_monthly_fee = sc_config.get("monthly_fee", 0.0)
        sc_concurrency = sc_config.get("concurrency_limit", 8)
        logger.info(f"S_C config: {args.sc_config} (${sc_monthly_fee}/mo, C={sc_concurrency})")

    # Allow command-line override of concurrency
    if args.concurrency is not None:
        sc_concurrency = args.concurrency
    args.concurrency = sc_concurrency

    # Create config
    config = create_dual_config(
        daily_quota=args.daily_quota,
        concurrency_limit=args.concurrency,
        sq_monthly_fee=sq_monthly_fee,
        sc_monthly_fee=sc_monthly_fee,
        model_pricing=model_pricing,
        subscriptions=subscriptions,
    )

    # Store active S_C config for strategy classes to read
    config["active_sc_subscription"] = sc_config

    loader = DataLoader(config)

    # Map dataset names to paths
    dataset_paths = {
        "sharegpt": "data/sharegpt_burstgpt/converted.csv",
        "freeinference": "data/freeinference_logs.csv",
        "rednote": "data/rednote_logs.csv",
    }

    # Load dataset (with caching)
    try:
        # Try loading from cache first (unless --no-cache)
        requests = None
        if not args.no_cache:
            requests = load_cached_dataset(args.data)

        if requests is None:
            # Load from CSV
            dataset_path = dataset_paths.get(args.data, args.data)
            requests = loader.load(dataset_path)

            # Save to cache for next time
            if not args.no_cache:
                save_dataset_cache(args.data, requests)

        # Apply limit if specified
        if args.limit:
            logger.info(f"Limiting to first {args.limit} requests")
            requests = requests[: args.limit]

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    logger.info(f"Loaded {len(requests):,} requests")

    results = {}

    # 1. ILP Optimal (The Gold Standard)
    # Note: ILP can be slow for large datasets. Using small delta or limiting scope might be needed.
    # For this script, we assume the user knows what they are doing or runs on small data first.
    # We'll use a larger delta (e.g. 60s) to make it faster for testing, or 1.0s for accuracy.
    # Warning: 1.0s on 30 days of data will be very slow.
    # Let's default to a safe delta or warn.
    # Convert --latency-slo: -1 means unlimited (None), 0+ means that many slots
    latency_slo = None if args.latency_slo < 0 else args.latency_slo
    logger.info(
        f"Running ILP Optimal Strategy (delta={args.delta}s, solver={args.solver}, "
        f"latency_slo={'unlimited' if latency_slo is None else latency_slo})..."
    )
    results["ilp_optimal"] = run_single_experiment(
        requests,
        ILPOptimalStrategy,
        config,
        "ILP-Optimal",
        delta=args.delta,
        daily_quota=args.daily_quota,
        concurrency_limit=args.concurrency,
        solver=args.solver,
        dataset_name=args.data,
        use_cache=not args.no_cache,
        max_workers=args.workers,
        latency_slo=latency_slo,
    )

    # 2. Baselines
    logger.info("Running Baselines...")

    # B2: Daily Quota Only
    results["daily_quota_only"] = run_single_experiment(
        requests, DailyQuotaOnlyStrategy, config, "Daily-Quota-Only", daily_quota=args.daily_quota
    )

    # B3: Concurrency Only (Offline optimal for S_C)
    results["concurrency_only"] = run_single_experiment(
        requests,
        ConcurrencyOnlyStrategy,
        config,
        "Concurrency-Only",
        concurrency_limit=args.concurrency,
        delta=args.delta,
        dataset_name=args.data,
        max_workers=args.workers,
    )

    # Save Results (per dataset)
    output_file = output_dir / f"stage2_results_{args.data}.json"

    # Clean non-serializable
    def clean_result(obj):
        if isinstance(obj, dict):
            return {k: clean_result(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_result(v) for v in obj]
        elif callable(obj):
            return str(obj)
        else:
            return obj

    with open(output_file, "w") as f:
        json.dump(clean_result(results), f, indent=2)

    logger.info(f"Results saved to {output_file}")

    # Print Summary
    print("\n" + "=" * 60)
    print("STAGE 2 EXPERIMENT SUMMARY")
    print("=" * 60)
    for name, res in results.items():
        total = res["costs"]["total"]
        print(f"{name:<30}: ${total:.2f}")


if __name__ == "__main__":
    main()
