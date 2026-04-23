#!/usr/bin/env python3
"""Run online routing experiments.

This script evaluates online routing strategies against offline optimal baselines.

Experiments:
1. Stage 1 (S_Q + S_A): Daily quota optimization
   - Dataset: BurstGPT (no latency required)
   - Strategies: Greedy, PrimalDual, LA-PD (P10/P20/P50), Offline-Optimal

2. Stage 2 (S_Q + S_C + S_A): Joint optimization
   - Dataset: rednote/freeinference (latency required)
   - Strategies: Greedy, PrimalDual, LA-PD-Unified, Offline-Optimal (ILP)

Usage:
    python experiment/scripts/run_online_experiments.py --stage 1
    python experiment/scripts/run_online_experiments.py --stage 2
    python experiment/scripts/run_online_experiments.py --stage both
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiment.config import ExperimentConfig
from experiment.cost import CostCalculator
from experiment.data.loader import DataLoader
from experiment.predictors import EMAOutputPredictor, HistogramOutputPredictor
from experiment.quota import QuotaManager
from experiment.simulator import OfflineSimulator

# Strategies
from experiment.strategies.all_api import AllAPIStrategy
from experiment.strategies.online import (
    GreedyOnlineStrategy,
    LAPDConfig,
    LearningAugmentedPrimalDualStrategy,
    LearningAugmentedUnifiedStrategy,
    PrimalDualOnlineStrategy,
)
from experiment.strategies.stage1_optimal import OptimalStrategy
from experiment.strategies.stage2_baselines import (
    ConcurrencyOnlyStrategy,
    DailyQuotaOnlyStrategy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _validate_model_pricing_coverage(
    requests,
    model_pricing: dict[str, dict[str, float]] | None,
    model_override: str | None,
) -> None:
    """Validate that all request models are priced when model_pricing is enabled.

    Args:
        requests: List of requests.
        model_pricing: Per-model pricing dictionary (per 1M tokens).
        model_override: Optional model override passed to the loader.

    Raises:
        ValueError: If model_pricing is enabled but required models are missing.
    """
    if not model_pricing:
        return

    if model_override:
        if model_override not in model_pricing:
            raise ValueError(
                f"--model '{model_override}' is not found in model_pricing. "
                "Please use a model name that exists in config/experiment.yaml:model_pricing, "
                "or add pricing for it."
            )
        return

    models = {r.model for r in requests if getattr(r, "model", None)}
    missing = sorted(m for m in models if m not in model_pricing)
    if missing:
        raise ValueError(
            "Missing model_pricing entries for dataset models: "
            f"{missing}. Either add pricing in config/experiment.yaml:model_pricing, "
            "or run with --model <target_model> to map all requests to a single priced model."
        )


def _safe_get_request_cost(cost_calculator: CostCalculator, request) -> float:
    """Get request cost with graceful fallback.

    Prefer per-model pricing when available; fall back to cheapest API provider
    pricing when the model is missing in model_pricing.

    Args:
        cost_calculator: Cost calculator.
        request: Request object.

    Returns:
        Cost in dollars.
    """
    try:
        return cost_calculator.calculate_cost_by_model(request)
    except Exception:
        try:
            _, cost = cost_calculator.get_cheapest_api_provider(request)
            return cost
        except Exception:
            return 0.0


def run_stage1_experiments(
    config: ExperimentConfig,
    data_path: str,
    output_dir: Path,
    model_override: str | None = None,
):
    """Run Stage 1 experiments: S_Q + S_A (daily quota optimization).

    Args:
        config: Experiment configuration
        data_path: Path to dataset (BurstGPT)
        output_dir: Output directory for results
        model_override: If set, map all requests to this model for pricing
    """
    logger.info("=" * 60)
    logger.info("Stage 1 Experiments: Daily Quota Optimization (S_Q + S_A)")
    logger.info("=" * 60)

    # Load data
    loader = DataLoader(config.to_dict())
    requests = loader.load(data_path, model_override=model_override)
    stats = loader.get_statistics(requests)

    if model_override:
        logger.info(f"Using model override: {model_override}")
    logger.info(f"Loaded {stats['total_requests']} requests over {stats['num_days']} days")

    # Get configuration
    config_dict = config.to_dict()
    num_subscriptions = config_dict["simulation"]["num_subscriptions"]
    subscription_provider = config.get_subscription_provider()
    daily_quota = subscription_provider.daily_quota * num_subscriptions

    logger.info(f"Daily quota: {daily_quota}")

    _validate_model_pricing_coverage(requests, config_dict.get("model_pricing"), model_override)

    # Calculate cost statistics for L/U estimation
    cost_calculator = CostCalculator(config_dict["providers"], config_dict.get("model_pricing"))
    costs = [_safe_get_request_cost(cost_calculator, r) for r in requests[:10000]]
    costs.sort()
    L = costs[int(len(costs) * 0.05)] if costs else 0.0001
    U = costs[int(len(costs) * 0.95)] if costs else 0.01
    logger.info(f"Estimated cost bounds: L={L:.6f}, U={U:.6f}")

    results = {}

    # 1. All-API Baseline
    logger.info("\n--- Running All-API Baseline ---")
    quota_mgr = QuotaManager(daily_quota)
    strategy = AllAPIStrategy(cost_calculator, quota_mgr, config_dict)
    simulator = OfflineSimulator(requests, strategy, config_dict)
    results["All-API"] = simulator.run().to_dict()

    # 2. Greedy Online
    logger.info("\n--- Running Greedy Online ---")
    quota_mgr = QuotaManager(daily_quota)
    strategy = GreedyOnlineStrategy(
        cost_calculator,
        quota_mgr,
        config_dict,
        daily_quota=daily_quota,
        concurrency_limit=0,  # Stage 1: no S_C
    )
    simulator = OfflineSimulator(requests, strategy, config_dict)
    results["Greedy-Online"] = simulator.run().to_dict()

    # 3. Primal-Dual Online (EMA baseline)
    logger.info("\n--- Running Primal-Dual Online (EMA) ---")
    quota_mgr = QuotaManager(daily_quota)
    strategy = PrimalDualOnlineStrategy(
        cost_calculator,
        quota_mgr,
        config_dict,
        daily_quota=daily_quota,
        sq_min_value=L,
        sq_max_value=U,
        concurrency_limit=0,  # Stage 1: no S_C
    )
    simulator = OfflineSimulator(requests, strategy, config_dict)
    results["PrimalDual-Online"] = simulator.run().to_dict()

    # 3b. PD-Hist (Ablation: PD decision rule with Histogram predictor)
    logger.info("\n--- Running PD-Hist (Ablation) ---")
    quota_mgr = QuotaManager(daily_quota)
    strategy = PrimalDualOnlineStrategy(
        cost_calculator,
        quota_mgr,
        config_dict,
        daily_quota=daily_quota,
        sq_min_value=L,
        sq_max_value=U,
        concurrency_limit=0,  # Stage 1: no S_C
        output_predictor=HistogramOutputPredictor(),  # Use Histogram instead of EMA
    )
    simulator = OfflineSimulator(requests, strategy, config_dict)
    results["PD-Hist"] = simulator.run().to_dict()

    # 4. LA-PD (P10) - Learning-Augmented with conservative quantile
    logger.info("\n--- Running LA-PD (P10) ---")
    quota_mgr = QuotaManager(daily_quota)
    lapd_config_p10 = LAPDConfig(quantile_for_lcb=0.10)
    strategy = LearningAugmentedPrimalDualStrategy(
        cost_calculator,
        quota_mgr,
        config_dict,
        daily_quota=daily_quota,
        sq_min_value=L,
        sq_max_value=U,
        lapd_config=lapd_config_p10,
    )
    simulator = OfflineSimulator(requests, strategy, config_dict)
    result = simulator.run()
    results["LA-PD-P10"] = result.to_dict()
    # Add calibration stats
    if hasattr(strategy, "get_stats"):
        results["LA-PD-P10"]["strategy_stats"] = strategy.get_stats()

    # 5. LA-PD (P20) - Ablation with moderate quantile
    logger.info("\n--- Running LA-PD (P20) ---")
    quota_mgr = QuotaManager(daily_quota)
    lapd_config_p20 = LAPDConfig(quantile_for_lcb=0.20)
    strategy = LearningAugmentedPrimalDualStrategy(
        cost_calculator,
        quota_mgr,
        config_dict,
        daily_quota=daily_quota,
        sq_min_value=L,
        sq_max_value=U,
        lapd_config=lapd_config_p20,
    )
    simulator = OfflineSimulator(requests, strategy, config_dict)
    result = simulator.run()
    results["LA-PD-P20"] = result.to_dict()
    if hasattr(strategy, "get_stats"):
        results["LA-PD-P20"]["strategy_stats"] = strategy.get_stats()

    # 6. LA-PD (P50) - Ablation with aggressive quantile
    logger.info("\n--- Running LA-PD (P50) ---")
    quota_mgr = QuotaManager(daily_quota)
    lapd_config_p50 = LAPDConfig(quantile_for_lcb=0.50)
    strategy = LearningAugmentedPrimalDualStrategy(
        cost_calculator,
        quota_mgr,
        config_dict,
        daily_quota=daily_quota,
        sq_min_value=L,
        sq_max_value=U,
        lapd_config=lapd_config_p50,
    )
    simulator = OfflineSimulator(requests, strategy, config_dict)
    result = simulator.run()
    results["LA-PD-P50"] = result.to_dict()
    if hasattr(strategy, "get_stats"):
        results["LA-PD-P50"]["strategy_stats"] = strategy.get_stats()

    # 7. LA-EMA (Ablation: LA-PD decision rule with EMA predictor)
    logger.info("\n--- Running LA-EMA (Ablation) ---")
    quota_mgr = QuotaManager(daily_quota)
    lapd_config_ema = LAPDConfig(quantile_for_lcb=0.10)
    strategy = LearningAugmentedPrimalDualStrategy(
        cost_calculator,
        quota_mgr,
        config_dict,
        daily_quota=daily_quota,
        sq_min_value=L,
        sq_max_value=U,
        lapd_config=lapd_config_ema,
        output_predictor=EMAOutputPredictor(),  # Use EMA instead of Histogram
    )
    simulator = OfflineSimulator(requests, strategy, config_dict)
    result = simulator.run()
    results["LA-EMA"] = result.to_dict()
    if hasattr(strategy, "get_stats"):
        results["LA-EMA"]["strategy_stats"] = strategy.get_stats()

    # 8. Offline Optimal
    logger.info("\n--- Running Offline Optimal ---")
    quota_mgr = QuotaManager(daily_quota)
    strategy = OptimalStrategy(cost_calculator, quota_mgr, config_dict)
    simulator = OfflineSimulator(requests, strategy, config_dict)
    results["Offline-Optimal"] = simulator.run().to_dict()

    # Calculate competitive ratios and savings
    optimal_cost = results["Offline-Optimal"]["costs"]["total"]
    all_api_cost = results["All-API"]["costs"]["total"]
    optimal_savings = all_api_cost - optimal_cost

    logger.info("\n" + "=" * 60)
    logger.info("Stage 1 Results Summary")
    logger.info("=" * 60)

    for name, res in results.items():
        total_cost = res["costs"]["total"]
        api_cost = res["costs"]["api"]
        ratio = total_cost / optimal_cost if optimal_cost > 0 else float("inf")
        savings = all_api_cost - total_cost
        savings_ratio = savings / optimal_savings if optimal_savings > 0 else 0.0
        logger.info(
            f"{name:20s}: Total=${total_cost:8.2f}, API=${api_cost:8.2f}, "
            f"Ratio={ratio:.4f}, Savings={savings_ratio:.1%}, Quota={res['quota_utilization']:.1%}"
        )

    # Print calibration stats for LA-PD strategies
    logger.info("\n" + "-" * 60)
    logger.info("Prediction Calibration (LA-PD strategies)")
    logger.info("-" * 60)
    for name, res in results.items():
        if "strategy_stats" in res and "predictor_calibration" in res["strategy_stats"]:
            cal = res["strategy_stats"]["predictor_calibration"]
            logger.info(
                f"{name:20s}: q10_cov={cal['q10_coverage']:.1%}, "
                f"q50_cov={cal['q50_coverage']:.1%}, q90_cov={cal['q90_coverage']:.1%}, "
                f"samples={cal['total']}"
            )

    # Save results
    output_file = output_dir / "stage1_online_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_file}")

    return results


def run_stage2_experiments(
    config: ExperimentConfig,
    data_path: str,
    output_dir: Path,
    model_override: str | None = None,
    sc_config_name: str = "local_gpu",
    sample_size: int | None = None,
):
    """Run Stage 2 experiments: S_Q + S_C + S_A (joint optimization).

    Args:
        config: Experiment configuration
        data_path: Path to dataset (rednote/freeinference with latency)
        output_dir: Output directory for results
        model_override: If set, map all requests to this model for pricing
        sc_config_name: Name of S_C configuration in YAML (local_gpu, featherless_premium, featherless_scale)
        sample_size: If set, sample N requests for faster testing
    """
    logger.info("=" * 60)
    logger.info("Stage 2 Experiments: Joint Optimization (S_Q + S_C + S_A)")
    logger.info(f"S_C Configuration: {sc_config_name}")
    logger.info("=" * 60)

    # Load data
    loader = DataLoader(config.to_dict())
    requests = loader.load(data_path, model_override=model_override)

    # Sample if requested (for faster testing)
    if sample_size is not None and sample_size < len(requests):
        logger.info(f"Sampling {sample_size} requests from {len(requests)} total")
        random.seed(42)  # For reproducibility
        requests = sorted(random.sample(requests, sample_size), key=lambda r: r.timestamp)

    stats = loader.get_statistics(requests)

    if model_override:
        logger.info(f"Using model override: {model_override}")
    logger.info(f"Loaded {stats['total_requests']} requests over {stats['num_days']} days")

    # Verify latency data
    has_latency = sum(1 for r in requests if r.latency_ms is not None)
    logger.info(
        f"Requests with latency: {has_latency}/{len(requests)} ({100*has_latency/len(requests):.1f}%)"
    )

    if has_latency < len(requests) * 0.9:
        logger.warning(
            "Less than 90% of requests have latency data. Stage 2 results may be inaccurate."
        )

    # Get configuration
    config_dict = config.to_dict()
    num_subscriptions = config_dict["simulation"]["num_subscriptions"]
    subscription_provider = config.get_subscription_provider()
    daily_quota = subscription_provider.daily_quota * num_subscriptions

    # S_C configuration from YAML by name
    subscriptions = config_dict.get("subscriptions", {})
    sc_config = subscriptions.get(sc_config_name, {})
    concurrency_limit = sc_config.get("concurrency_limit", 8)
    sc_monthly_fee = sc_config.get("monthly_fee", 0.0)
    sc_supported_models = sc_config.get("supported_models", {})

    # Store active S_C config under a canonical key for strategy classes to read
    config_dict["active_sc_subscription"] = sc_config

    # Store model override for ILP cache key differentiation
    if model_override:
        config_dict["model_override"] = model_override

    logger.info(f"Daily quota (S_Q): {daily_quota}")
    logger.info(f"S_C config: {sc_config_name}")
    logger.info(f"  - Concurrency limit: {concurrency_limit}")
    logger.info(f"  - Monthly fee: ${sc_monthly_fee:.2f}")
    logger.info(f"  - Supported models: {len(sc_supported_models)}")

    _validate_model_pricing_coverage(requests, config_dict.get("model_pricing"), model_override)

    # Calculate cost statistics
    cost_calculator = CostCalculator(config_dict["providers"], config_dict.get("model_pricing"))
    costs = [_safe_get_request_cost(cost_calculator, r) for r in requests[:10000]]
    costs.sort()
    L = costs[int(len(costs) * 0.05)] if costs else 0.0001
    U = costs[int(len(costs) * 0.95)] if costs else 0.01
    logger.info(f"Estimated cost bounds: L={L:.6f}, U={U:.6f}")

    results = {}

    # Store experiment configuration in results
    experiment_config = {
        "dataset": Path(data_path).stem,
        "sc_config": sc_config_name,
        "daily_quota": daily_quota,
        "concurrency_limit": concurrency_limit,
        "sc_monthly_fee": sc_monthly_fee,
        "num_days": stats["num_days"],
        "total_requests": len(requests),
        "sample_size": sample_size,
    }

    # 1. All-API Baseline (B1)
    logger.info("\n--- Running All-API Baseline (B1) ---")
    quota_mgr = QuotaManager(daily_quota)
    strategy = AllAPIStrategy(cost_calculator, quota_mgr, config_dict)
    simulator = OfflineSimulator(requests, strategy, config_dict)
    results["All-API"] = simulator.run().to_dict()

    # 2. S_Q-Only Baseline (B2) - Daily quota only, no concurrency
    logger.info("\n--- Running S_Q-Only Baseline (B2) ---")
    quota_mgr = QuotaManager(daily_quota)
    strategy = DailyQuotaOnlyStrategy(
        cost_calculator,
        quota_mgr,
        config_dict,
        daily_quota=daily_quota,
    )
    simulator = OfflineSimulator(requests, strategy, config_dict)
    results["SQ-Only"] = simulator.run().to_dict()

    # 3. S_C-Only Baseline (B3) - Concurrency only, no daily quota
    logger.info("\n--- Running S_C-Only Baseline (B3) ---")
    quota_mgr = QuotaManager(daily_quota)
    try:
        strategy = ConcurrencyOnlyStrategy(
            cost_calculator,
            quota_mgr,
            config_dict,
            concurrency_limit=concurrency_limit,
            dataset_name=f"stage2_online_{sc_config_name}",
            use_cache=True,
        )
        simulator = OfflineSimulator(requests, strategy, config_dict)
        results["SC-Only"] = simulator.run().to_dict()
    except Exception as e:
        logger.warning(f"S_C-Only baseline failed (requires ILP solver): {e}")

    # 4. Greedy Online (S_Q + S_C)
    logger.info("\n--- Running Greedy Online (Stage 2) ---")
    quota_mgr = QuotaManager(daily_quota)
    strategy = GreedyOnlineStrategy(
        cost_calculator,
        quota_mgr,
        config_dict,
        daily_quota=daily_quota,
        concurrency_limit=concurrency_limit,
    )
    simulator = OfflineSimulator(requests, strategy, config_dict)
    results["Greedy-Online-Stage2"] = simulator.run().to_dict()

    # 5. Primal-Dual Online (S_Q + S_C)
    logger.info("\n--- Running Primal-Dual Online (Stage 2) ---")
    quota_mgr = QuotaManager(daily_quota)
    strategy = PrimalDualOnlineStrategy(
        cost_calculator,
        quota_mgr,
        config_dict,
        daily_quota=daily_quota,
        sq_min_value=L,
        sq_max_value=U,
        concurrency_limit=concurrency_limit,
        queue_capacity=0,  # No queueing: route immediately or send to API
    )
    simulator = OfflineSimulator(requests, strategy, config_dict)
    results["PrimalDual-Online-Stage2"] = simulator.run().to_dict()

    # 6. LA-PD Unified (S_Q + S_C with learning)
    logger.info("\n--- Running LA-PD Unified (Stage 2) ---")
    quota_mgr = QuotaManager(daily_quota)
    lapd_config = LAPDConfig(quantile_for_lcb=0.10)
    strategy = LearningAugmentedUnifiedStrategy(
        cost_calculator,
        quota_mgr,
        config_dict,
        daily_quota=daily_quota,
        sq_min_value=L,
        sq_max_value=U,
        concurrency_limit=concurrency_limit,
        queue_capacity=0,  # No queueing: route immediately or send to API
        lapd_config=lapd_config,
    )
    simulator = OfflineSimulator(requests, strategy, config_dict)
    result = simulator.run()
    results["LA-PD-Unified-Stage2"] = result.to_dict()
    if hasattr(strategy, "get_stats"):
        results["LA-PD-Unified-Stage2"]["strategy_stats"] = strategy.get_stats()

    # 7. Offline Optimal (ILP) - if available
    try:
        from experiment.strategies.stage2_optimal import ILPOptimalStrategy

        logger.info("\n--- Running Offline Optimal (ILP) ---")
        quota_mgr = QuotaManager(daily_quota)
        strategy = ILPOptimalStrategy(
            cost_calculator,
            quota_mgr,
            config_dict,
            daily_quota=daily_quota,
            concurrency_limit=concurrency_limit,
            latency_slo=0,  # Zero-wait: no queueing allowed
            dataset_name=f"stage2_online_{sc_config_name}",
            use_cache=True,
        )
        simulator = OfflineSimulator(requests, strategy, config_dict)
        results["Offline-Optimal-ILP"] = simulator.run().to_dict()
    except ImportError:
        logger.warning("ILPOptimalStrategy not available, skipping offline optimal")
    except Exception as e:
        logger.warning(f"ILP optimization failed: {e}")

    # Post-process: adjust subscription costs per strategy.
    # Simulator only includes S_Q fee. We add S_C fee for strategies that use it,
    # and zero out subscription for All-API (no subscription baseline).
    num_days = results["All-API"]["num_days"]
    sc_prorated = sc_monthly_fee * (num_days / 30.0)
    sq_prorated = results["All-API"]["costs"]["subscription"]  # S_Q fee from simulator

    # Define which subscriptions each strategy uses
    strategy_subscriptions = {
        "All-API": (False, False),         # No subscriptions
        "SQ-Only": (True, False),          # S_Q only
        "SC-Only": (False, True),          # S_C only
        "Greedy-Online-Stage2": (True, True),
        "PrimalDual-Online-Stage2": (True, True),
        "LA-PD-Unified-Stage2": (True, True),
        "Offline-Optimal-ILP": (True, True),
    }

    for name, res in results.items():
        uses_sq, uses_sc = strategy_subscriptions.get(name, (True, True))
        correct_sub = (sq_prorated if uses_sq else 0.0) + (sc_prorated if uses_sc else 0.0)
        api_cost = res["costs"]["api"]
        res["costs"]["subscription"] = round(correct_sub, 4)
        res["costs"]["total"] = round(correct_sub + api_cost, 4)

    # Calculate competitive ratios and savings
    optimal_key = "Offline-Optimal-ILP" if "Offline-Optimal-ILP" in results else "All-API"
    optimal_cost = results[optimal_key]["costs"]["total"]
    all_api_cost = results["All-API"]["costs"]["total"]
    optimal_savings = all_api_cost - optimal_cost if optimal_key != "All-API" else 0.0

    # Add competitive ratios to each result
    for _name, res in results.items():
        total_cost = res["costs"]["total"]
        res["competitive_ratio"] = round(total_cost / optimal_cost, 4) if optimal_cost > 0 else None

    logger.info("\n" + "=" * 60)
    logger.info("Stage 2 Results Summary")
    logger.info(
        f"S_C Config: {sc_config_name} (C={concurrency_limit}, fee=${sc_monthly_fee:.2f}/mo)"
    )
    logger.info("=" * 60)

    logger.info(
        f"{'Strategy':<25} {'Total':>10} {'API':>10} {'Ratio':>8} {'Savings':>10} {'SubReqs':>10}"
    )
    logger.info("-" * 85)

    for name, res in results.items():
        total_cost = res["costs"]["total"]
        api_cost = res["costs"]["api"]
        ratio = res.get("competitive_ratio", float("inf"))
        savings = all_api_cost - total_cost
        savings_ratio = savings / optimal_savings if optimal_savings > 0 else 0.0
        sub_reqs = res["requests"]["subscription"]
        logger.info(
            f"{name:<25} ${total_cost:>9.2f} ${api_cost:>9.2f} {ratio:>8.4f} "
            f"{savings_ratio:>9.1%} {sub_reqs:>10}"
        )

    # Print calibration stats for LA-PD strategies
    logger.info("\n" + "-" * 60)
    logger.info("Prediction Calibration (LA-PD strategies)")
    logger.info("-" * 60)
    for name, res in results.items():
        if "strategy_stats" in res and "output_calibration" in res["strategy_stats"]:
            cal = res["strategy_stats"]["output_calibration"]
            logger.info(
                f"{name:25s}: q10_cov={cal['q10_coverage']:.1%}, "
                f"q50_cov={cal['q50_coverage']:.1%}, q90_cov={cal['q90_coverage']:.1%}"
            )

    # Wrap results with experiment config
    output_data = {
        "experiment_config": experiment_config,
        "strategies": results,
    }

    # Save results with S_C config name in filename
    output_file = output_dir / f"stage2_online_results_{sc_config_name}.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"\nResults saved to {output_file}")

    return output_data


def main():
    """Run online routing experiments with configurable strategies."""
    parser = argparse.ArgumentParser(description="Run online routing experiments")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["1", "2", "both"],
        default="1",
        help="Which stage to run (1, 2, or both)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment.yaml",
        help="Path to experiment config",
    )
    parser.add_argument(
        "--stage1-data",
        type=str,
        default="data/BurstGPT_1.csv",
        help="Dataset for Stage 1 (BurstGPT)",
    )
    parser.add_argument(
        "--stage2-data",
        type=str,
        default="data/rednote_logs.csv",
        help="Dataset for Stage 2 (with latency)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiment/results/online",
        help="Output directory for results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Target model for pricing. Use this to treat a dataset as single-model workload. "
        "E.g., '--model deepseek-r1' maps all BurstGPT requests to deepseek-r1 pricing.",
    )
    parser.add_argument(
        "--sc-config",
        type=str,
        choices=["local_gpu", "featherless_premium", "featherless_scale"],
        default="local_gpu",
        help="S_C (concurrency) configuration to use for Stage 2 (default: local_gpu)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample N requests for faster testing (useful for large datasets)",
    )

    args = parser.parse_args()

    # Load config
    config = ExperimentConfig(args.config)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiments
    try:
        if args.stage in ["1", "both"]:
            if Path(args.stage1_data).exists():
                run_stage1_experiments(config, args.stage1_data, output_dir, args.model)
            else:
                logger.error(f"Stage 1 data not found: {args.stage1_data}")

        if args.stage in ["2", "both"]:
            if Path(args.stage2_data).exists():
                run_stage2_experiments(
                    config,
                    args.stage2_data,
                    output_dir,
                    args.model,
                    sc_config_name=args.sc_config,
                    sample_size=args.sample,
                )
            else:
                logger.error(f"Stage 2 data not found: {args.stage2_data}")
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
