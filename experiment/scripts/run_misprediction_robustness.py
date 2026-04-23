#!/usr/bin/env python3
"""Misprediction robustness experiment.

Sweeps the oracle predictor's output with controlled distortion and measures
how the primal-dual cost router degrades. Produces evidence for the paper
claim that RouteWise's cost router is structurally insensitive to predictor
miscalibration.

Two sweeps:
    Bias sweep: `predicted = true * bias_factor`, for
        bias_factor in {0.5, 0.67, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0}.
        Preserves request ordering -- isolates the effect of uniform
        miscalibration on shadow-price thresholds.
    Noise sweep: `predicted = true * exp(noise_std * Z)`, for
        noise_std in {0.0, 0.1, 0.3, 0.5, 1.0}.
        Breaks request ordering -- isolates the effect of per-request
        prediction variance.

The primal-dual threshold uses L, U bounds computed from the true cost
distribution (not predictor output), so the threshold itself is invariant
across runs; only the predictor value `v_hat` changes.

Baselines Optimal/Greedy/All-API/Oracle are cached across bias/noise sweeps
(they do not depend on predictor configuration).

Usage:
    python experiment/scripts/run_misprediction_robustness.py
    python experiment/scripts/run_misprediction_robustness.py --smoke
    python experiment/scripts/run_misprediction_robustness.py --data burstgpt --plan Pro
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiment.config import ExperimentConfig
from experiment.cost import CostCalculator
from experiment.data.loader import DataLoader
from experiment.predictors import (
    BiasedOraclePredictor,
    OracleOutputPredictor,
)
from experiment.quota import QuotaManager
from experiment.simulator import OfflineSimulator
from experiment.strategies.all_api import AllAPIStrategy
from experiment.strategies.online import (
    GreedyOnlineStrategy,
    PrimalDualOnlineStrategy,
)
from experiment.strategies.stage1_optimal import OptimalStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

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

QUOTA_PLANS = {
    "Base": {"monthly_fee": 3.0, "daily_quota": 300},
    "Plus": {"monthly_fee": 10.0, "daily_quota": 2000},
    "Pro": {"monthly_fee": 20.0, "daily_quota": 5000},
}

# Bias factors match Juncheng's Apr 17 guidance: "10%, 20%, 50% shorter or longer",
# extended with 0.5/2.0 endpoints to probe aggressive miscalibration.
DEFAULT_BIAS_FACTORS = [0.5, 0.667, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]

# Log-normal noise std sweep. noise_std=1.0 corresponds to ~1.65x spread
# between 16th and 84th percentiles (i.e., realistic "EMA has noise" regime).
DEFAULT_NOISE_STDS = [0.0, 0.1, 0.3, 0.5, 1.0]

# Number of seeds for noise runs (deterministic-bias runs only need 1 seed).
NOISE_SEEDS = [42, 43, 44]

# Smoke test configuration (small request slice, few sweep points).
SMOKE_NUM_REQUESTS = 10_000
SMOKE_BIAS_FACTORS = [0.5, 1.0, 2.0]
SMOKE_NOISE_STDS = [0.0, 0.3]


def compute_value_bounds(
    requests: list,
    cost_calculator: CostCalculator,
    sample_size: int = 10_000,
) -> tuple[float, float]:
    """Compute L and U bounds for shadow-price threshold.

    Uses 5th and 95th percentile of per-request API cost, matching
    paper default and run_oracle_experiment.py.

    Args:
        requests: Full request list.
        cost_calculator: Cost calculator instance.
        sample_size: Number of requests to use for percentile estimation.

    Returns:
        Tuple (L, U) of cost bounds.
    """
    costs: list[float] = []
    for r in requests[:sample_size]:
        try:
            c = cost_calculator.calculate_cost_by_model(r)
        except Exception:
            try:
                _, c = cost_calculator.get_cheapest_api_provider(r)
            except Exception:
                c = 0.0
        costs.append(c)
    costs.sort()
    if not costs:
        return 0.0001, 0.01
    L = costs[int(len(costs) * 0.05)]
    U = costs[int(len(costs) * 0.95)]
    return L, U


def run_pd_with_predictor(
    requests: list,
    config_dict: dict,
    cost_calculator: CostCalculator,
    daily_quota: int,
    value_bounds: tuple[float, float],
    predictor,
    label: str,
) -> dict:
    """Run primal-dual routing with a given output predictor.

    Args:
        requests: Request list to replay.
        config_dict: Config dict for simulator.
        cost_calculator: Cost calculator.
        daily_quota: Daily subscription quota Q.
        value_bounds: (L, U) tuple for threshold.
        predictor: OutputTokenPredictor instance.
        label: Human-readable label for logging.

    Returns:
        Result dict with total cost, api cost, quota utilization, and timing.
    """
    L, U = value_bounds
    quota_mgr = QuotaManager(daily_quota)
    strategy = PrimalDualOnlineStrategy(
        cost_calculator,
        quota_mgr,
        config_dict,
        daily_quota=daily_quota,
        sq_min_value=L,
        sq_max_value=U,
        concurrency_limit=0,
        output_predictor=predictor,
    )
    t0 = time.time()
    simulator = OfflineSimulator(requests, strategy, config_dict)
    result = simulator.run()
    elapsed = time.time() - t0
    d = result.to_dict()
    d["_elapsed_sec"] = round(elapsed, 2)
    d["_label"] = label
    return d


def run_baseline(
    requests: list,
    config_dict: dict,
    strategy,
    label: str,
) -> dict:
    """Run a baseline strategy (no predictor dependency).

    Args:
        requests: Request list.
        config_dict: Config dict.
        strategy: Strategy instance.
        label: Human-readable label.

    Returns:
        Result dict with timing.
    """
    t0 = time.time()
    simulator = OfflineSimulator(requests, strategy, config_dict)
    result = simulator.run()
    elapsed = time.time() - t0
    d = result.to_dict()
    d["_elapsed_sec"] = round(elapsed, 2)
    d["_label"] = label
    return d


def apply_quota_to_config(
    config_dict: dict,
    plan: dict,
) -> None:
    """Mutate config_dict in place to set the subscription plan quota."""
    sub_key = None
    for key, p in config_dict["providers"].items():
        if hasattr(p, "is_subscription") and p.is_subscription():
            sub_key = key
            break
    if sub_key is None:
        return
    old = config_dict["providers"][sub_key]
    config_dict["providers"][sub_key] = type(old)(
        name=old.name,
        type=old.type,
        monthly_fee=plan["monthly_fee"],
        daily_quota=plan["daily_quota"],
    )


def run_sweep_for_plan(
    requests: list,
    config: ExperimentConfig,
    plan_name: str,
    plan: dict,
    bias_factors: list[float],
    noise_stds: list[float],
    noise_seeds: list[int],
    dataset_info: dict,
) -> dict:
    """Run misprediction sweep for a single quota plan.

    Args:
        requests: Full request list for the dataset.
        config: Experiment config.
        plan_name: Plan name (Base/Plus/Pro).
        plan: Plan dict with monthly_fee and daily_quota.
        bias_factors: List of bias factors to sweep.
        noise_stds: List of log-normal noise stds to sweep.
        noise_seeds: List of seeds for noise runs.
        dataset_info: Dataset info dict (for model pricing override).

    Returns:
        Results dict keyed by sweep type and configuration.
    """
    config_dict = config.to_dict()

    if not dataset_info["multimodel"]:
        config_dict.pop("model_pricing", None)
    if dataset_info["model_override"]:
        model_pricing = config_dict.get("model_pricing", {})
        if dataset_info["model_override"] in model_pricing:
            model_pricing["default"] = model_pricing[dataset_info["model_override"]]
            config_dict["model_pricing"] = model_pricing

    apply_quota_to_config(config_dict, plan)
    daily_quota = plan["daily_quota"]

    cost_calculator = CostCalculator(
        config_dict["providers"],
        config_dict.get("model_pricing"),
    )
    L, U = compute_value_bounds(requests, cost_calculator)
    logger.info(f"  Plan={plan_name} Q={daily_quota}  L={L:.6f} U={U:.6f}")

    results: dict = {
        "plan": plan_name,
        "quota": daily_quota,
        "L": L,
        "U": U,
        "baselines": {},
        "bias_sweep": [],
        "noise_sweep": [],
    }

    # --- Cached baselines (predictor-independent) ---
    logger.info("  Running baselines (All-API, Greedy, Optimal, PD-Oracle)...")

    quota_mgr = QuotaManager(daily_quota)
    results["baselines"]["All-API"] = run_baseline(
        requests,
        config_dict,
        AllAPIStrategy(cost_calculator, quota_mgr, config_dict),
        "All-API",
    )
    logger.info(
        f"    All-API cost={results['baselines']['All-API']['costs']['total']:.2f} "
        f"({results['baselines']['All-API']['_elapsed_sec']}s)"
    )

    quota_mgr = QuotaManager(daily_quota)
    results["baselines"]["Greedy"] = run_baseline(
        requests,
        config_dict,
        GreedyOnlineStrategy(
            cost_calculator,
            quota_mgr,
            config_dict,
            daily_quota=daily_quota,
            concurrency_limit=0,
        ),
        "Greedy",
    )
    logger.info(
        f"    Greedy cost={results['baselines']['Greedy']['costs']['total']:.2f} "
        f"({results['baselines']['Greedy']['_elapsed_sec']}s)"
    )

    quota_mgr = QuotaManager(daily_quota)
    results["baselines"]["Optimal"] = run_baseline(
        requests,
        config_dict,
        OptimalStrategy(cost_calculator, quota_mgr, config_dict),
        "Optimal",
    )
    logger.info(
        f"    Optimal cost={results['baselines']['Optimal']['costs']['total']:.2f} "
        f"({results['baselines']['Optimal']['_elapsed_sec']}s)"
    )

    results["baselines"]["PD-Oracle"] = run_pd_with_predictor(
        requests,
        config_dict,
        cost_calculator,
        daily_quota,
        (L, U),
        OracleOutputPredictor(),
        "PD-Oracle",
    )
    logger.info(
        f"    PD-Oracle cost={results['baselines']['PD-Oracle']['costs']['total']:.2f} "
        f"({results['baselines']['PD-Oracle']['_elapsed_sec']}s)"
    )

    optimal_cost = results["baselines"]["Optimal"]["costs"]["total"]
    oracle_cost = results["baselines"]["PD-Oracle"]["costs"]["total"]

    # --- Bias sweep (deterministic, 1 seed per point) ---
    logger.info(f"  Bias sweep ({len(bias_factors)} points)...")
    for bias in bias_factors:
        # bias=1.0 is exactly PD-Oracle -- reuse cached result.
        if bias == 1.0:
            entry = {
                "bias_factor": 1.0,
                "noise_std": 0.0,
                "seed": None,
                "total_cost": oracle_cost,
                "relative_cost_vs_optimal": oracle_cost / optimal_cost
                if optimal_cost > 0
                else float("inf"),
                "relative_cost_vs_oracle": 1.0,
                "elapsed_sec": 0.0,
                "cached_from": "PD-Oracle",
            }
            results["bias_sweep"].append(entry)
            continue

        predictor = BiasedOraclePredictor(bias_factor=bias, noise_std=0.0, seed=42)
        res = run_pd_with_predictor(
            requests,
            config_dict,
            cost_calculator,
            daily_quota,
            (L, U),
            predictor,
            f"PD-Biased(bias={bias})",
        )
        total = res["costs"]["total"]
        entry = {
            "bias_factor": bias,
            "noise_std": 0.0,
            "seed": 42,
            "total_cost": total,
            "relative_cost_vs_optimal": total / optimal_cost if optimal_cost > 0 else float("inf"),
            "relative_cost_vs_oracle": total / oracle_cost if oracle_cost > 0 else float("inf"),
            "elapsed_sec": res["_elapsed_sec"],
        }
        results["bias_sweep"].append(entry)
        logger.info(
            f"    bias={bias:.3f}  cost={total:.2f}  "
            f"cr_vs_optimal={entry['relative_cost_vs_optimal']:.4f}  "
            f"cr_vs_oracle={entry['relative_cost_vs_oracle']:.4f}  "
            f"({res['_elapsed_sec']}s)"
        )

    # --- Noise sweep (stochastic, multiple seeds per point) ---
    logger.info(
        f"  Noise sweep ({len(noise_stds)} points x {len(noise_seeds)} seeds)..."
    )
    for noise_std in noise_stds:
        if noise_std == 0.0:
            # noise=0 is oracle -- one seed, cached.
            entry = {
                "noise_std": 0.0,
                "bias_factor": 1.0,
                "seeds": [None],
                "total_costs": [oracle_cost],
                "relative_cost_vs_optimal_mean": oracle_cost / optimal_cost
                if optimal_cost > 0
                else float("inf"),
                "relative_cost_vs_optimal_std": 0.0,
                "relative_cost_vs_oracle_mean": 1.0,
                "relative_cost_vs_oracle_std": 0.0,
                "cached_from": "PD-Oracle",
            }
            results["noise_sweep"].append(entry)
            continue

        total_costs: list[float] = []
        elapsed_list: list[float] = []
        for seed in noise_seeds:
            predictor = BiasedOraclePredictor(
                bias_factor=1.0, noise_std=noise_std, seed=seed
            )
            res = run_pd_with_predictor(
                requests,
                config_dict,
                cost_calculator,
                daily_quota,
                (L, U),
                predictor,
                f"PD-Noise(sigma={noise_std},seed={seed})",
            )
            total_costs.append(res["costs"]["total"])
            elapsed_list.append(res["_elapsed_sec"])

        rc_optimal = [c / optimal_cost for c in total_costs] if optimal_cost > 0 else []
        rc_oracle = [c / oracle_cost for c in total_costs] if oracle_cost > 0 else []
        n = len(total_costs)
        mean_optimal = sum(rc_optimal) / n
        mean_oracle = sum(rc_oracle) / n
        std_optimal = (sum((x - mean_optimal) ** 2 for x in rc_optimal) / n) ** 0.5
        std_oracle = (sum((x - mean_oracle) ** 2 for x in rc_oracle) / n) ** 0.5

        entry = {
            "noise_std": noise_std,
            "bias_factor": 1.0,
            "seeds": list(noise_seeds),
            "total_costs": total_costs,
            "relative_cost_vs_optimal_mean": mean_optimal,
            "relative_cost_vs_optimal_std": std_optimal,
            "relative_cost_vs_oracle_mean": mean_oracle,
            "relative_cost_vs_oracle_std": std_oracle,
            "elapsed_sec_sum": sum(elapsed_list),
        }
        results["noise_sweep"].append(entry)
        logger.info(
            f"    sigma={noise_std:.2f}  "
            f"cr_vs_optimal={mean_optimal:.4f}+-{std_optimal:.4f}  "
            f"cr_vs_oracle={mean_oracle:.4f}+-{std_oracle:.4f}  "
            f"({sum(elapsed_list):.1f}s)"
        )

    return results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Misprediction robustness experiment (bias + noise sweeps)"
    )
    parser.add_argument(
        "--data",
        type=str,
        choices=[*list(DATASETS.keys()), "all"],
        default="burstgpt",
        help="Dataset to use",
    )
    parser.add_argument(
        "--plan",
        type=str,
        choices=[*list(QUOTA_PLANS.keys()), "all"],
        default="Pro",
        help="Quota plan (default: Pro, matching paper main results)",
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
        default="experiment/results/misprediction",
        help="Output directory for results",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: 10K requests, 3 bias points, 2 noise points",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=0,
        help="Truncate trace to this many requests (0 = full trace)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        bias_factors = SMOKE_BIAS_FACTORS
        noise_stds = SMOKE_NOISE_STDS
        noise_seeds = [42]
        num_requests = SMOKE_NUM_REQUESTS
        output_file_suffix = "_smoke"
    else:
        bias_factors = DEFAULT_BIAS_FACTORS
        noise_stds = DEFAULT_NOISE_STDS
        noise_seeds = NOISE_SEEDS
        num_requests = args.num_requests
        output_file_suffix = ""

    dataset_names = list(DATASETS.keys()) if args.data == "all" else [args.data]
    plan_names = list(QUOTA_PLANS.keys()) if args.plan == "all" else [args.plan]

    all_results: dict = {
        "experiment": "misprediction_robustness",
        "bias_factors": bias_factors,
        "noise_stds": noise_stds,
        "noise_seeds": noise_seeds,
        "num_requests_cap": num_requests,
        "datasets": {},
    }

    for dataset_name in dataset_names:
        ds = DATASETS[dataset_name]
        logger.info(f"\n{'#' * 60}")
        logger.info(f"Dataset: {ds['description']}")
        logger.info(f"{'#' * 60}")

        if not Path(ds["path"]).exists():
            logger.warning(f"Dataset not found: {ds['path']}, skipping")
            continue

        config = ExperimentConfig(args.config)
        config_dict = config.to_dict()
        if not ds["multimodel"]:
            config_dict.pop("model_pricing", None)

        loader = DataLoader(config_dict)
        requests = loader.load(ds["path"], model_override=ds["model_override"])

        if num_requests > 0 and len(requests) > num_requests:
            requests = requests[:num_requests]

        stats = loader.get_statistics(requests)
        logger.info(
            f"Loaded {stats['total_requests']} requests over {stats['num_days']} days"
        )

        dataset_results: dict = {
            "description": ds["description"],
            "num_requests": stats["total_requests"],
            "num_days": stats["num_days"],
            "plans": {},
        }

        for plan_name in plan_names:
            plan = QUOTA_PLANS[plan_name]
            logger.info(f"\n--- Plan: {plan_name} (Q={plan['daily_quota']}) ---")
            plan_results = run_sweep_for_plan(
                requests,
                config,
                plan_name,
                plan,
                bias_factors,
                noise_stds,
                noise_seeds,
                ds,
            )
            dataset_results["plans"][plan_name] = plan_results

        all_results["datasets"][dataset_name] = dataset_results

    output_file = output_dir / f"misprediction_results{output_file_suffix}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_file}")

    # Print summary table.
    print("\n" + "=" * 72)
    print("  SUMMARY: Misprediction Robustness (Relative Cost vs PD-Oracle)")
    print("=" * 72)
    for dataset_name, ds_res in all_results["datasets"].items():
        for plan_name, plan_res in ds_res["plans"].items():
            print(f"\n  {dataset_name} | {plan_name} | "
                  f"Q={plan_res['quota']} | "
                  f"Oracle cost=${plan_res['baselines']['PD-Oracle']['costs']['total']:.2f}")
            print(f"  {'bias':>8} {'cr_opt':>8} {'cr_oracle':>10}")
            for entry in plan_res["bias_sweep"]:
                print(f"  {entry['bias_factor']:>8.3f} "
                      f"{entry['relative_cost_vs_optimal']:>8.4f} "
                      f"{entry['relative_cost_vs_oracle']:>10.4f}")
            print(f"  {'sigma':>8} {'cr_opt':>8} {'cr_oracle':>10}")
            for entry in plan_res["noise_sweep"]:
                print(f"  {entry['noise_std']:>8.2f} "
                      f"{entry['relative_cost_vs_optimal_mean']:>8.4f} "
                      f"{entry['relative_cost_vs_oracle_mean']:>10.4f}")


if __name__ == "__main__":
    main()
