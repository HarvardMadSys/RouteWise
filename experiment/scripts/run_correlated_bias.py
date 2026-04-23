#!/usr/bin/env python3
"""Input-correlated bias experiment (adversarial robustness).

Unlike uniform bias (trivially preserves ordering), this experiment injects
bias that *correlates with request features*, breaking ordering at specific
points of the value distribution. Three modes tested:

    long_underestimate:
        Requests with input_tokens > median(input_tokens) have predicted
        output = true_output x bias_factor. Models "EMA trained on short
        context under-predicts long-context requests."

    short_overestimate:
        Requests with input_tokens <= median(input_tokens) have predicted
        output = true_output x bias_factor. Sanity-check direction.

    tail_underestimate:
        Requests with true_output >= P90(true_output) have predicted
        output = true_output x bias_factor. Worst-case adversarial:
        the highest-value requests are precisely the ones misjudged.

For each mode we sweep bias_factor in {0.3, 0.5, 0.7, 1.0, 1.5, 2.0}.
bias_factor=1.0 is equivalent to PD-Oracle.

Datasets: BurstGPT (Pro plan, paper main) and FreeInference (Plus plan,
most sensitive regime from the uniform sweep).

Usage:
    python experiment/scripts/run_correlated_bias.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiment.config import ExperimentConfig
from experiment.cost import CostCalculator
from experiment.data.loader import DataLoader
from experiment.predictors import (
    InputCorrelatedBiasedPredictor,
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

# Default focus: BurstGPT Pro (paper main) + FreeInference Plus (most sensitive).
DEFAULT_CONFIGS = [
    ("burstgpt", "Pro"),
    ("freeinference", "Plus"),
]

MODES = ("long_underestimate", "short_overestimate", "tail_underestimate")
BIAS_FACTORS = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]


def apply_quota_to_config(config_dict: dict, plan: dict) -> None:
    """Mutate config_dict to set the subscription plan quota."""
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


def compute_value_bounds(
    requests: list,
    cost_calculator: CostCalculator,
    sample_size: int = 10_000,
) -> tuple[float, float]:
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
    return costs[int(len(costs) * 0.05)], costs[int(len(costs) * 0.95)]


def compute_thresholds(requests: list, sample_size: int = 10_000) -> dict:
    """Compute input-token median and output-token P90 from the trace."""
    sample = requests[:sample_size] if len(requests) > sample_size else requests
    input_tokens = np.array([r.request_tokens for r in sample])
    output_tokens = np.array([r.response_tokens for r in sample])
    return {
        "median_input_tokens": int(np.median(input_tokens)),
        "p90_output_tokens": float(np.percentile(output_tokens, 90)),
        "p50_input_tokens": int(np.median(input_tokens)),
        "p50_output_tokens": float(np.median(output_tokens)),
    }


def run_with_predictor(
    requests: list,
    config_dict: dict,
    cost_calculator: CostCalculator,
    daily_quota: int,
    value_bounds: tuple[float, float],
    predictor,
) -> dict:
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
    return d


def run_sweep(
    dataset_name: str,
    plan_name: str,
    bias_factors: list[float],
    modes: list[str],
    config_path: str,
) -> dict:
    """Run correlated-bias sweep for one (dataset, plan) config."""
    ds = DATASETS[dataset_name]
    plan = QUOTA_PLANS[plan_name]

    config = ExperimentConfig(config_path)
    config_dict = config.to_dict()
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
    logger.info(
        f"Loaded {stats['total_requests']} requests over {stats['num_days']} days"
    )

    apply_quota_to_config(config_dict, plan)
    daily_quota = plan["daily_quota"]

    cost_calculator = CostCalculator(
        config_dict["providers"], config_dict.get("model_pricing")
    )
    L, U = compute_value_bounds(requests, cost_calculator)
    thresholds = compute_thresholds(requests)
    logger.info(
        f"  Plan={plan_name} Q={daily_quota}  L={L:.6f} U={U:.6f}"
    )
    logger.info(
        f"  Thresholds: median_input={thresholds['median_input_tokens']}, "
        f"P90_output={thresholds['p90_output_tokens']:.0f}"
    )

    results = {
        "dataset": dataset_name,
        "plan": plan_name,
        "quota": daily_quota,
        "L": L,
        "U": U,
        "thresholds": thresholds,
        "baselines": {},
        "sweeps": {mode: [] for mode in modes},
    }

    # --- Baselines (predictor-independent) ---
    logger.info("  Running baselines...")
    quota_mgr = QuotaManager(daily_quota)
    all_api = OfflineSimulator(
        requests,
        AllAPIStrategy(cost_calculator, quota_mgr, config_dict),
        config_dict,
    ).run()
    results["baselines"]["All-API"] = all_api.to_dict()

    quota_mgr = QuotaManager(daily_quota)
    greedy = OfflineSimulator(
        requests,
        GreedyOnlineStrategy(
            cost_calculator,
            quota_mgr,
            config_dict,
            daily_quota=daily_quota,
            concurrency_limit=0,
        ),
        config_dict,
    ).run()
    results["baselines"]["Greedy"] = greedy.to_dict()

    quota_mgr = QuotaManager(daily_quota)
    optimal = OfflineSimulator(
        requests,
        OptimalStrategy(cost_calculator, quota_mgr, config_dict),
        config_dict,
    ).run()
    results["baselines"]["Optimal"] = optimal.to_dict()

    oracle_res = run_with_predictor(
        requests,
        config_dict,
        cost_calculator,
        daily_quota,
        (L, U),
        OracleOutputPredictor(),
    )
    results["baselines"]["PD-Oracle"] = oracle_res

    optimal_cost = optimal.to_dict()["costs"]["total"]
    oracle_cost = oracle_res["costs"]["total"]
    logger.info(
        f"    Optimal=${optimal_cost:.2f}  Oracle=${oracle_cost:.2f}  "
        f"Greedy=${greedy.to_dict()['costs']['total']:.2f}"
    )

    # --- Correlated-bias sweeps ---
    for mode in modes:
        logger.info(f"  Mode = {mode}")
        for bias in bias_factors:
            if bias == 1.0:
                # bias=1.0 is exactly PD-Oracle; reuse.
                entry = {
                    "bias_factor": 1.0,
                    "total_cost": oracle_cost,
                    "relative_cost_vs_optimal": oracle_cost / optimal_cost
                    if optimal_cost > 0
                    else float("inf"),
                    "relative_cost_vs_oracle": 1.0,
                    "elapsed_sec": 0.0,
                    "biased_fraction": None,
                    "cached_from": "PD-Oracle",
                }
                results["sweeps"][mode].append(entry)
                continue

            if mode == "long_underestimate":
                predictor = InputCorrelatedBiasedPredictor(
                    mode=mode,
                    bias_factor=bias,
                    threshold_tokens=thresholds["median_input_tokens"],
                )
            elif mode == "short_overestimate":
                predictor = InputCorrelatedBiasedPredictor(
                    mode=mode,
                    bias_factor=bias,
                    threshold_tokens=thresholds["median_input_tokens"],
                )
            elif mode == "tail_underestimate":
                predictor = InputCorrelatedBiasedPredictor(
                    mode=mode,
                    bias_factor=bias,
                    tail_value_threshold=thresholds["p90_output_tokens"],
                )
            else:
                raise AssertionError(f"unreachable mode {mode}")

            res = run_with_predictor(
                requests,
                config_dict,
                cost_calculator,
                daily_quota,
                (L, U),
                predictor,
            )
            total = res["costs"]["total"]
            entry = {
                "bias_factor": bias,
                "total_cost": total,
                "relative_cost_vs_optimal": total / optimal_cost
                if optimal_cost > 0
                else float("inf"),
                "relative_cost_vs_oracle": total / oracle_cost
                if oracle_cost > 0
                else float("inf"),
                "elapsed_sec": res["_elapsed_sec"],
                "biased_fraction": predictor.get_biased_fraction(),
            }
            results["sweeps"][mode].append(entry)
            logger.info(
                f"    bias={bias:.2f}  cost=${total:.2f}  "
                f"cr_vs_optimal={entry['relative_cost_vs_optimal']:.4f}  "
                f"cr_vs_oracle={entry['relative_cost_vs_oracle']:.4f}  "
                f"biased_frac={entry['biased_fraction']:.2f}  "
                f"({res['_elapsed_sec']}s)"
            )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Input-correlated bias experiment (adversarial robustness)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiment/results/misprediction",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated list of dataset:plan pairs "
        "(default: burstgpt:Pro,freeinference:Plus)",
    )
    args = parser.parse_args()

    if args.datasets:
        configs = []
        for item in args.datasets.split(","):
            d, p = item.strip().split(":")
            configs.append((d.strip(), p.strip()))
    else:
        configs = DEFAULT_CONFIGS

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        "experiment": "correlated_bias",
        "modes": list(MODES),
        "bias_factors": BIAS_FACTORS,
        "configs": configs,
        "datasets": {},
    }

    for dataset_name, plan_name in configs:
        logger.info(f"\n{'#' * 60}")
        logger.info(f"Config: {dataset_name} x {plan_name}")
        logger.info(f"{'#' * 60}")
        result = run_sweep(
            dataset_name,
            plan_name,
            BIAS_FACTORS,
            list(MODES),
            args.config,
        )
        if dataset_name not in all_results["datasets"]:
            all_results["datasets"][dataset_name] = {"plans": {}}
        all_results["datasets"][dataset_name]["plans"][plan_name] = result

    output_file = output_dir / "correlated_bias_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_file}")

    # Print summary.
    print("\n" + "=" * 78)
    print("  SUMMARY: Input-correlated bias (Relative Cost vs PD-Oracle)")
    print("=" * 78)
    for dataset_name, plan_name in configs:
        res = all_results["datasets"][dataset_name]["plans"][plan_name]
        print(f"\n  {dataset_name} | {plan_name} | Q={res['quota']}  "
              f"Oracle=${res['baselines']['PD-Oracle']['costs']['total']:.2f}")
        for mode in MODES:
            print(f"    Mode: {mode}")
            print(f"      {'bias':>8} {'cr_opt':>8} {'cr_oracle':>10} {'biased_frac':>12}")
            for e in res["sweeps"][mode]:
                bf = f"{e['biased_fraction']:.2f}" if e['biased_fraction'] is not None else "(cached)"
                print(f"      {e['bias_factor']:>8.2f} "
                      f"{e['relative_cost_vs_optimal']:>8.4f} "
                      f"{e['relative_cost_vs_oracle']:>10.4f} "
                      f"{bf:>12}")


if __name__ == "__main__":
    main()
