#!/usr/bin/env python3
"""Stage 2 misprediction robustness experiment.

Unlike Stage 1 (quota only), Stage 2 also admits requests into a
concurrency-limited subscription S_C. The routing decision involves:

    G_Q = v_hat - psi(z)                        (for S_Q, quota)
    G_C = v_hat - lambda_t * weight * p_hat     (for S_C, concurrency)
    G_A = 0                                     (for API)

Both `v_hat` (value via output length predictor) and `p_hat` (duration
via latency predictor) are noisy in practice. This experiment biases
the output-length predictor while keeping the duration estimator as-is
(matches Juncheng's ask: output-length sensitivity). Result tells us
whether Stage 2's extra mechanism (CAPQ concurrency admission with
value-density ordering) amplifies or absorbs bias in the output
prediction.

Dataset: FreeInference (has latency_ms + ttft_ms required for S_C).
BurstGPT has no latency data and would fall back to a synthetic estimate,
so we skip it.

Usage:
    python experiment/scripts/run_misprediction_stage2.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiment.config import ExperimentConfig
from experiment.cost import CostCalculator
from experiment.data.loader import DataLoader
from experiment.data.schema import ProviderConfig, ProviderType
from experiment.predictors import (
    BiasedOraclePredictor,
    OracleOutputPredictor,
)
from experiment.quota import QuotaManager
from experiment.simulator import OfflineSimulator, SimulationResult
from experiment.strategies.online import PrimalDualOnlineStrategy
from experiment.strategies.stage2_baselines import (
    ConcurrencyOnlyStrategy,
    DailyQuotaOnlyStrategy,
)
from experiment.strategies.stage2_optimal import ILPOptimalStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Stage 2 S_C configurations from experiment.yaml.
# We use local_gpu (C=8, $0/mo) as the baseline to match paper convention.
SC_CONFIG_NAME = "local_gpu"
SC_CONCURRENCY = 8

# Bias sweep matches Stage 1 for direct comparison.
BIAS_FACTORS = [0.5, 0.667, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]

# Quota plan: Pro (Q=5000) matches paper main.
DAILY_QUOTA = 5000
SQ_MONTHLY_FEE = 20.0
SC_MONTHLY_FEE = 0.0


class DualSubscriptionSimulator(OfflineSimulator):
    """Simulator that applies both S_Q and S_C monthly fees based on usage.

    Replicates run_stage2_experiments.DualSubscriptionSimulator so that
    PrimalDualOnlineStrategy (which uses both) is charged both fees.
    """

    def run(self) -> SimulationResult:
        result = super().run()

        num_days = result.num_days
        if num_days == 0 and self.requests:
            num_days = self.requests[-1].day - self.requests[0].day + 1

        sq_provider = self.config["providers"]["daily-quota-sub"]
        sc_provider = self.config["providers"]["concurrency-sub"]

        sq_cost = sq_provider.monthly_fee * (num_days / 30.0)
        sc_cost = sc_provider.monthly_fee * (num_days / 30.0)

        # Primal-Dual strategy always uses both S_Q + S_C.
        total_sub_cost = sq_cost + sc_cost
        result.subscription_cost = total_sub_cost
        result.total_cost = total_sub_cost + result.api_cost

        return result


def build_stage2_config(
    config_path: str,
    daily_quota: int,
    sc_concurrency: int,
    subscriptions_from_yaml: dict,
) -> dict:
    """Build a Stage 2 config dict for PrimalDualOnlineStrategy.

    Structure: S_Q (daily-quota-sub) + S_C (concurrency-sub) + API fallback.
    """
    exp_config = ExperimentConfig(config_path)
    base = exp_config.to_dict()
    model_pricing = base.get("model_pricing", {})

    providers = {
        "daily-quota-sub": ProviderConfig(
            name="Daily Quota Subscription",
            type=ProviderType.SUBSCRIPTION,
            monthly_fee=SQ_MONTHLY_FEE,
            daily_quota=daily_quota,
        ),
        "concurrency-sub": ProviderConfig(
            name="Concurrency Subscription",
            type=ProviderType.SUBSCRIPTION,
            monthly_fee=SC_MONTHLY_FEE,
            daily_quota=0,
        ),
        # API fallback: preserve per-model pricing via model_pricing dict.
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
            "num_subscriptions": 1,
            "default_subscription": "daily-quota-sub",
            "subscription_provider": "daily-quota-sub",
            "default_api_fallback": "openai-chatgpt",
            "sq_count": 1,
            "sc_count": 1,
        },
        "dataset": {},
        "output": {},
        "model_pricing": model_pricing,
        "subscriptions": subscriptions_from_yaml,
    }

    # Active S_C subscription info (for PrimalDualOnlineStrategy to read multipliers).
    if SC_CONFIG_NAME in subscriptions_from_yaml:
        config["active_sc_subscription"] = subscriptions_from_yaml[SC_CONFIG_NAME]

    return config


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


def run_pd_stage2(
    requests: list,
    config: dict,
    value_bounds: tuple[float, float],
    daily_quota: int,
    sc_concurrency: int,
    predictor,
    label: str,
) -> dict:
    """Run PD Stage 2 routing with a given output predictor."""
    cost_calculator = CostCalculator(
        config["providers"], config.get("model_pricing")
    )
    quota_manager = QuotaManager(daily_quota=daily_quota, num_subscriptions=1)
    L, U = value_bounds
    strategy = PrimalDualOnlineStrategy(
        cost_calculator,
        quota_manager,
        config,
        daily_quota=daily_quota,
        sq_min_value=L,
        sq_max_value=U,
        concurrency_limit=sc_concurrency,
        queue_capacity=0,
        sq_monthly_fee=SQ_MONTHLY_FEE,
        sc_monthly_fee=SC_MONTHLY_FEE,
        output_predictor=predictor,
    )
    t0 = time.time()
    simulator = DualSubscriptionSimulator(requests, strategy, config)
    result = simulator.run()
    elapsed = time.time() - t0

    d = result.to_dict()
    d["_elapsed_sec"] = round(elapsed, 2)
    d["_label"] = label
    return d


def run_stage2_baselines(
    requests: list,
    config: dict,
    daily_quota: int,
    sc_concurrency: int,
    dataset_name: str,
    skip_ilp: bool = True,
) -> dict:
    """Run Stage 2 offline baselines (quota-only, concurrency-only, and optionally ILP)."""
    cost_calculator = CostCalculator(
        config["providers"], config.get("model_pricing")
    )
    results: dict = {}

    # Daily Quota Only (S_Q only).
    quota_mgr = QuotaManager(daily_quota=daily_quota, num_subscriptions=1)
    strategy = DailyQuotaOnlyStrategy(
        cost_calculator, quota_mgr, config, daily_quota=daily_quota
    )
    t0 = time.time()
    res = DualSubscriptionSimulator(requests, strategy, config).run()
    results["DailyQuotaOnly"] = res.to_dict()
    results["DailyQuotaOnly"]["_elapsed_sec"] = round(time.time() - t0, 2)

    # Concurrency Only (S_C only, offline optimal).
    quota_mgr = QuotaManager(daily_quota=daily_quota, num_subscriptions=1)
    strategy = ConcurrencyOnlyStrategy(
        cost_calculator,
        quota_mgr,
        config,
        concurrency_limit=sc_concurrency,
        delta=300.0,
        dataset_name=dataset_name,
    )
    t0 = time.time()
    res = DualSubscriptionSimulator(requests, strategy, config).run()
    results["ConcurrencyOnly"] = res.to_dict()
    results["ConcurrencyOnly"]["_elapsed_sec"] = round(time.time() - t0, 2)

    # ILP joint optimal (slow, skipped by default).
    if not skip_ilp:
        quota_mgr = QuotaManager(daily_quota=daily_quota, num_subscriptions=1)
        strategy = ILPOptimalStrategy(
            cost_calculator,
            quota_mgr,
            config,
            delta=300.0,
            daily_quota=daily_quota,
            concurrency_limit=sc_concurrency,
            solver="gurobi",
            dataset_name=dataset_name,
            use_cache=True,
        )
        t0 = time.time()
        res = DualSubscriptionSimulator(requests, strategy, config).run()
        results["ILPOptimal"] = res.to_dict()
        results["ILPOptimal"]["_elapsed_sec"] = round(time.time() - t0, 2)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2 misprediction robustness (S_Q + S_C + API)"
    )
    parser.add_argument("--config", type=str, default="config/experiment.yaml")
    parser.add_argument(
        "--data", type=str, default="freeinference", choices=["freeinference"]
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/freeinference_logs.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiment/results/misprediction",
    )
    parser.add_argument(
        "--daily-quota",
        type=int,
        default=DAILY_QUOTA,
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=SC_CONCURRENCY,
    )
    parser.add_argument(
        "--include-ilp",
        action="store_true",
        help="Include ILP offline optimal baseline (slow, minutes per run)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Truncate trace to first N requests (0 = full)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading experiment config from {args.config}")
    exp_config = ExperimentConfig(args.config)
    subscriptions_from_yaml = exp_config.to_dict().get("subscriptions", {})

    config = build_stage2_config(
        args.config,
        daily_quota=args.daily_quota,
        sc_concurrency=args.concurrency,
        subscriptions_from_yaml=subscriptions_from_yaml,
    )

    loader = DataLoader(config)
    logger.info(f"Loading {args.data} dataset from {args.data_path}...")
    requests = loader.load(args.data_path)
    if args.limit > 0 and len(requests) > args.limit:
        requests = requests[: args.limit]
    stats = loader.get_statistics(requests)
    logger.info(
        f"Loaded {stats['total_requests']} requests over {stats['num_days']} days"
    )

    cost_calculator = CostCalculator(
        config["providers"], config.get("model_pricing")
    )
    L, U = compute_value_bounds(requests, cost_calculator)
    logger.info(
        f"Stage 2  Q={args.daily_quota}  C={args.concurrency}  "
        f"L={L:.6f}  U={U:.6f}"
    )

    # --- Offline baselines ---
    logger.info("Running offline baselines (Stage 2)...")
    baselines = run_stage2_baselines(
        requests,
        config,
        args.daily_quota,
        args.concurrency,
        args.data,
        skip_ilp=not args.include_ilp,
    )
    for name, r in baselines.items():
        logger.info(
            f"  {name}: total=${r['costs']['total']:.2f}  "
            f"api=${r['costs']['api']:.2f}  sub=${r['costs']['subscription']:.2f}  "
            f"({r['_elapsed_sec']}s)"
        )

    # --- Oracle (PD-Oracle for Stage 2) ---
    logger.info("Running PD-Oracle Stage 2...")
    oracle_res = run_pd_stage2(
        requests,
        config,
        (L, U),
        args.daily_quota,
        args.concurrency,
        OracleOutputPredictor(),
        "PD-Oracle-Stage2",
    )
    oracle_cost = oracle_res["costs"]["total"]
    logger.info(
        f"  PD-Oracle-Stage2 cost=${oracle_cost:.2f}  "
        f"api=${oracle_res['costs']['api']:.2f}  "
        f"sub=${oracle_res['costs']['subscription']:.2f}  "
        f"({oracle_res['_elapsed_sec']}s)"
    )

    # --- Bias sweep ---
    logger.info(f"Bias sweep ({len(BIAS_FACTORS)} points)...")
    bias_entries: list[dict] = []
    for bias in BIAS_FACTORS:
        if bias == 1.0:
            entry = {
                "bias_factor": 1.0,
                "total_cost": oracle_cost,
                "api_cost": oracle_res["costs"]["api"],
                "sub_cost": oracle_res["costs"]["subscription"],
                "relative_cost_vs_oracle": 1.0,
                "elapsed_sec": 0.0,
                "cached_from": "PD-Oracle-Stage2",
            }
            bias_entries.append(entry)
            continue

        predictor = BiasedOraclePredictor(bias_factor=bias, noise_std=0.0, seed=42)
        res = run_pd_stage2(
            requests,
            config,
            (L, U),
            args.daily_quota,
            args.concurrency,
            predictor,
            f"PD-BiasedOracle(bias={bias})",
        )
        total = res["costs"]["total"]
        entry = {
            "bias_factor": bias,
            "total_cost": total,
            "api_cost": res["costs"]["api"],
            "sub_cost": res["costs"]["subscription"],
            "relative_cost_vs_oracle": total / oracle_cost
            if oracle_cost > 0
            else float("inf"),
            "elapsed_sec": res["_elapsed_sec"],
        }
        bias_entries.append(entry)
        logger.info(
            f"  bias={bias:.3f}  total=${total:.2f}  "
            f"cr_vs_oracle={entry['relative_cost_vs_oracle']:.4f}  "
            f"({res['_elapsed_sec']}s)"
        )

    # --- Save results ---
    all_results = {
        "experiment": "stage2_misprediction_robustness",
        "dataset": args.data,
        "daily_quota": args.daily_quota,
        "concurrency_limit": args.concurrency,
        "sc_config_name": SC_CONFIG_NAME,
        "bias_factors": BIAS_FACTORS,
        "value_bounds": {"L": L, "U": U},
        "baselines": baselines,
        "oracle": oracle_res,
        "bias_sweep": bias_entries,
        "num_requests": len(requests),
    }
    output_file = output_dir / "misprediction_stage2_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_file}")

    # --- Summary ---
    print("\n" + "=" * 78)
    print(f"  STAGE 2 MISPREDICTION ({args.data.upper()}, "
          f"Q={args.daily_quota}, C={args.concurrency})")
    print("=" * 78)
    print(f"  Oracle cost:     ${oracle_cost:.2f}")
    for name, r in baselines.items():
        c = r["costs"]["total"]
        rel = c / oracle_cost if oracle_cost > 0 else float("inf")
        print(f"  {name:<18} ${c:>10.2f}  (vs oracle: {rel:.4f}x)")
    print()
    print(f"  {'bias':>8} {'total':>10} {'cr_oracle':>10}")
    for e in bias_entries:
        print(f"  {e['bias_factor']:>8.3f} ${e['total_cost']:>9.2f} "
              f"{e['relative_cost_vs_oracle']:>10.4f}")


if __name__ == "__main__":
    main()
