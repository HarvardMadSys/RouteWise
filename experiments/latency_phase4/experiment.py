#!/usr/bin/env python3
"""Phase 4 Simulation: Evaluate Smart Hedging Strategies.

This script replays Phase 1 latency profiling data to evaluate hedging strategies
built on top of Phase 3 LP-Mix routing.

Key features:
- Time causality: Only use data with timestamp < t for decisions at time t
- Multiple hedging strategies: never, always, fixed-timeout, smart-survival, smart-residual
- Backup selection methods: fastest, lp_other, cheapest_viable
- Cost modeling with full billing (no refunds)

Usage:
    python -m experiments.latency_phase4.experiment --slo 3.0 --output results/latency_phase4/
    python -m experiments.latency_phase4.experiment --ablation  # Run full ablation study
"""

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rwsim.policies.hedgers.smart_economic import (
    BackupSelectionMethod,
    HedgingParams,
    HedgingResult,
    HedgingStrategy,
    SmartHedger,
    select_backup,
)
from rwsim.policies.latency_routers.online_lp import (
    FailureMode,
    OnlineLatencyRouter,
    ProviderProfile,
    load_pricing,
)


@dataclass
class SimulationConfig:
    """Configuration for a single simulation run."""

    slo_sec: float = 3.0
    failure_mode: FailureMode = FailureMode.INFINITY
    kappa: float = 0.0
    weight_smoothing: float = 0.3
    warmup_fraction: float = 0.1  # Use first 10% as warmup


@dataclass
class HedgingSimulationResult:
    """Results from a hedging simulation run."""

    strategy: str
    backup_method: str
    config: SimulationConfig
    hedging_params: HedgingParams
    total_requests: int
    successful_requests: int
    total_cost: float
    latencies_ms: list[float]  # Successful requests only
    slo_violations: int  # Including failures
    errors: int
    hedge_count: int  # Number of requests that triggered hedging
    routing_log: list[dict]

    @property
    def error_rate(self) -> float:
        """Return error rate as fraction of total requests."""
        return self.errors / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def slo_violation_rate(self) -> float:
        """Return SLO violation rate as fraction of total requests."""
        return self.slo_violations / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def avg_cost(self) -> float:
        """Return average cost per request."""
        return self.total_cost / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def hedge_rate(self) -> float:
        """Return fraction of requests that triggered hedging."""
        return self.hedge_count / self.total_requests if self.total_requests > 0 else 0.0

    def get_percentile(self, p: float) -> float:
        """Get latency percentile (success only)."""
        if not self.latencies_ms:
            return float("inf")
        return float(np.percentile(self.latencies_ms, p))

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV export."""
        return {
            "strategy": self.strategy,
            "backup_method": self.backup_method,
            "slo_sec": self.config.slo_sec,
            "alpha": self.hedging_params.alpha,
            "cost_ratio": self.hedging_params.cost_ratio,
            "dispatch_overhead_sec": self.hedging_params.dispatch_overhead_sec,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "error_rate": self.error_rate,
            "slo_violation_rate": self.slo_violation_rate,
            "avg_cost": self.avg_cost,
            "hedge_rate": self.hedge_rate,
            "p50_ms": self.get_percentile(50),
            "p90_ms": self.get_percentile(90),
            "p99_ms": self.get_percentile(99),
        }


class HedgingSimulator:
    """Simulator for hedging strategies on Phase 1 data."""

    def __init__(
        self,
        latency_df: pd.DataFrame,
        pricing: dict[str, float],
        config: SimulationConfig,
    ):
        """Initialize simulator.

        Args:
            latency_df: Phase 1 latency data with columns:
                        [provider, timestamp, ttft_ms, status]
            pricing: Provider -> cost per request.
            config: Simulation configuration.
        """
        self.latency_df = latency_df.sort_values("timestamp").reset_index(drop=True)
        self.pricing = pricing
        self.config = config

        # Get list of providers
        self.providers = sorted(latency_df["provider"].unique().tolist())

        # Build per-provider sorted indices for nearest-neighbor lookup
        self._provider_timestamps: dict[str, np.ndarray] = {}
        self._provider_latencies: dict[str, np.ndarray] = {}
        self._provider_statuses: dict[str, np.ndarray] = {}

        for provider in self.providers:
            mask = latency_df["provider"] == provider
            provider_df = latency_df[mask].sort_values("timestamp")
            self._provider_timestamps[provider] = provider_df["timestamp"].values
            self._provider_latencies[provider] = provider_df["ttft_ms"].values
            self._provider_statuses[provider] = provider_df["status"].values

    def sample_latency(
        self,
        provider: str,
        request_time: float,
    ) -> tuple[float, str | None]:
        """Sample latency for a provider at a given time.

        Uses nearest-neighbor interpolation from actual data.

        Args:
            provider: Provider name.
            request_time: Request timestamp.

        Returns:
            (ttft_sec, error_type) where error_type is None for success.
        """
        if provider not in self._provider_timestamps:
            return 30.0, "server_error"  # Default timeout

        timestamps = self._provider_timestamps[provider]
        latencies = self._provider_latencies[provider]
        statuses = self._provider_statuses[provider]

        # Find nearest timestamp
        idx = np.searchsorted(timestamps, request_time)

        # Use nearest neighbor
        if idx >= len(timestamps):
            idx = len(timestamps) - 1
        elif idx > 0 and (request_time - timestamps[idx - 1]) < (timestamps[idx] - request_time):
            idx = idx - 1

        ttft_ms = latencies[idx]
        status = statuses[idx]

        if status != "success":
            return 30.0, "server_error"  # Return timeout for failed requests

        return ttft_ms / 1000.0, None  # Convert to seconds

    def build_profiles_until(
        self,
        cutoff_time: float,
    ) -> dict[str, ProviderProfile]:
        """Build provider profiles using only data before cutoff_time."""
        profiles = {}

        for provider in self.providers:
            profile = ProviderProfile(
                provider=provider,
                failure_mode=self.config.failure_mode,
            )

            timestamps = self._provider_timestamps[provider]
            latencies = self._provider_latencies[provider]
            statuses = self._provider_statuses[provider]

            idx = np.searchsorted(timestamps, cutoff_time)

            for i in range(idx):
                error_type = None if statuses[i] == "success" else "server_error"
                ttft_ms = latencies[i] if statuses[i] == "success" else -1.0
                profile.add_sample(timestamps[i], ttft_ms, error_type)

            profiles[provider] = profile

        return profiles

    def get_time_range(self) -> tuple[float, float]:
        """Return (min_timestamp, max_timestamp) of the data."""
        return self.latency_df["timestamp"].min(), self.latency_df["timestamp"].max()

    def get_evaluation_times(self, max_points: int = 500) -> list[float]:
        """Get list of timestamps to use for evaluation."""
        min_t, max_t = self.get_time_range()
        warmup_end = min_t + (max_t - min_t) * self.config.warmup_fraction

        eval_times = sorted(
            self.latency_df[self.latency_df["timestamp"] >= warmup_end]["timestamp"].unique()
        )

        if len(eval_times) > max_points:
            indices = np.linspace(0, len(eval_times) - 1, max_points, dtype=int)
            eval_times = [eval_times[i] for i in indices]

        return eval_times


def run_hedging_simulation(
    simulator: HedgingSimulator,
    hedging_params: HedgingParams,
    config: SimulationConfig,
    eval_times: list[float] | None = None,
    show_progress: bool = True,
) -> HedgingSimulationResult:
    """Run simulation for a hedging strategy.

    Args:
        simulator: Configured simulator instance.
        hedging_params: Hedging configuration.
        config: Simulation configuration.
        eval_times: List of timestamps to evaluate at.
        show_progress: Whether to show progress updates.

    Returns:
        HedgingSimulationResult with metrics.
    """
    if eval_times is None:
        eval_times = simulator.get_evaluation_times()

    # Build initial profiles
    profiles = simulator.build_profiles_until(eval_times[0])

    # Track profile indices for incremental updates
    profile_indices = dict.fromkeys(simulator.providers, 0)
    for provider in simulator.providers:
        timestamps = simulator._provider_timestamps[provider]
        profile_indices[provider] = np.searchsorted(timestamps, eval_times[0])

    # Initialize Phase 3 router for LP-Mix primary selection
    router = OnlineLatencyRouter(
        costs=simulator.pricing,
        slo_sec=config.slo_sec,
        failure_mode=config.failure_mode,
        kappa=config.kappa,
        weight_smoothing=config.weight_smoothing,
    )
    for p, profile in profiles.items():
        router.profiles[p] = profile

    # Initialize hedger
    hedger = SmartHedger(hedging_params, simulator.pricing)

    # Results tracking
    latencies_ms = []
    total_cost = 0.0
    slo_violations = 0
    errors = 0
    hedge_count = 0
    routing_log = []

    for idx, t in enumerate(eval_times):
        # Incrementally update profiles
        for provider in simulator.providers:
            timestamps = simulator._provider_timestamps[provider]
            latencies = simulator._provider_latencies[provider]
            statuses = simulator._provider_statuses[provider]

            start_idx = profile_indices[provider]
            end_idx = np.searchsorted(timestamps, t)

            for i in range(start_idx, end_idx):
                error_type = None if statuses[i] == "success" else "server_error"
                ttft_ms = latencies[i] if statuses[i] == "success" else -1.0
                profiles[provider].add_sample(timestamps[i], ttft_ms, error_type)

            profile_indices[provider] = end_idx

        # Update router profiles
        for p, profile in profiles.items():
            router.profiles[p] = profile

        # Select primary provider using LP-Mix
        primary = router.route(t)
        if primary is None:
            primary = np.random.choice(simulator.providers)

        # Get current LP weights for backup selection
        lp_weights = router.last_lp_weights if router.last_lp_weights else None

        # Select backup provider
        backup = select_backup(
            method=hedging_params.backup_method,
            profiles=profiles,
            costs=simulator.pricing,
            primary=primary,
            slo_sec=hedging_params.slo_sec,
            now=t,
            lp_weights=lp_weights,
            min_viable_cdf=hedging_params.min_viable_cdf,
        )

        # Fallback if no backup found
        if backup is None:
            candidates = [p for p in simulator.providers if p != primary]
            backup = candidates[0] if candidates else primary

        # Sample latencies for both providers
        T_primary_sec, err_primary = simulator.sample_latency(primary, t)
        T_backup_sec, err_backup = simulator.sample_latency(backup, t)

        # Simulate hedged request
        result = hedger.simulate_request(
            primary=primary,
            profiles=profiles,
            now=t,
            T_primary_sec=T_primary_sec,
            err_primary=err_primary,
            T_backup_sec=T_backup_sec,
            err_backup=err_backup,
            backup=backup,
        )

        # Record results
        total_cost += result.total_cost

        if result.hedged:
            hedge_count += 1

        # Determine final status
        final_error = result.primary_error if result.winner == "primary" else result.backup_error

        if final_error is not None:
            errors += 1
            slo_violations += 1
        else:
            latencies_ms.append(result.final_ttft_sec * 1000)
            if result.final_ttft_sec > config.slo_sec:
                slo_violations += 1

        # Log routing decision
        log_entry = {
            "timestamp": t,
            "primary": primary,
            "backup": backup,
            "hedged": result.hedged,
            "hedge_time_sec": result.hedge_time_sec,
            "winner": result.winner,
            "final_ttft_ms": result.final_ttft_sec * 1000,
            "primary_cost": result.primary_cost,
            "backup_cost": result.backup_cost,
            "total_cost": result.total_cost,
            "slo_violated": result.final_ttft_sec > config.slo_sec or final_error is not None,
        }
        routing_log.append(log_entry)

        # Progress update
        if show_progress and (idx + 1) % 100 == 0:
            print(
                f"  [{hedging_params.strategy.value}] {idx + 1}/{len(eval_times)} "
                f"hedge_rate={hedge_count / (idx + 1):.2%}"
            )

    return HedgingSimulationResult(
        strategy=hedging_params.strategy.value,
        backup_method=hedging_params.backup_method.value,
        config=config,
        hedging_params=hedging_params,
        total_requests=len(eval_times),
        successful_requests=len(latencies_ms),
        total_cost=total_cost,
        latencies_ms=latencies_ms,
        slo_violations=slo_violations,
        errors=errors,
        hedge_count=hedge_count,
        routing_log=routing_log,
    )


def run_ablation_study(
    simulator: HedgingSimulator,
    config: SimulationConfig,
    eval_times: list[float],
    output_dir: str,
) -> pd.DataFrame:
    """Run full ablation study across strategies and parameters.

    Args:
        simulator: Configured simulator instance.
        config: Base simulation configuration.
        eval_times: Evaluation timestamps.
        output_dir: Directory for output files.

    Returns:
        DataFrame with all results.
    """
    results = []

    # Strategy configurations: (strategy, alpha, cost_ratio, label_suffix)
    strategies = [
        (HedgingStrategy.NEVER, None, None, ""),
        (HedgingStrategy.ALWAYS, None, None, ""),
        (HedgingStrategy.FIXED_TIMEOUT, 0.5, None, "_0.5"),
        (HedgingStrategy.FIXED_TIMEOUT, 0.7, None, "_0.7"),
        (HedgingStrategy.FIXED_TIMEOUT, 0.9, None, "_0.9"),
        (HedgingStrategy.SMART_SURVIVAL, None, None, ""),
        (HedgingStrategy.SMART_RESIDUAL, None, None, ""),
        (HedgingStrategy.PERCENTILE_BASED, None, None, ""),
        (HedgingStrategy.SMART_ECONOMIC, None, 0.01, "_cr0.01"),
        (HedgingStrategy.SMART_ECONOMIC, None, 0.05, "_cr0.05"),
        (HedgingStrategy.SMART_ECONOMIC, None, 0.1, "_cr0.1"),
        (HedgingStrategy.SMART_ECONOMIC, None, 0.2, "_cr0.2"),
        (HedgingStrategy.SMART_ECONOMIC, None, 0.5, "_cr0.5"),
    ]

    backup_methods = [
        BackupSelectionMethod.FASTEST,
        BackupSelectionMethod.LP_OTHER,
        BackupSelectionMethod.CHEAPEST_VIABLE,
    ]

    total_runs = len(strategies) * len(backup_methods)
    current_run = 0

    for strategy, alpha, cost_ratio, label_suffix in strategies:
        for backup_method in backup_methods:
            current_run += 1
            alpha_val = alpha if alpha else 0.7
            cost_ratio_val = cost_ratio if cost_ratio else 0.1
            strategy_label = strategy.value + label_suffix

            print(
                f"\n[{current_run}/{total_runs}] Running: {strategy_label} + {backup_method.value}"
            )

            params = HedgingParams(
                strategy=strategy,
                slo_sec=config.slo_sec,
                alpha=alpha_val,
                cost_ratio=cost_ratio_val,
                dispatch_overhead_sec=0.05,
                backup_method=backup_method,
            )

            result = run_hedging_simulation(
                simulator=simulator,
                hedging_params=params,
                config=config,
                eval_times=eval_times,
                show_progress=True,
            )

            # Get dict and override strategy name with suffix
            result_dict = result.to_dict()
            result_dict["strategy"] = strategy_label
            results.append(result_dict)

            print(
                f"  -> SLO violation: {result.slo_violation_rate:.2%}, "
                f"Hedge rate: {result.hedge_rate:.2%}, "
                f"Avg cost: {result.avg_cost:.6f}"
            )

    df = pd.DataFrame(results)

    # Save results
    output_file = os.path.join(output_dir, f"ablation_slo{config.slo_sec}.csv")
    df.to_csv(output_file, index=False)
    print(f"\nSaved ablation results to {output_file}")

    return df


def run_single_strategy(
    simulator: HedgingSimulator,
    config: SimulationConfig,
    eval_times: list[float],
    strategy: HedgingStrategy,
    backup_method: BackupSelectionMethod,
    alpha: float,
    output_dir: str,
) -> pd.DataFrame:
    """Run single strategy evaluation."""
    print(f"\nRunning: {strategy.value} + {backup_method.value} (alpha={alpha})")

    params = HedgingParams(
        strategy=strategy,
        slo_sec=config.slo_sec,
        alpha=alpha,
        dispatch_overhead_sec=0.05,
        backup_method=backup_method,
    )

    result = run_hedging_simulation(
        simulator=simulator,
        hedging_params=params,
        config=config,
        eval_times=eval_times,
        show_progress=True,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy.value}")
    print(f"Backup method: {backup_method.value}")
    print(f"SLO: {config.slo_sec}s")
    print(f"{'='*60}")
    print(f"Total requests: {result.total_requests}")
    print(f"Successful requests: {result.successful_requests}")
    print(f"Error rate: {result.error_rate:.2%}")
    print(f"SLO violation rate: {result.slo_violation_rate:.2%}")
    print(f"Hedge rate: {result.hedge_rate:.2%}")
    print(f"Average cost: {result.avg_cost:.6f}")
    print(f"P50 latency: {result.get_percentile(50):.2f} ms")
    print(f"P90 latency: {result.get_percentile(90):.2f} ms")
    print(f"P99 latency: {result.get_percentile(99):.2f} ms")

    df = pd.DataFrame([result.to_dict()])
    output_file = os.path.join(
        output_dir, f"result_{strategy.value}_{backup_method.value}_slo{config.slo_sec}.csv"
    )
    df.to_csv(output_file, index=False)
    print(f"\nSaved results to {output_file}")

    return df


def main():
    """Run Phase 4 hedging simulation."""
    parser = argparse.ArgumentParser(description="Phase 4 Smart Hedging Simulation")
    parser.add_argument(
        "--data",
        type=str,
        default="experiment/data/data/latency_llama70b_24h.csv",
        help="Path to Phase 1 latency data CSV",
    )
    parser.add_argument(
        "--pricing",
        type=str,
        default="experiment/data/openrouter_llama33_70b.json",
        help="Path to pricing JSON file",
    )
    parser.add_argument(
        "--slo",
        type=float,
        default=3.0,
        help="SLO in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiment/results/latency_phase4",
        help="Output directory",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run full ablation study",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="smart_survival",
        choices=[
            "never",
            "always",
            "fixed_timeout",
            "smart_survival",
            "smart_residual",
            "smart_economic",
        ],
        help="Hedging strategy (default: smart_survival)",
    )
    parser.add_argument(
        "--backup-method",
        type=str,
        default="fastest",
        choices=["fastest", "lp_other", "cheapest_viable"],
        help="Backup selection method (default: fastest)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Alpha for fixed-timeout strategy (default: 0.7)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--workload",
        type=str,
        default="base",
        help="Workload filter (base, prefill, decode, or all)",
    )
    parser.add_argument(
        "--max-eval-points",
        type=int,
        default=500,
        help="Maximum number of evaluation points (default: 500)",
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load data
    print(f"Loading latency data from {args.data}...")
    data_path = args.data
    if not os.path.exists(data_path):
        alt_path = os.path.join("experiment", "data", "data", "latency_llama70b_24h.csv")
        if os.path.exists(alt_path):
            data_path = alt_path
        else:
            print(f"Error: Data file not found at {args.data}")
            sys.exit(1)

    latency_df = pd.read_csv(data_path)
    print(f"  Loaded {len(latency_df)} records")

    # Filter by workload
    if args.workload != "all":
        latency_df["workload"] = latency_df["request_id"].apply(
            lambda x: x.split("-")[1] if "-" in x else "unknown"
        )
        latency_df = latency_df[latency_df["workload"] == args.workload]
        print(f"  Filtered to '{args.workload}' workload: {len(latency_df)} records")

    print(f"  Providers: {sorted(latency_df['provider'].unique())}")

    # Load pricing
    print(f"\nLoading pricing from {args.pricing}...")
    if not os.path.exists(args.pricing):
        print(f"Error: Pricing file not found at {args.pricing}")
        sys.exit(1)

    pricing = load_pricing(args.pricing)
    print(f"  Loaded pricing for {len(pricing)} providers")

    # Filter to common providers
    data_providers = set(latency_df["provider"].unique())
    pricing_providers = set(pricing.keys())
    common_providers = data_providers & pricing_providers

    if not common_providers:
        print("Error: No common providers between data and pricing")
        sys.exit(1)

    latency_df = latency_df[latency_df["provider"].isin(common_providers)]
    pricing = {p: v for p, v in pricing.items() if p in common_providers}
    print(f"  Common providers: {sorted(common_providers)}")

    # Create config and simulator
    config = SimulationConfig(slo_sec=args.slo)
    simulator = HedgingSimulator(latency_df, pricing, config)

    # Get evaluation times
    eval_times = simulator.get_evaluation_times(args.max_eval_points)
    print(f"\nEvaluation points: {len(eval_times)}")

    # Run simulation
    if args.ablation:
        run_ablation_study(simulator, config, eval_times, args.output)
    else:
        strategy = HedgingStrategy(args.strategy)
        backup_method = BackupSelectionMethod(args.backup_method)
        run_single_strategy(
            simulator=simulator,
            config=config,
            eval_times=eval_times,
            strategy=strategy,
            backup_method=backup_method,
            alpha=args.alpha,
            output_dir=args.output,
        )


if __name__ == "__main__":
    main()
