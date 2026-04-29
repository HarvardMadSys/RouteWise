#!/usr/bin/env python3
"""DEPRECATED Phase 3 Simulation: Evaluate Online Latency-Aware Routing.

This script replays Phase 1 latency profiling data to evaluate the LP-Mix
routing strategy against baselines. It is preserved as an early LP-CDF / LP-Mix
iteration and is not the current paper-facing evaluation path.

Key features:
- Time causality: Only use data with timestamp < t for decisions at time t
- Multiple baselines: Single-Best, Cheapest, Random, LP-Mix
- Configurable SLOs and ablation parameters

Usage:
    python -m experiments.latency_phase3.experiment --slo 3.0 --output results/latency_phase3/
    python -m experiments.latency_phase3.experiment --ablation  # Run full ablation study
"""

import argparse

import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rwsim.policies.latency_routers.online_lp import (
    FailureMode,
    OnlineLatencyRouter,
    ProviderProfile,
    load_pricing,
    pre_filter,
    solve_lp_with_fallback,
)


@dataclass
class SimulationConfig:
    """Configuration for a single simulation run."""

    slo_sec: float = 3.0
    prior_strength: float = 10.0
    failure_mode: FailureMode = FailureMode.INFINITY
    kappa: float = 0.0
    weight_smoothing: float = 0.3
    warmup_fraction: float = 0.1  # Use first 10% as warmup (profile building)


@dataclass
class SimulationResult:
    """Results from a simulation run."""

    policy: str
    config: SimulationConfig
    total_requests: int
    successful_requests: int
    total_cost: float
    latencies_ms: list[float]  # Successful requests only
    slo_violations: int  # Including failures
    errors: int
    routing_log: list[dict]  # Per-request routing decisions

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
    def effective_cost(self) -> float:
        """Return effective cost per successful request."""
        return self.total_cost / self.successful_requests if self.successful_requests > 0 else 0.0

    def get_percentile(self, p: float) -> float:
        """Get latency percentile (success only)."""
        if not self.latencies_ms:
            return float("inf")
        return np.percentile(self.latencies_ms, p)

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV export."""
        return {
            "policy": self.policy,
            "slo_sec": self.config.slo_sec,
            "prior_strength": self.config.prior_strength,
            "failure_mode": self.config.failure_mode.value,
            "kappa": self.config.kappa,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "error_rate": self.error_rate,
            "slo_violation_rate": self.slo_violation_rate,
            "avg_cost": self.avg_cost,
            "effective_cost": self.effective_cost,
            "p50_ms": self.get_percentile(50),
            "p90_ms": self.get_percentile(90),
            "p99_ms": self.get_percentile(99),
        }


class LatencySimulator:
    """Simulator that replays Phase 1 data for evaluation.

    Maintains provider profiles with time causality:
    - At decision time t, only use probe samples with timestamp < t
    - Uses nearest-neighbor interpolation to sample latency for evaluation
    """

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
        self._provider_indices: dict[str, np.ndarray] = {}
        self._provider_timestamps: dict[str, np.ndarray] = {}
        self._provider_latencies: dict[str, np.ndarray] = {}
        self._provider_statuses: dict[str, np.ndarray] = {}

        for provider in self.providers:
            mask = latency_df["provider"] == provider
            provider_df = latency_df[mask].sort_values("timestamp")
            self._provider_indices[provider] = provider_df.index.values
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
            (ttft_ms, error_type) where error_type is None for success.
        """
        if provider not in self._provider_timestamps:
            return -1.0, "server_error"

        timestamps = self._provider_timestamps[provider]
        latencies = self._provider_latencies[provider]
        statuses = self._provider_statuses[provider]

        # Find nearest timestamp
        idx = np.searchsorted(timestamps, request_time)

        # Use nearest neighbor (prefer future sample to simulate real response)
        if idx >= len(timestamps):
            idx = len(timestamps) - 1
        elif idx > 0 and (request_time - timestamps[idx - 1]) < (timestamps[idx] - request_time):
            # Use closer timestamp
            idx = idx - 1

        ttft_ms = latencies[idx]
        status = statuses[idx]

        if status != "success":
            return -1.0, "server_error"

        return ttft_ms, None

    def build_profiles_until(
        self,
        cutoff_time: float,
    ) -> dict[str, ProviderProfile]:
        """Build provider profiles using only data before cutoff_time.

        Uses cached data arrays for efficiency.

        Args:
            cutoff_time: Only use samples with timestamp < cutoff_time.

        Returns:
            Dict of provider -> ProviderProfile.
        """
        profiles = {}

        for provider in self.providers:
            profile = ProviderProfile(
                provider=provider,
                prior_strength=self.config.prior_strength,
                failure_mode=self.config.failure_mode,
            )

            # Use cached arrays for efficiency
            timestamps = self._provider_timestamps[provider]
            latencies = self._provider_latencies[provider]
            statuses = self._provider_statuses[provider]

            # Find samples before cutoff using binary search
            idx = np.searchsorted(timestamps, cutoff_time)

            # Add samples (vectorized where possible)
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
        """Get list of timestamps to use for evaluation.

        Uses actual request times from the data, skipping warmup period.
        Subsamples to max_points if there are too many unique timestamps.

        Args:
            max_points: Maximum number of evaluation points.

        Returns:
            List of timestamps for evaluation.
        """
        min_t, max_t = self.get_time_range()
        warmup_end = min_t + (max_t - min_t) * self.config.warmup_fraction

        # Use unique timestamps after warmup
        eval_times = sorted(
            self.latency_df[self.latency_df["timestamp"] >= warmup_end]["timestamp"].unique()
        )

        # Subsample if too many
        if len(eval_times) > max_points:
            indices = np.linspace(0, len(eval_times) - 1, max_points, dtype=int)
            eval_times = [eval_times[i] for i in indices]

        return eval_times


def run_policy_simulation(
    simulator: LatencySimulator,
    policy: str,
    config: SimulationConfig,
    eval_times: list[float] | None = None,
    show_progress: bool = True,
) -> SimulationResult:
    """Run simulation for a single policy.

    Uses incremental profile building for efficiency.

    Args:
        simulator: Configured simulator instance.
        policy: One of "single_best", "cheapest", "random", "lp_mix".
        config: Simulation configuration.
        eval_times: List of timestamps to evaluate at. If None, uses all data points.
        show_progress: Whether to show progress updates.

    Returns:
        SimulationResult with metrics.
    """
    if eval_times is None:
        eval_times = simulator.get_evaluation_times()

    # Build profiles incrementally - start with data before first eval time
    profiles = simulator.build_profiles_until(eval_times[0])

    # Track which data points have been added to profiles
    profile_indices = {p: 0 for p in simulator.providers}
    for provider in simulator.providers:
        timestamps = simulator._provider_timestamps[provider]
        profile_indices[provider] = np.searchsorted(timestamps, eval_times[0])

    # Initialize router for LP-Mix
    router = None
    if policy == "lp_mix":
        router = OnlineLatencyRouter(
            costs=simulator.pricing,
            slo_sec=config.slo_sec,
            prior_strength=config.prior_strength,
            failure_mode=config.failure_mode,
            kappa=config.kappa,
            weight_smoothing=config.weight_smoothing,
        )
        # Set initial profiles
        for p, profile in profiles.items():
            router.profiles[p] = profile

    # Results tracking
    latencies_ms = []
    total_cost = 0.0
    slo_violations = 0
    errors = 0
    routing_log = []

    for idx, t in enumerate(eval_times):
        # Incrementally update profiles with samples between last_profile_time and t
        for provider in simulator.providers:
            timestamps = simulator._provider_timestamps[provider]
            latencies = simulator._provider_latencies[provider]
            statuses = simulator._provider_statuses[provider]

            # Find new samples to add
            start_idx = profile_indices[provider]
            end_idx = np.searchsorted(timestamps, t)

            for i in range(start_idx, end_idx):
                error_type = None if statuses[i] == "success" else "server_error"
                ttft_ms = latencies[i] if statuses[i] == "success" else -1.0
                profiles[provider].add_sample(timestamps[i], ttft_ms, error_type)

            profile_indices[provider] = end_idx

        # Select provider based on policy
        if policy == "single_best":
            # Use provider with lowest P99 in short window
            p99s = {p: profiles[p].get_p99(t) for p in simulator.providers}
            selected = min(p99s, key=lambda p: p99s[p])

        elif policy == "cheapest":
            # Use cheapest provider
            selected = min(simulator.pricing, key=lambda p: simulator.pricing[p])

        elif policy == "random":
            # Uniform random among eligible providers
            eligible = pre_filter(profiles, t)
            if not eligible:
                eligible = simulator.providers
            selected = np.random.choice(eligible)

        elif policy == "lp_mix":
            # Update router profiles
            for p, profile in profiles.items():
                router.profiles[p] = profile

            # Get routing decision (LP update is throttled inside router.route()).
            selected = router.route(t)

            if selected is None:
                selected = np.random.choice(simulator.providers)

        else:
            raise ValueError(f"Unknown policy: {policy}")

        # Sample latency from selected provider
        ttft_ms, error_type = simulator.sample_latency(selected, t)

        # Record results
        cost = simulator.pricing.get(selected, 0.0)
        total_cost += cost

        if error_type is not None:
            errors += 1
            slo_violations += 1  # Errors are SLO violations
        else:
            latencies_ms.append(ttft_ms)
            if ttft_ms / 1000.0 > config.slo_sec:
                slo_violations += 1

        # Log routing decision
        log_entry = {
            "timestamp": t,
            "provider": selected,
            "ttft_ms": ttft_ms,
            "error": error_type,
            "cost": cost,
        }
        if policy == "lp_mix" and router is not None:
            log_entry["lp_weights"] = router.last_lp_weights.copy()
            log_entry["lp_status"] = router.last_lp_status
        routing_log.append(log_entry)

        # Progress update
        if show_progress and (idx + 1) % 100 == 0:
            print(f"    Progress: {idx + 1}/{len(eval_times)}")

    return SimulationResult(
        policy=policy,
        config=config,
        total_requests=len(eval_times),
        successful_requests=len(latencies_ms),
        total_cost=total_cost,
        latencies_ms=latencies_ms,
        slo_violations=slo_violations,
        errors=errors,
        routing_log=routing_log,
    )


def run_ablation_study(
    simulator: LatencySimulator,
    output_dir: str,
    max_eval_points: int = 500,
) -> pd.DataFrame:
    """Run full ablation study across all parameter combinations.

    Args:
        simulator: Configured simulator instance.
        output_dir: Directory to save results.

    Returns:
        DataFrame with all results.
    """
    # Ablation parameters
    slo_values = [1.0, 2.0, 3.0, 5.0, 10.0]
    prior_strengths = [10, 20, 50]
    failure_modes = [FailureMode.INFINITY, FailureMode.SEPARATE]
    kappa_values = [0.0, 5.0, 10.0]
    policies = ["single_best", "cheapest", "random", "lp_mix"]

    results = []
    total_runs = len(slo_values) * len(policies)

    # For baselines, only vary SLO
    baseline_policies = ["single_best", "cheapest", "random"]
    run_count = 0

    # Get evaluation times once
    eval_times = simulator.get_evaluation_times(max_points=max_eval_points)
    print(f"Evaluation points: {len(eval_times)}")

    for slo in slo_values:
        print(f"\n=== SLO = {slo}s ===")

        # Run baselines (no ablation parameters)
        for policy in baseline_policies:
            run_count += 1
            print(f"  [{run_count}/{total_runs}] Running {policy}...")

            config = SimulationConfig(slo_sec=slo)
            result = run_policy_simulation(simulator, policy, config, eval_times)
            results.append(result.to_dict())

            print(f"    Cost: ${result.avg_cost:.6f}, SLO Viol: {result.slo_violation_rate:.2%}")

        # Run LP-Mix with ablations
        for prior in prior_strengths:
            for fm in failure_modes:
                for kappa in kappa_values:
                    run_count += 1
                    print(f"  [{run_count}] LP-Mix: prior={prior}, fm={fm.value}, kappa={kappa}...")

                    config = SimulationConfig(
                        slo_sec=slo,
                        prior_strength=prior,
                        failure_mode=fm,
                        kappa=kappa,
                    )
                    result = run_policy_simulation(simulator, "lp_mix", config, eval_times)
                    results.append(result.to_dict())

                    print(
                        f"    Cost: ${result.avg_cost:.6f}, SLO Viol: {result.slo_violation_rate:.2%}"
                    )

    # Save results
    df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, "ablation_results.csv")
    df.to_csv(output_file, index=False)
    print(f"\nSaved ablation results to {output_file}")

    return df


def run_single_simulation(
    simulator: LatencySimulator,
    slo: float,
    output_dir: str,
    max_eval_points: int = 500,
) -> pd.DataFrame:
    """Run simulation for a single SLO value.

    Args:
        simulator: Configured simulator instance.
        slo: SLO in seconds.
        output_dir: Directory to save results.
        max_eval_points: Maximum number of evaluation points.

    Returns:
        DataFrame with results.
    """
    policies = ["single_best", "cheapest", "random", "lp_mix"]
    config = SimulationConfig(slo_sec=slo)

    eval_times = simulator.get_evaluation_times(max_points=max_eval_points)
    print(f"Evaluation points: {len(eval_times)}")

    results = []
    for policy in policies:
        print(f"Running {policy}...")
        result = run_policy_simulation(simulator, policy, config, eval_times)
        results.append(result.to_dict())

        print(f"  Cost: ${result.avg_cost:.6f}")
        print(f"  SLO Violation: {result.slo_violation_rate:.2%}")
        print(
            f"  P50/P90/P99: {result.get_percentile(50):.1f} / {result.get_percentile(90):.1f} / {result.get_percentile(99):.1f} ms"
        )

        # Save routing log
        log_file = os.path.join(output_dir, f"routing_log_{policy}_slo{slo}.json")
        with open(log_file, "w") as f:
            json.dump(result.routing_log, f, indent=2)

    df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, f"simulation_results_slo{slo}.csv")
    df.to_csv(output_file, index=False)
    print(f"\nSaved results to {output_file}")

    return df


def main():
    """Run Phase 3 latency routing simulation."""
    parser = argparse.ArgumentParser(description="Phase 3 Latency Routing Simulation")
    parser.add_argument(
        "--data",
        type=str,
        default="ICML2026_HybridInference/data/latency_llama70b_24h.csv",
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
        default="experiment/results/latency_phase3",
        help="Output directory",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run full ablation study",
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
    # Handle both CSV and Git LFS paths
    data_path = args.data
    if not os.path.exists(data_path):
        # Try alternative path
        alt_path = os.path.join("ICML2026_HybridInference", "data", "latency_llama70b_24h.csv")
        if os.path.exists(alt_path):
            data_path = alt_path
        else:
            print(f"Error: Data file not found at {args.data}")
            print("Please ensure the Phase 1 data is available.")
            sys.exit(1)

    latency_df = pd.read_csv(data_path)
    print(f"  Loaded {len(latency_df)} records")

    # Filter by workload if specified
    if args.workload != "all":
        # Extract workload type from request_id (e.g., "Cerebras-base-R0-0" -> "base")
        latency_df["workload"] = latency_df["request_id"].apply(
            lambda x: x.split("-")[1] if "-" in x else "unknown"
        )
        latency_df = latency_df[latency_df["workload"] == args.workload]
        print(f"  Filtered to '{args.workload}' workload: {len(latency_df)} records")

    print(f"  Providers: {sorted(latency_df['provider'].unique())}")
    print(
        f"  Time range: {latency_df['timestamp'].min():.0f} - {latency_df['timestamp'].max():.0f}"
    )

    # Load pricing
    print(f"\nLoading pricing from {args.pricing}...")
    if not os.path.exists(args.pricing):
        print(f"Error: Pricing file not found at {args.pricing}")
        sys.exit(1)

    pricing = load_pricing(args.pricing)
    print(f"  Loaded pricing for {len(pricing)} providers")

    # Initialize simulator
    config = SimulationConfig(slo_sec=args.slo)
    simulator = LatencySimulator(latency_df, pricing, config)

    # Run simulation
    if args.ablation:
        print("\n=== Running Ablation Study ===")
        run_ablation_study(simulator, args.output, args.max_eval_points)
    else:
        print(f"\n=== Running Simulation with SLO = {args.slo}s ===")
        run_single_simulation(simulator, args.slo, args.output, args.max_eval_points)

    print("\nDone!")


if __name__ == "__main__":
    main()
