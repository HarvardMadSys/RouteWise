"""Latency-aware offline routing under a tail-latency SLO.

This module solves a category-level linear program (LP) that minimizes expected
cost subject to a tail-latency constraint for a mixed routing policy:
    sum_j pi_j * F_j(L) >= target

It is intended for Phase 2/3-style offline analysis using empirical latency
profiles from probing logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

# Increase CSV field size limit
csv.field_size_limit(sys.maxsize)

LatencyMetric = Literal["ttft", "e2e"]


@dataclass(frozen=True)
class LatencyStats:
    """Summary statistics for a latency sample array (in seconds)."""

    p50: float
    p90: float
    p99: float
    mean: float
    n: int


def _compute_latency_stats(samples_ms: np.ndarray) -> LatencyStats:
    """Compute summary statistics from latency samples in milliseconds."""
    if samples_ms.size == 0:
        raise ValueError("samples_ms must be non-empty")
    samples_sec = samples_ms / 1000.0
    return LatencyStats(
        p50=float(np.percentile(samples_sec, 50)),
        p90=float(np.percentile(samples_sec, 90)),
        p99=float(np.percentile(samples_sec, 99)),
        mean=float(np.mean(samples_sec)),
        n=int(samples_sec.size),
    )


@dataclass
class LatencyProfile:
    """Empirical latency profile for a provider."""

    provider: str
    ttft_samples: np.ndarray  # Time to first token (ms)
    e2e_samples: np.ndarray  # End-to-end latency (ms)

    ttft_stats: LatencyStats = field(init=False)
    e2e_stats: LatencyStats = field(init=False)

    def __post_init__(self) -> None:
        self.ttft_stats = _compute_latency_stats(self.ttft_samples)
        self.e2e_stats = _compute_latency_stats(self.e2e_samples)

    @property
    def n_samples(self) -> int:
        """Return the number of latency samples."""
        return int(self.ttft_samples.size)

    def stats(self, metric: LatencyMetric) -> LatencyStats:
        """Return summary statistics for the selected latency metric."""
        if metric == "ttft":
            return self.ttft_stats
        if metric == "e2e":
            return self.e2e_stats
        raise ValueError(f"Unsupported metric: {metric}")

    def cdf_at(self, deadline_sec: float, metric: LatencyMetric) -> float:
        """Compute empirical CDF at deadline (seconds): P(T <= deadline)."""
        samples_ms = self.ttft_samples if metric == "ttft" else self.e2e_samples
        return float(np.mean(samples_ms / 1000.0 <= deadline_sec))

    def sample(self, rng: np.random.Generator, n: int, metric: LatencyMetric) -> np.ndarray:
        """Sample latencies (in seconds) from the empirical distribution."""
        if n <= 0:
            raise ValueError("n must be positive")
        samples_ms = self.ttft_samples if metric == "ttft" else self.e2e_samples
        indices = rng.integers(0, samples_ms.size, size=n)
        return samples_ms[indices] / 1000.0


@dataclass
class ProviderConfig:
    """Provider configuration with cost and latency profile."""

    name: str
    display_name: str
    input_price: float  # Cost in $ per 1M input tokens
    output_price: float  # Cost in $ per 1M output tokens
    profile: LatencyProfile

    def cost_per_request(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a request with given token counts."""
        return (input_tokens * self.input_price + output_tokens * self.output_price) / 1_000_000


def _parse_workload_from_request_id(request_id: str) -> str | None:
    """Parse workload name from request_id emitted by latency_profiling.py."""
    # Expected format: "{provider}-{workload}-R{round}-{i}"
    parts = request_id.split("-")
    if len(parts) < 2:
        return None
    return parts[1]


def load_latency_profiles(
    csv_path: str,
    metric: LatencyMetric,
    workloads: tuple[str, ...],
    min_samples: int,
) -> dict[str, LatencyProfile]:
    """Load latency profiles from Phase 1 profiling data.

    Expected CSV format:
        request_id,provider,status,ttft_ms,e2e_ms,input_len,output_len,timestamp,...

    Args:
        csv_path: Path to Phase 1 latency profiling CSV
        metric: Latency metric used for sample-count filtering.
        workloads: Workload names to include (e.g., ("base",)).
        min_samples: Minimum number of successful samples to include a provider.

    Returns:
        Dict mapping provider name to LatencyProfile
    """
    from collections import defaultdict

    provider_data = defaultdict(lambda: {"ttft": [], "e2e": []})

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "success":
                continue

            request_id = row.get("request_id", "")
            if request_id:
                workload = _parse_workload_from_request_id(request_id)
                if workload and workload not in workloads:
                    continue

            provider = row.get("provider", "")
            ttft_ms = row.get("ttft_ms", "")
            e2e_ms = row.get("e2e_ms", "")

            if not provider or not ttft_ms:
                continue

            try:
                ttft = float(ttft_ms)
                e2e = float(e2e_ms) if e2e_ms else ttft
                if ttft > 0:  # Filter invalid samples
                    provider_data[provider]["ttft"].append(ttft)
                    provider_data[provider]["e2e"].append(e2e)
            except (ValueError, TypeError):
                continue

    profiles = {}
    for provider, data in provider_data.items():
        selected = data["ttft"] if metric == "ttft" else data["e2e"]
        if len(selected) >= min_samples:
            profiles[provider] = LatencyProfile(
                provider=provider,
                ttft_samples=np.array(data["ttft"]),
                e2e_samples=np.array(data["e2e"]),
            )

    return profiles


def load_burstgpt_workload(csv_path: str, max_requests: int | None = None) -> list[dict]:
    """Load BurstGPT workload trace.

    Expected CSV format:
        Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type

    Args:
        csv_path: Path to BurstGPT CSV
        max_requests: Maximum number of requests to load (None for all)

    Returns:
        List of request dicts with timestamp and token counts
    """
    requests = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_requests and i >= max_requests:
                break

            try:
                timestamp = int(row.get("Timestamp", 0))
                input_tokens = int(row.get("Request tokens", 0))
                output_tokens = int(row.get("Response tokens", 0))

                if input_tokens > 0:
                    requests.append(
                        {
                            "id": i,
                            "timestamp": timestamp,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                        }
                    )
            except (ValueError, TypeError):
                continue

    return requests


def solve_lp_routing(
    providers: list[ProviderConfig],
    slo_sec: float,
    metric: LatencyMetric,
    avg_input_tokens: int = 50,
    avg_output_tokens: int = 10,
    target_percentile: float = 99.0,
) -> tuple[np.ndarray | None, float]:
    """Solve the LP for optimal routing given SLO constraint.

    minimize: sum(pi_j * c_j)  (expected cost)
    subject to:
        sum(pi_j * F_j(slo)) >= target_percentile/100  (SLO constraint)
        sum(pi_j) = 1
        pi_j >= 0

    Args:
        providers: List of provider configs with latency profiles
        slo_sec: SLO threshold in seconds (P99 target)
        avg_input_tokens: Average input tokens per request
        avg_output_tokens: Average output tokens per request
        target_percentile: Target percentile (default 99.0)

    Returns:
        Tuple of (routing_probs, expected_cost) or (None, inf) if infeasible
    """
    n = len(providers)

    # Objective: minimize cost (per request)
    c = np.array([p.cost_per_request(avg_input_tokens, avg_output_tokens) for p in providers])

    # Inequality constraint: sum(pi_j * F_j(slo)) >= target/100
    # Rewrite as: -sum(pi_j * F_j(slo)) <= -target/100
    cdf_values = []
    for p in providers:
        cdf_values.append(p.profile.cdf_at(slo_sec, metric))

    A_ub = np.array([[-v for v in cdf_values]])
    b_ub = np.array([-target_percentile / 100.0])

    # Equality constraint: sum(pi_j) = 1
    A_eq = np.ones((1, n))
    b_eq = np.array([1.0])

    # Bounds: 0 <= pi_j <= 1
    bounds = [(0, 1) for _ in range(n)]

    # Solve LP
    result = optimize.linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )

    if result.success:
        return result.x, result.fun
    else:
        return None, np.inf


def compute_pareto_frontier(
    providers: list[ProviderConfig],
    metric: LatencyMetric,
    slo_range: tuple[float, float] = (0.5, 30.0),
    n_points: int = 100,
    avg_input_tokens: int = 50,
    avg_output_tokens: int = 10,
) -> list[tuple[float, float, np.ndarray]]:
    """Compute Pareto frontier by sweeping over SLO values.

    Args:
        providers: List of provider configs
        slo_range: (min_slo, max_slo) in seconds
        n_points: Number of points to compute
        avg_input_tokens: Average input tokens per request
        avg_output_tokens: Average output tokens per request

    Returns:
        List of (slo, cost, routing_probs) tuples
    """
    slos = np.linspace(slo_range[0], slo_range[1], n_points)
    pareto = []

    for slo in slos:
        pi, cost = solve_lp_routing(providers, slo, metric, avg_input_tokens, avg_output_tokens)
        if pi is not None:
            pareto.append((slo, cost, pi))

    return pareto


def simulate_routing(
    requests: list[dict],
    providers: list[ProviderConfig],
    routing_probs: np.ndarray,
    rng: np.random.Generator,
    metric: LatencyMetric,
    slo_sec: float,
    use_trace_tokens: bool,
    default_input_tokens: int,
    default_output_tokens: int,
) -> dict:
    """Simulate routing with given probabilities and compute metrics.

    Args:
        requests: List of request dicts
        providers: List of provider configs
        routing_probs: Routing probability vector
        rng: Random number generator

    Returns:
        Dict with simulation results (cost, latencies, violations, etc.)
    """
    n = len(requests)
    total_cost = 0.0
    latencies = []
    provider_counts = {p.name: 0 for p in providers}

    # Assign each request to a provider based on routing probs
    provider_indices = rng.choice(len(providers), size=n, p=routing_probs)

    for i, req in enumerate(requests):
        provider = providers[provider_indices[i]]
        provider_counts[provider.name] += 1

        # Sample latency from provider's distribution
        latency = provider.profile.sample(rng, n=1, metric=metric)[0]

        latencies.append(latency)
        # Calculate cost based on request token counts
        if use_trace_tokens:
            input_tokens = int(req.get("input_tokens", default_input_tokens))
            output_tokens = int(req.get("output_tokens", default_output_tokens))
        else:
            input_tokens = default_input_tokens
            output_tokens = default_output_tokens
        total_cost += provider.cost_per_request(input_tokens, output_tokens)

    latencies = np.array(latencies)

    violation_rate = float(np.mean(latencies > slo_sec)) if n > 0 else 0.0

    return {
        "total_cost": total_cost,
        "cost_per_request": total_cost / n if n > 0 else 0,
        "latencies": latencies,
        "p50": np.percentile(latencies, 50),
        "p90": np.percentile(latencies, 90),
        "p99": np.percentile(latencies, 99),
        "violation_rate": violation_rate,
        "provider_counts": provider_counts,
    }


def print_provider_stats(profiles: dict[str, LatencyProfile], metric: LatencyMetric) -> None:
    """Print latency statistics for all providers."""
    print("\n" + "=" * 80)
    print(f"Provider Latency Statistics (metric={metric})")
    print("=" * 80)
    print(f"{'Provider':<15} {'N':>8} {'P50':>10} {'P90':>10} {'P99':>10} {'Mean':>10}")
    print("-" * 80)

    # Sort by P99 latency
    sorted_providers = sorted(profiles.items(), key=lambda x: x[1].stats(metric).p99)

    for name, profile in sorted_providers:
        stats = profile.stats(metric)
        print(
            f"{name:<15} {stats.n:>8} {stats.p50:>10.2f}s "
            f"{stats.p90:>10.2f}s {stats.p99:>10.2f}s {stats.mean:>10.2f}s"
        )


def print_pareto_summary(
    pareto: list[tuple[float, float, np.ndarray]], providers: list[ProviderConfig]
) -> None:
    """Print summary of Pareto frontier."""
    print("\n" + "=" * 80)
    print("Pareto Frontier Summary")
    print("=" * 80)

    for target_slo in [1.0, 2.0, 5.0, 10.0, 20.0, 30.0]:
        for slo, cost, pi in pareto:
            if abs(slo - target_slo) < 0.5:
                print(f"\nSLO = {target_slo}s (P99 constraint)")
                print(f"  Expected Cost: ${cost * 1000:.4f}/1000 req")
                print("  Routing Mix:")
                for i, prob in enumerate(pi):
                    if prob > 0.01:  # Only show providers with >1% traffic
                        print(f"    {providers[i].display_name}: {prob * 100:.1f}%")
                break


def load_provider_pricing(pricing_path: str | None) -> dict[str, tuple[float, float]]:
    """Load provider pricing map: provider -> (input_usd_per_1m, output_usd_per_1m).

    The pricing file must be a JSON object mapping provider name to a two-element
    list or tuple, e.g.:
        {"Groq": [0.25, 0.75], "Together": [0.3, 0.9]}
    """
    if not pricing_path:
        return {}
    with open(pricing_path) as f:
        raw = json.load(f)

    pricing: dict[str, tuple[float, float]] = {}
    if not isinstance(raw, dict):
        raise ValueError("pricing JSON must be an object mapping provider -> [in, out]")

    for provider, value in raw.items():
        if not isinstance(provider, str):
            continue
        if (
            isinstance(value, list | tuple)
            and len(value) == 2
            and isinstance(value[0], int | float)
            and isinstance(value[1], int | float)
        ):
            pricing[provider] = (float(value[0]), float(value[1]))

    return pricing


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Offline LP analysis for latency-cost tradeoff under a tail SLO.",
    )
    parser.add_argument(
        "--latency-csv",
        type=str,
        default="ICML2026_HybridInference/data/latency_llama70b_24h.csv",
        help="Path to Phase 1 profiling CSV.",
    )
    parser.add_argument(
        "--workload",
        type=str,
        default="base",
        help="Workload name to filter (e.g., base).",
    )
    parser.add_argument(
        "--metric",
        choices=["ttft", "e2e"],
        default="ttft",
        help="Latency metric used for SLO and plots.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Minimum successful samples per provider to include.",
    )
    parser.add_argument(
        "--avg-input-tokens",
        type=int,
        default=50,
        help="Average input tokens per request for cost model.",
    )
    parser.add_argument(
        "--avg-output-tokens",
        type=int,
        default=10,
        help="Average output tokens per request for cost model.",
    )
    parser.add_argument(
        "--burstgpt-csv",
        type=str,
        default="data/BurstGPT_1.csv",
        help="Path to BurstGPT trace (used only for simulation).",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=10000,
        help="Max requests to load from trace for simulation.",
    )
    parser.add_argument(
        "--use-trace-tokens",
        action="store_true",
        help="Use per-request tokens from trace for cost (may be inconsistent with probing).",
    )
    parser.add_argument(
        "--pricing-json",
        type=str,
        default=None,
        help="JSON file mapping provider -> [input_usd_per_1m, output_usd_per_1m].",
    )
    parser.add_argument(
        "--default-input-usd-per-1m",
        type=float,
        default=0.10,
        help="Default input price used when provider missing from pricing map.",
    )
    parser.add_argument(
        "--default-output-usd-per-1m",
        type=float,
        default=0.32,
        help="Default output price used when provider missing from pricing map.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for simulation sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiment/results/latency_phase2",
        help="Output directory for figures and CSV.",
    )
    return parser


def main():
    """Main function for Phase 2 offline latency analysis."""
    args = build_arg_parser().parse_args()

    latency_data_path = args.latency_csv
    burstgpt_path = args.burstgpt_csv
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric: LatencyMetric = args.metric
    workloads = (args.workload,)

    print("=" * 80)
    print("Phase 2: Latency-Aware Offline Routing Analysis")
    print("=" * 80)

    # 1. Load latency profiles from Phase 1
    print(f"\n[1] Loading latency profiles from: {latency_data_path}")
    profiles = load_latency_profiles(
        latency_data_path,
        metric=metric,
        workloads=workloads,
        min_samples=args.min_samples,
    )
    print(f"    Loaded profiles for {len(profiles)} providers")
    print_provider_stats(profiles, metric)

    # 2. Load BurstGPT workload
    print(f"\n[2] Loading BurstGPT workload from: {burstgpt_path}")
    requests = load_burstgpt_workload(burstgpt_path, max_requests=args.max_requests)
    print(f"    Loaded {len(requests)} requests")

    # 3. Setup providers with cost configuration
    provider_pricing = load_provider_pricing(args.pricing_json)
    if not provider_pricing:
        print(
            "\n[Warning] No provider pricing file provided. "
            "Costs will fall back to default prices, so cost-latency tradeoffs may be unreliable.\n"
        )

    avg_input_tokens = int(args.avg_input_tokens)
    avg_output_tokens = int(args.avg_output_tokens)

    providers = []
    for name, profile in profiles.items():
        input_price, output_price = provider_pricing.get(
            name, (args.default_input_usd_per_1m, args.default_output_usd_per_1m)
        )
        providers.append(
            ProviderConfig(
                name=name,
                display_name=name,
                input_price=input_price,
                output_price=output_price,
                profile=profile,
            )
        )

    # Sort by P99 latency for consistent ordering
    providers.sort(key=lambda p: p.profile.stats(metric).p99)

    print(f"\n[3] Configured {len(providers)} providers")

    # 4. Compute Pareto frontier
    print("\n[4] Computing Pareto frontier...")

    # Find SLO range based on provider P99s
    p99_values = [p.profile.stats(metric).p99 for p in providers]
    min_p99 = min(p99_values)
    max_p99 = max(p99_values)
    print(f"    Provider P99 range: {min_p99:.1f}s - {max_p99:.1f}s")

    pareto = compute_pareto_frontier(
        providers,
        metric=metric,
        slo_range=(min_p99 * 0.8, max_p99 * 1.1),
        n_points=200,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
    )
    print(f"    Computed {len(pareto)} Pareto points")

    print_pareto_summary(pareto, providers)

    # 5. Generate plots
    print("\n[5] Generating plots...")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )

    # Figure 1: 3-panel figure (CDF, Pareto, Routing Mix)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # Panel (a): CDF comparison
    ax1 = axes[0]
    colors = plt.cm.tab20(np.linspace(0, 1, len(providers)))

    for i, p in enumerate(providers[:8]):  # Top 8 by P99
        samples_ms = p.profile.ttft_samples if metric == "ttft" else p.profile.e2e_samples
        sorted_vals = np.sort(samples_ms / 1000.0)
        ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax1.plot(sorted_vals, ecdf, color=colors[i], linewidth=1.5, label=p.name)

    ax1.axvline(x=2.0, color="gray", linestyle="--", alpha=0.7, linewidth=1)
    ax1.text(2.1, 0.5, "SLO=2s", fontsize=8, color="gray")
    ax1.set_xlabel(f"{metric.upper()} (seconds)")
    ax1.set_ylabel("CDF")
    ax1.set_title("(a) Latency Distribution")
    ax1.legend(loc="lower right", fontsize=7, ncol=2)
    ax1.set_xlim(0, 15)
    ax1.set_ylim(0, 1.02)

    # Panel (b): Pareto frontier
    ax2 = axes[1]
    if pareto:
        p_slos = [p[0] for p in pareto]
        p_costs = [p[1] * 1000 for p in pareto]  # $/1000 requests
        ax2.plot(p_slos, p_costs, "b-", linewidth=2.5, label="LP Routing")

        # Baseline: cheapest single provider
        cheapest = min(
            providers,
            key=lambda p: p.cost_per_request(avg_input_tokens, avg_output_tokens),
        )
        cheapest_cost = cheapest.cost_per_request(avg_input_tokens, avg_output_tokens)
        ax2.axhline(
            y=cheapest_cost * 1000,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Cheapest ({cheapest.name})",
        )

    ax2.set_xlabel("Latency SLO (P99, seconds)")
    ax2.set_ylabel("Cost ($/1000 req)")
    ax2.set_title("(b) Cost-Latency Pareto Frontier")
    ax2.legend(loc="upper right")

    # Panel (c): Routing mix
    ax3 = axes[2]
    if pareto:
        routing_matrix = np.array([p[2] for p in pareto])
        # Only show top providers by average allocation; merge the rest into "Other".
        avg_allocation = routing_matrix.mean(axis=0)
        top_indices = [i for i in np.argsort(avg_allocation)[-6:] if avg_allocation[i] > 0.01]
        top_indices = list(reversed(top_indices))
        top_matrix = routing_matrix[:, top_indices] if top_indices else np.zeros((len(p_slos), 0))
        other = 1.0 - np.sum(top_matrix, axis=1) if top_indices else np.ones(len(p_slos))
        stack = np.column_stack([top_matrix, other])
        labels = [providers[i].name for i in top_indices] + ["Other"]
        colors_mix = [colors[i] for i in top_indices] + ["#bdc3c7"]
        ax3.stackplot(p_slos, stack.T, labels=labels, colors=colors_mix, alpha=0.8)

    ax3.set_xlabel("Latency SLO (P99, seconds)")
    ax3.set_ylabel("Routing Fraction")
    ax3.set_title("(c) Optimal Provider Mix")
    ax3.legend(loc="upper right", fontsize=7)
    ax3.set_ylim(0, 1)

    plt.tight_layout()
    output_path = output_dir / "latency_pareto_phase2.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {output_path}")

    # Figure 2: CCDF (log scale) for heavy-tail visualization
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, p in enumerate(providers):
        samples_ms = p.profile.ttft_samples if metric == "ttft" else p.profile.e2e_samples
        sorted_vals = np.sort(samples_ms / 1000.0)
        ccdf = 1 - np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.semilogy(sorted_vals, ccdf, color=colors[i], linewidth=1.5, label=p.name)

    # SLO reference lines
    for slo in [1.0, 2.0, 5.0]:
        ax.axvline(x=slo, color="gray", linestyle=":", alpha=0.5)
        ax.text(slo + 0.1, 0.5, f"SLO={slo}s", fontsize=8, color="gray", rotation=90)

    ax.axhline(y=0.01, color="red", linestyle="--", alpha=0.5, label="P99")
    ax.set_xlabel(f"{metric.upper()} (seconds)")
    ax.set_ylabel("P(Latency > x)")
    ax.set_title("Latency CCDF (Log Scale) - Heavy Tail Visualization")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_xlim(0, 30)
    ax.set_ylim(1e-4, 1)
    ax.grid(True, alpha=0.3, which="both")

    output_path = output_dir / "latency_ccdf_phase2.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {output_path}")

    # 6. Save results to CSV
    results_path = output_dir / "pareto_frontier.csv"
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["slo_sec", "cost_per_1000req"] + [p.name for p in providers]
        writer.writerow(header)
        for slo, cost, pi in pareto:
            row = [f"{slo:.2f}", f"{cost * 1000:.6f}"] + [f"{p:.4f}" for p in pi]
            writer.writerow(row)
    print(f"    Saved: {results_path}")

    print("\n" + "=" * 80)
    print("Phase 2 Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
