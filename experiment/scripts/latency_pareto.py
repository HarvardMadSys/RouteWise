"""Latency-Aware Routing: LP Formulation and Pareto Frontier Analysis.

This script implements the "SLO First" approach for latency-aware routing:
1. LP formulation with dynamic P99 tolerance
2. Pareto frontier analysis (cost vs latency tradeoff)
3. Comparison of different provider combinations

Key insight: With LP formulation, at most 2 providers are used at optimal solution.
"""

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

# Increase CSV field size limit
csv.field_size_limit(sys.maxsize)


@dataclass
class ProviderProfile:
    """Latency profile for a provider."""

    name: str
    display_name: str
    latencies: np.ndarray  # Raw latency samples
    cost_per_request: float  # Cost in $ per request (or per 1M tokens)

    @property
    def n_samples(self) -> int:
        """Return number of latency samples."""
        return len(self.latencies)

    def cdf_at(self, L: float) -> float:
        """Compute empirical CDF at latency L: P(latency <= L)."""
        return np.mean(self.latencies <= L)

    def percentile(self, p: float) -> float:
        """Return p-th percentile latency."""
        return np.percentile(self.latencies, p)

    def ecdf(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (x, y) for ECDF plot."""
        sorted_vals = np.sort(self.latencies)
        ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        return sorted_vals, ecdf


def load_latency_data(
    csv_path: str, model_filter: str | None = None
) -> dict[str, dict[str, list[float]]]:
    """Load latency data grouped by (model, provider).

    Supports two data formats:
    1. Legacy format: model_id, provider, latency_ms
    2. New format: request_id, provider, status, ttft_ms, e2e_ms, ...
    """
    data = defaultdict(lambda: defaultdict(list))

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        first_row = None
        for row in reader:
            if first_row is None:
                first_row = row

            # Detect format by checking columns
            if "ttft_ms" in row:
                # New format: use ttft_ms as latency
                provider = row["provider"]
                status = row.get("status", "success")
                ttft_ms = row.get("ttft_ms", "")

                # Only use successful requests
                if status != "success":
                    continue

                if ttft_ms and ttft_ms.strip():
                    try:
                        latency = float(ttft_ms) / 1000.0  # Convert to seconds
                        # Use a generic model name since new format doesn't have model_id
                        model = row.get("model_id", "llama-3.3-70b-instruct")
                        data[model][provider].append(latency)
                    except (ValueError, TypeError):
                        pass
            else:
                # Legacy format
                model = row["model_id"]
                provider = row["provider"]
                latency_ms = row.get("latency_ms", "")

                if model_filter and model != model_filter:
                    continue

                if latency_ms and latency_ms.strip():
                    try:
                        latency = float(latency_ms) / 1000.0  # Convert to seconds
                        data[model][provider].append(latency)
                    except (ValueError, TypeError):
                        pass

    return data


def solve_lp_routing(
    profiles: list[ProviderProfile], slo: float, target_percentile: float = 99.0
) -> tuple[np.ndarray, float]:
    """Solve the LP for optimal routing given SLO constraint.

    minimize: sum(pi_j * c_j)  (expected cost)
    subject to:
        sum(pi_j * F_j(slo)) >= target_percentile/100  (SLO constraint)
        sum(pi_j) = 1
        pi_j >= 0

    Returns:
        pi: routing probabilities
        cost: expected cost
    """
    n = len(profiles)

    # Objective: minimize cost
    c = np.array([p.cost_per_request for p in profiles])

    # Inequality constraint: sum(pi_j * F_j(slo)) >= 0.99
    # Rewrite as: -sum(pi_j * F_j(slo)) <= -0.99
    A_ub = np.array([[-p.cdf_at(slo) for p in profiles]])
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
        # Infeasible - return None
        return None, np.inf


def compute_pareto_frontier(
    profiles: list[ProviderProfile],
    slo_range: tuple[float, float] = (0.5, 30.0),
    n_points: int = 100,
) -> list[tuple[float, float, np.ndarray]]:
    """Compute Pareto frontier by sweeping over SLO values.

    Returns:
        List of (slo, cost, routing_probs) tuples.
    """
    slos = np.linspace(slo_range[0], slo_range[1], n_points)
    pareto = []

    for slo in slos:
        pi, cost = solve_lp_routing(profiles, slo)
        if pi is not None:
            pareto.append((slo, cost, pi))

    return pareto


def plot_pareto_frontier(
    pareto: list[tuple[float, float, np.ndarray]],
    profiles: list[ProviderProfile],
    output_path: Path,
    title: str = "Cost-Latency Pareto Frontier",
):
    """Plot the Pareto frontier with provider breakdown."""
    if not pareto:
        print("No valid Pareto points found")
        return

    slos = [p[0] for p in pareto]
    costs = [p[1] for p in pareto]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Pareto frontier
    ax1 = axes[0]
    ax1.plot(slos, costs, "b-", linewidth=2, label="Pareto Frontier")
    ax1.set_xlabel("Latency SLO (P99, seconds)", fontsize=12)
    ax1.set_ylabel("Expected Cost ($/request)", fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Mark key points
    for target_slo in [2.0, 5.0, 10.0]:
        for slo, cost, _pi in pareto:
            if abs(slo - target_slo) < 0.5:
                ax1.scatter([slo], [cost], s=100, zorder=5)
                ax1.annotate(
                    f"SLO={target_slo}s\nCost=${cost:.4f}",
                    (slo, cost),
                    textcoords="offset points",
                    xytext=(10, 10),
                    fontsize=9,
                )
                break

    ax1.legend()

    # Plot 2: Provider routing fractions
    ax2 = axes[1]
    colors = get_color_palette(len(profiles))

    # Stack plot for routing fractions
    routing_matrix = np.array([p[2] for p in pareto])

    ax2.stackplot(
        slos,
        routing_matrix.T,
        labels=[p.display_name for p in profiles],
        colors=colors[: len(profiles)],
        alpha=0.8,
    )

    ax2.set_xlabel("Latency SLO (P99, seconds)", fontsize=12)
    ax2.set_ylabel("Routing Fraction", fontsize=12)
    ax2.set_title("Optimal Provider Mix vs SLO", fontsize=14)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def get_color_palette(n: int) -> list[str]:
    """Get a color palette with at least n colors."""
    base_colors = [
        "#2ecc71",
        "#e74c3c",
        "#3498db",
        "#9b59b6",
        "#f39c12",
        "#1abc9c",
        "#e67e22",
        "#34495e",
        "#16a085",
        "#c0392b",
        "#2980b9",
        "#8e44ad",
        "#27ae60",
        "#d35400",
        "#7f8c8d",
    ]
    # Extend with matplotlib colormap if needed
    if n > len(base_colors):
        import matplotlib.cm as cm

        cmap = cm.get_cmap("tab20")
        base_colors = [cmap(i / n) for i in range(n)]
    return base_colors[:n]


def analyze_provider_tradeoffs(profiles: list[ProviderProfile], output_path: Path):
    """Analyze and visualize the cost-latency tradeoffs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = get_color_palette(len(profiles))

    # Plot 1: ECDF comparison
    ax1 = axes[0]
    for i, profile in enumerate(profiles):
        x, y = profile.ecdf()
        ax1.plot(
            x,
            y,
            color=colors[i],
            linewidth=2,
            label=f"{profile.display_name} (n={profile.n_samples})",
        )

    # Add SLO reference lines
    for slo in [2.0, 5.0, 10.0]:
        ax1.axvline(x=slo, color="red", linestyle=":", alpha=0.5)
        ax1.text(slo + 0.2, 0.5, f"{slo}s", fontsize=9, color="red")

    ax1.set_xlabel("Latency (seconds)", fontsize=12)
    ax1.set_ylabel("CDF", fontsize=12)
    ax1.set_title("Latency Distributions by Provider", fontsize=14)
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 30)

    # Plot 2: Cost vs P99 scatter
    ax2 = axes[1]
    for i, profile in enumerate(profiles):
        p99 = profile.percentile(99)
        ax2.scatter(
            [p99],
            [profile.cost_per_request],
            s=200,
            color=colors[i],
            label=profile.display_name,
            edgecolors="black",
            linewidth=2,
        )
        ax2.annotate(
            profile.display_name,
            (p99, profile.cost_per_request),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=10,
        )

    ax2.set_xlabel("P99 Latency (seconds)", fontsize=12)
    ax2.set_ylabel("Cost ($/request)", fontsize=12)
    ax2.set_title("Provider Cost vs Latency Tradeoff", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def compute_baseline_costs(
    profiles: list[ProviderProfile], slo: float
) -> dict[str, tuple[float, bool]]:
    """Compute costs for baseline strategies at a given SLO.

    Returns:
        Dict of strategy_name -> (cost, meets_slo).
    """
    baselines = {}

    for profile in profiles:
        # Single-provider strategy
        p99 = profile.percentile(99)
        meets_slo = p99 <= slo
        baselines[profile.display_name] = (profile.cost_per_request, meets_slo)

    return baselines


def print_summary_table(
    profiles: list[ProviderProfile], pareto: list[tuple[float, float, np.ndarray]]
):
    """Print summary statistics and routing decisions."""
    print("\n" + "=" * 80)
    print("Provider Statistics")
    print("=" * 80)
    print(f"{'Provider':<25} {'N':>8} {'P50':>8} {'P90':>8} {'P99':>8} {'Cost':>10}")
    print("-" * 80)

    for p in profiles:
        print(
            f"{p.display_name:<25} {p.n_samples:>8} {p.percentile(50):>8.2f} "
            f"{p.percentile(90):>8.2f} {p.percentile(99):>8.2f} {p.cost_per_request:>10.4f}"
        )

    print("\n" + "=" * 80)
    print("Optimal Routing at Key SLO Points")
    print("=" * 80)

    for target_slo in [2.0, 5.0, 10.0, 15.0, 20.0]:
        for slo, cost, pi in pareto:
            if abs(slo - target_slo) < 0.5:
                print(f"\nSLO = {target_slo}s (P99 constraint)")
                print(f"  Expected Cost: ${cost:.6f}/request")
                print("  Routing Mix:")
                for i, prob in enumerate(pi):
                    if prob > 0.01:  # Only show providers with >1% traffic
                        print(f"    {profiles[i].display_name}: {prob*100:.1f}%")

                # Compare with baselines
                baselines = compute_baseline_costs(profiles, target_slo)
                print("  Baseline Comparison:")
                for name, (base_cost, meets_slo) in baselines.items():
                    status = "OK" if meets_slo else "FAIL"
                    if meets_slo and base_cost > 0:
                        savings = (base_cost - cost) / base_cost * 100
                        print(f"    vs {name}: {status}, Savings={savings:.1f}%")
                    else:
                        print(f"    vs {name}: {status}")
                break


def plot_strategy_comparison(
    profiles: list[ProviderProfile],
    pareto: list[tuple[float, float, np.ndarray]],
    output_path: Path,
):
    """Plot comparison of LP routing vs baseline strategies."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    slos = [p[0] for p in pareto]
    lp_costs = [p[1] for p in pareto]

    colors = get_color_palette(len(profiles))

    # Plot 1: Cost comparison
    ax1 = axes[0]
    ax1.plot(slos, lp_costs, "b-", linewidth=2.5, label="LP Routing (Ours)")

    for i, profile in enumerate(profiles):
        p99 = profile.percentile(99)
        # Baseline: use this provider for all requests
        ax1.axhline(
            y=profile.cost_per_request,
            color=colors[i],
            linestyle="--",
            alpha=0.7,
            label=f"{profile.display_name} only",
        )
        ax1.axvline(x=p99, color=colors[i], linestyle=":", alpha=0.5)

    ax1.set_xlabel("Latency SLO (P99, seconds)", fontsize=12)
    ax1.set_ylabel("Expected Cost ($/request)", fontsize=12)
    ax1.set_title("LP Routing vs Single-Provider Baselines", fontsize=14)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 50)

    # Plot 2: Cost savings percentage
    ax2 = axes[1]

    # Find the API provider (most expensive that meets tight SLO)
    api_profile = None
    for p in profiles:
        if "API" in p.display_name:
            api_profile = p
            break

    if api_profile:
        api_cost = api_profile.cost_per_request
        if api_cost > 0:
            savings = [(api_cost - cost) / api_cost * 100 for _, cost, _ in pareto]
            ax2.fill_between(slos, 0, savings, alpha=0.3, color="green")
            ax2.plot(slos, savings, "g-", linewidth=2, label="Cost Savings vs API-only")

            ax2.set_xlabel("Latency SLO (P99, seconds)", fontsize=12)
            ax2.set_ylabel("Cost Savings (%)", fontsize=12)
            ax2.set_title("Cost Savings from LP Routing", fontsize=14)
            ax2.legend(loc="lower right")
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, 50)
            ax2.set_ylim(0, 105)

            # Annotate key points
            for target_slo in [5.0, 10.0, 20.0]:
                for _i, (slo, cost, _) in enumerate(pareto):
                    if abs(slo - target_slo) < 0.5:
                        save_pct = (api_cost - cost) / api_cost * 100
                        ax2.scatter([slo], [save_pct], s=100, color="green", zorder=5)
                        ax2.annotate(
                            f"{save_pct:.0f}%",
                            (slo, save_pct),
                            textcoords="offset points",
                            xytext=(5, 10),
                            fontsize=10,
                        )
                        break

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LP-based Pareto frontier analysis for latency-aware routing"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="ICML2026_HybridInference/data/latency_llama70b_24h_combined.csv",
        help="Path to latency data CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiment/results/latency_phase2",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--pricing",
        type=str,
        default="experiment/data/openrouter_llama33_70b.json",
        help="Path to pricing JSON file",
    )
    parser.add_argument(
        "--model", type=str, default="llama-3.3-70b-instruct", help="Model name to analyze"
    )
    parser.add_argument(
        "--slo-min", type=float, default=0.5, help="Minimum SLO value for Pareto sweep (seconds)"
    )
    parser.add_argument(
        "--slo-max", type=float, default=15.0, help="Maximum SLO value for Pareto sweep (seconds)"
    )
    parser.add_argument(
        "--n-points", type=int, default=200, help="Number of points for Pareto frontier sweep"
    )
    return parser.parse_args()


def main():
    """Run latency Pareto analysis."""
    args = parse_args()
    csv_path = args.data
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading latency data from {csv_path}...")
    all_data = load_latency_data(csv_path)

    # Check what models are available
    print(f"Available models: {list(all_data.keys())}")

    model = args.model

    if model not in all_data:
        # Try to find a matching model
        for m in all_data:
            if model.lower() in m.lower():
                model = m
                print(f"Using model: {model}")
                break
        else:
            print(f"Model {args.model} not found")
            return

    model_data = all_data[model]

    # Load cost structure from pricing config file
    # Format: [input_price, output_price] in $/1M tokens
    pricing_file = Path(args.pricing)
    cost_config = {}
    display_names = {}

    if pricing_file.exists():
        import json

        with open(pricing_file) as f:
            pricing_data = json.load(f)

        # Compute cost per request assuming average 500 input + 500 output tokens
        avg_input_tokens = 500
        avg_output_tokens = 500
        for provider, (input_price, output_price) in pricing_data.items():
            # Cost = (input_tokens * input_price + output_tokens * output_price) / 1M
            cost_per_request = (
                avg_input_tokens * input_price + avg_output_tokens * output_price
            ) / 1_000_000
            cost_config[provider] = cost_per_request
            display_names[provider] = provider
        print(f"Loaded pricing from {pricing_file}")
    else:
        # Fallback to hardcoded values
        cost_config = {
            "Groq": 0.00069,
            "SambaNova": 0.000675,
            "Cerebras": 0.001025,
            "Fireworks": 0.0009,
            "Together": 0.00088,
            "Parasail": 0.00021,
            "Friendli": 0.00036,
            "Novita": 0.0002675,
            "Cloudflare": 0.00127,
            "Nebius": 0.000265,
            "Hyperbolic": 0.0004,
            "Crusoe": 0.0005,
        }
        display_names = {k: k for k in cost_config}

    # Add legacy providers
    cost_config.update(
        {
            "sglang": 0.0,
            "chutes": 0.000133,
            "llama": 0.000275,
        }
    )
    display_names.update(
        {
            "sglang": "Local GPU",
            "chutes": "Chutes",
            "llama": "Llama API",
        }
    )

    # Create provider profiles
    profiles = []

    print(f"\nProviders found for {model}:")
    for provider, latencies in model_data.items():
        n_samples = len(latencies)
        print(f"  {provider}: {n_samples} samples")

        if n_samples < 20:  # Need enough samples
            print("    -> Skipping (too few samples)")
            continue

        profile = ProviderProfile(
            name=provider,
            display_name=display_names.get(provider, provider),
            latencies=np.array(latencies),
            cost_per_request=cost_config.get(provider, 0.0003),
        )
        profiles.append(profile)

    # Sort profiles by cost (cheapest first)
    profiles.sort(key=lambda p: p.cost_per_request)

    if len(profiles) < 1:
        print("No valid providers found")
        return

    print(f"\nAnalyzing {model} with {len(profiles)} providers")

    # Clean model name for filenames
    model_clean = model.replace(".", "_").replace("/", "_")

    # 1. Analyze provider tradeoffs
    analyze_provider_tradeoffs(profiles, output_dir / f"tradeoff_{model_clean}.png")

    if len(profiles) < 2:
        print("Need at least 2 providers for Pareto analysis")
        return

    # 2. Compute Pareto frontier
    print(f"\nComputing Pareto frontier (SLO range: {args.slo_min}s - {args.slo_max}s)...")
    pareto = compute_pareto_frontier(
        profiles, slo_range=(args.slo_min, args.slo_max), n_points=args.n_points
    )

    # 3. Plot Pareto frontier
    plot_pareto_frontier(
        pareto,
        profiles,
        output_dir / f"pareto_{model_clean}.png",
        title=f"{model} - Cost-Latency Pareto Frontier",
    )

    # 4. Print summary
    print_summary_table(profiles, pareto)

    # 5. Plot strategy comparison
    plot_strategy_comparison(profiles, pareto, output_dir / f"comparison_{model_clean}.png")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
