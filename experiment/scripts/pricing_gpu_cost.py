"""
Estimate Chutes' GPU cost from trace data and compute win-win pricing zone.

Approach:
1. From trace, compute peak throughput (output tokens/sec)
2. Compare to known DeepSeek-V3.1 serving benchmarks
3. Estimate GPU count and cost
4. Compute win-win zone: [GPU cost, per-token revenue]
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import timedelta


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and preprocess trace data."""
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} rows")

    df = df.dropna(subset=["input_tokens", "output_tokens", "started_at", "completed_at"])
    df["input_tokens"] = df["input_tokens"].astype(int)
    df["output_tokens"] = df["output_tokens"].astype(int)
    df["started_at"] = pd.to_datetime(df["started_at"], format="mixed")
    df["completed_at"] = pd.to_datetime(df["completed_at"], format="mixed")
    df["duration_s"] = (df["completed_at"] - df["started_at"]).dt.total_seconds()
    df = df[df["duration_s"] > 0]
    df["date"] = df["started_at"].dt.date

    print(f"  Valid rows: {len(df):,}")
    return df


def estimate_throughput(df: pd.DataFrame, bin_seconds: int = 60) -> pd.DataFrame:
    """Estimate throughput in time bins.

    For each time bin, compute:
    - Number of requests starting in this bin
    - Total output tokens from requests completing in this bin
    - Estimated concurrent requests (requests active during this bin)
    """
    # Use completed_at for throughput (tokens delivered)
    df = df.copy()
    df["bin"] = df["completed_at"].dt.floor(f"{bin_seconds}s")

    # Output throughput: tokens completed per bin
    throughput = df.groupby("bin").agg(
        completed_requests=("output_tokens", "count"),
        total_output_tokens=("output_tokens", "sum"),
        total_input_tokens=("input_tokens", "sum"),
        mean_duration=("duration_s", "mean"),
    ).reset_index()

    throughput["output_tps"] = throughput["total_output_tokens"] / bin_seconds
    throughput["input_tps"] = throughput["total_input_tokens"] / bin_seconds
    throughput["request_rate"] = throughput["completed_requests"] / bin_seconds

    return throughput


def estimate_peak_concurrency(df: pd.DataFrame, sample_points: int = 10000) -> dict:
    """Estimate peak concurrent requests by sampling time points."""
    # Sample random time points and count active requests
    min_time = df["started_at"].min()
    max_time = df["completed_at"].max()
    total_seconds = (max_time - min_time).total_seconds()

    # Sample time points
    np.random.seed(42)
    offsets = np.random.uniform(0, total_seconds, sample_points)
    sample_times = [min_time + timedelta(seconds=float(o)) for o in sorted(offsets)]

    # For efficiency, convert to numpy arrays
    starts = df["started_at"].values.astype("datetime64[ms]").astype(np.int64)
    ends = df["completed_at"].values.astype("datetime64[ms]").astype(np.int64)

    concurrencies = []
    for t in sample_times:
        t_ms = np.datetime64(t).astype("datetime64[ms]").astype(np.int64)
        active = np.sum((starts <= t_ms) & (ends > t_ms))
        concurrencies.append(active)

    concurrencies = np.array(concurrencies)

    return {
        "mean_concurrency": float(np.mean(concurrencies)),
        "median_concurrency": float(np.median(concurrencies)),
        "p90_concurrency": float(np.percentile(concurrencies, 90)),
        "p95_concurrency": float(np.percentile(concurrencies, 95)),
        "p99_concurrency": float(np.percentile(concurrencies, 99)),
        "max_concurrency": float(np.max(concurrencies)),
    }


def estimate_gpu_cost(
    peak_output_tps: float,
    p95_output_tps: float,
    mean_output_tps: float,
    model_name: str = "DeepSeek-V3.1",
) -> dict:
    """Estimate GPU cost based on throughput requirements.

    DeepSeek-V3.1 (685B MoE, 37B active):
    - Single 8xH100 instance: ~2000-4000 output tokens/sec aggregate with batching
    - Single 8xH200 instance: ~3000-6000 output tokens/sec

    H100 rental: ~$2.0-3.0/hr/GPU (cloud spot to on-demand range)
    """
    # Conservative and optimistic throughput per 8-GPU instance
    tps_per_instance_conservative = 2000  # tokens/sec, 8xH100
    tps_per_instance_optimistic = 4000    # tokens/sec, 8xH100 with good batching

    gpus_per_instance = 8

    # H100 cost range
    h100_cost_low = 2.0    # $/hr/GPU (spot/reserved)
    h100_cost_high = 3.0   # $/hr/GPU (on-demand)

    results = {}
    for scenario, tps_target in [
        ("provision_for_peak", peak_output_tps),
        ("provision_for_p95", p95_output_tps),
        ("provision_for_mean", mean_output_tps),
    ]:
        for tps_label, tps_per_inst in [
            ("conservative_tps", tps_per_instance_conservative),
            ("optimistic_tps", tps_per_instance_optimistic),
        ]:
            n_instances = max(1, int(np.ceil(tps_target / tps_per_inst)))
            n_gpus = n_instances * gpus_per_instance

            monthly_cost_low = n_gpus * h100_cost_low * 24 * 30
            monthly_cost_high = n_gpus * h100_cost_high * 24 * 30

            key = f"{scenario}_{tps_label}"
            results[key] = {
                "tps_target": tps_target,
                "n_instances": n_instances,
                "n_gpus": n_gpus,
                "monthly_cost_low": monthly_cost_low,
                "monthly_cost_high": monthly_cost_high,
            }

    return results


def compute_win_win_zone(
    monthly_api_revenue: float,
    gpu_cost_estimates: dict,
    subscription_analysis: list,
) -> None:
    """Print the win-win pricing zone analysis."""
    print("\n" + "=" * 70)
    print("WIN-WIN PRICING ZONE ANALYSIS")
    print("=" * 70)

    # Use a few representative GPU cost scenarios
    scenarios = [
        ("Best case (mean load, optimistic TPS, spot GPU)",
         gpu_cost_estimates["provision_for_mean_optimistic_tps"]["monthly_cost_low"]),
        ("Mid case (P95 load, conservative TPS, mid GPU)",
         gpu_cost_estimates["provision_for_p95_conservative_tps"]["monthly_cost_low"]),
        ("Worst case (peak load, conservative TPS, on-demand GPU)",
         gpu_cost_estimates["provision_for_peak_conservative_tps"]["monthly_cost_high"]),
    ]

    print(f"\nMonthly per-token revenue (status quo): ${monthly_api_revenue:,.2f}")
    print(f"\nGPU cost estimates:")
    for name, cost in scenarios:
        margin = (monthly_api_revenue - cost) / monthly_api_revenue * 100
        print(f"  {name}: ${cost:,.0f}/mo (margin: {margin:.1f}%)")

    # For each subscription quota level, compute the win-win zone
    print(f"\n{'='*70}")
    print("SUBSCRIPTION DEAL ANALYSIS")
    print("For each quota level, what subscription fee makes BOTH parties better off?")
    print(f"{'='*70}")

    # Use mid-case GPU cost as reference
    gpu_cost_mid = gpu_cost_estimates["provision_for_p95_conservative_tps"]["monthly_cost_low"]

    print(f"\nUsing mid-case GPU cost: ${gpu_cost_mid:,.0f}/mo")
    print(f"Provider profit at status quo: ${monthly_api_revenue - gpu_cost_mid:,.0f}/mo")
    print()

    print(f"{'Quota%':>7} {'Q/day':>8} {'OptSav$':>10} "
          f"{'MinFee':>10} {'MaxFee':>10} "
          f"{'Zone':>10} {'OR.Sav@mid':>11} {'Chutes.Rev@mid':>15}")
    print("-" * 95)

    for row in subscription_analysis:
        frac = row["quota_fraction"]
        quota = int(row["daily_quota"])
        optimal_savings = row["total_savings_optimal"]

        # Provider's minimum acceptable fee:
        # They need to cover the ADDITIONAL GPU cost for serving subscription requests
        # But since GPUs are already provisioned (fixed cost), marginal cost is ~0
        # So minimum fee is ~0 (any revenue from subscription is profit)
        # More realistically: they want at least to not lose money overall
        # If they give discount on subscription, they lose per-token revenue on those requests
        # Min fee = 0 (marginal cost argument)
        min_fee_provider = 0  # marginal cost of already-provisioned GPUs

        # OpenRouter's maximum acceptable fee:
        # fee must be < optimal_savings (otherwise no savings)
        max_fee_openrouter = optimal_savings

        # Win-win zone size
        zone_size = max_fee_openrouter - min_fee_provider

        # Mid-point fee (split savings equally)
        mid_fee = optimal_savings * 0.5

        # OpenRouter savings at mid fee
        or_savings_pct = (optimal_savings - mid_fee) / monthly_api_revenue * 100

        # Chutes total revenue at mid fee:
        # subscription fee + per-token from non-subscription requests
        remaining_api = monthly_api_revenue - optimal_savings
        chutes_total = mid_fee + remaining_api

        print(f"{frac*100:>6.0f}% "
              f"{quota:>8,} "
              f"${optimal_savings:>9,.0f} "
              f"${'~0':>9} "
              f"${max_fee_openrouter:>9,.0f} "
              f"${zone_size:>9,.0f} "
              f"{or_savings_pct:>10.1f}% "
              f"${chutes_total:>14,.0f}")

    # Detailed analysis of the 50% quota sweet spot
    row_50 = [r for r in subscription_analysis if abs(r["quota_fraction"] - 0.5) < 0.01][0]
    optimal_savings_50 = row_50["total_savings_optimal"]
    remaining_api_50 = monthly_api_revenue - optimal_savings_50

    print(f"\n{'='*70}")
    print("DETAILED: 50% Quota Sweet Spot")
    print(f"{'='*70}")
    print(f"Daily quota: {int(row_50['daily_quota']):,} requests")
    print(f"Optimal monthly savings: ${optimal_savings_50:,.2f}")
    print(f"Remaining API cost: ${remaining_api_50:,.2f}")
    print()

    print("Fee scenarios and outcomes:")
    print(f"{'Fee':>12} {'OR.Total':>12} {'OR.Save%':>9} "
          f"{'Chutes.Tot':>12} {'Chutes.vs.SQ':>13} {'Both happy?':>12}")
    print("-" * 75)

    for fee_frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        fee = fee_frac * optimal_savings_50
        or_total = fee + remaining_api_50
        or_save_pct = (monthly_api_revenue - or_total) / monthly_api_revenue * 100
        chutes_total = fee + remaining_api_50  # subscription fee + remaining per-token
        chutes_vs_sq = chutes_total / monthly_api_revenue * 100

        # Both happy if: OR saves money AND Chutes revenue > GPU cost
        or_happy = or_total < monthly_api_revenue
        chutes_happy = chutes_total > gpu_cost_mid
        both = "YES" if (or_happy and chutes_happy) else "no"

        print(f"${fee:>11,.0f} "
              f"${or_total:>11,.0f} "
              f"{or_save_pct:>8.1f}% "
              f"${chutes_total:>11,.0f} "
              f"{chutes_vs_sq:>12.1f}% "
              f"{'  ' + both:>12}")


def main():
    data_dir = Path("/scratch/juncheng/data/prefix_cache/data/metrics_30day/per_model")

    V3_IN, V3_OUT = 0.20, 0.60  # $/1M tokens

    # === Load DeepSeek-V3.1 ===
    df = load_data(str(data_dir / "large" / "DeepSeek-V3.1.csv"))

    # === Throughput analysis ===
    print("\n" + "=" * 60)
    print("THROUGHPUT ANALYSIS (1-minute bins)")
    print("=" * 60)

    tp = estimate_throughput(df, bin_seconds=60)

    print(f"Output tokens/sec:")
    print(f"  Mean: {tp['output_tps'].mean():,.0f}")
    print(f"  Median: {tp['output_tps'].median():,.0f}")
    print(f"  P90: {tp['output_tps'].quantile(0.9):,.0f}")
    print(f"  P95: {tp['output_tps'].quantile(0.95):,.0f}")
    print(f"  P99: {tp['output_tps'].quantile(0.99):,.0f}")
    print(f"  Max: {tp['output_tps'].max():,.0f}")

    print(f"\nInput tokens/sec:")
    print(f"  Mean: {tp['input_tps'].mean():,.0f}")
    print(f"  P95: {tp['input_tps'].quantile(0.95):,.0f}")
    print(f"  Max: {tp['input_tps'].max():,.0f}")

    print(f"\nRequest rate (req/sec):")
    print(f"  Mean: {tp['request_rate'].mean():,.1f}")
    print(f"  P95: {tp['request_rate'].quantile(0.95):,.1f}")
    print(f"  Max: {tp['request_rate'].max():,.1f}")

    # === Concurrency estimation (sampled) ===
    print("\n" + "=" * 60)
    print("CONCURRENCY ESTIMATION (sampled 10K time points)")
    print("=" * 60)

    conc = estimate_peak_concurrency(df, sample_points=10000)
    for k, v in conc.items():
        print(f"  {k}: {v:,.0f}")

    # === GPU cost estimation ===
    print("\n" + "=" * 60)
    print("GPU COST ESTIMATION")
    print("=" * 60)

    peak_tps = tp["output_tps"].max()
    p95_tps = tp["output_tps"].quantile(0.95)
    mean_tps = tp["output_tps"].mean()

    gpu_costs = estimate_gpu_cost(peak_tps, p95_tps, mean_tps)

    print(f"\nDeepSeek-V3.1 (685B MoE, 37B active params)")
    print(f"Serving: 8x H100 per instance")
    print(f"Throughput per instance: 2000-4000 output tok/s (with batching)")
    print(f"H100 cost: $2.0-3.0/hr/GPU")
    print()

    for key, val in gpu_costs.items():
        print(f"  {key}:")
        print(f"    Target TPS: {val['tps_target']:,.0f}")
        print(f"    Instances: {val['n_instances']}, GPUs: {val['n_gpus']}")
        print(f"    Monthly cost: ${val['monthly_cost_low']:,.0f} - ${val['monthly_cost_high']:,.0f}")

    # === Win-win zone ===
    # First recompute subscription analysis (from previous script's logic)
    df["api_cost"] = (df["input_tokens"] * V3_IN + df["output_tokens"] * V3_OUT) / 1_000_000
    total_api_cost = df["api_cost"].sum()
    n_days = df["date"].nunique()
    avg_daily_requests = len(df) / n_days

    subscription_analysis = []
    for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        quota = int(frac * avg_daily_requests)
        if quota == 0:
            continue
        daily_savings = []
        for date, day_df in df.groupby("date"):
            day_costs = day_df["api_cost"].values
            if len(day_costs) <= quota:
                saved = day_costs.sum()
            else:
                sorted_costs = np.sort(day_costs)[::-1]
                saved = sorted_costs[:quota].sum()
            daily_savings.append(saved)
        total_savings = sum(daily_savings)
        subscription_analysis.append({
            "quota_fraction": frac,
            "daily_quota": quota,
            "total_savings_optimal": total_savings,
        })

    compute_win_win_zone(total_api_cost, gpu_costs, subscription_analysis)

    # === Save results ===
    output = {
        "model": "DeepSeek-V3.1",
        "throughput": {
            "mean_output_tps": float(mean_tps),
            "p95_output_tps": float(p95_tps),
            "peak_output_tps": float(peak_tps),
        },
        "concurrency": conc,
        "gpu_cost_estimates": {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                                    for kk, vv in v.items()} for k, v in gpu_costs.items()},
        "monthly_api_revenue": float(total_api_cost),
        "subscription_analysis": subscription_analysis,
    }

    out_path = Path.home() / "hybridInference" / "experiment" / "results" / "pricing_gpu_cost.json"
    with open(str(out_path), "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
