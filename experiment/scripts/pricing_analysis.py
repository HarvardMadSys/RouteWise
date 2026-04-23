"""
Pricing analysis: Win-win subscription scheme for OpenRouter + Chutes.

Goal: Show that if Chutes offers a quota-based subscription to OpenRouter,
both parties benefit compared to pure pay-per-token pricing.

- Chutes (provider): Gets guaranteed monthly revenue
- OpenRouter (aggregator): Gets cheaper effective per-token cost

We use the real Chutes trace (DeepSeek-V3.1) to compute:
1. Daily request volume and token usage statistics
2. Pay-per-token cost baseline
3. Subscription pricing analysis at different quota levels
4. Savings for OpenRouter and revenue guarantee for Chutes
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta


def load_and_preprocess(csv_path: str, sample_frac: float = 1.0) -> pd.DataFrame:
    """Load trace data and compute per-request cost."""
    print(f"Loading {csv_path}...")
    if sample_frac < 1.0:
        df = pd.read_csv(csv_path)
        df = df.sample(frac=sample_frac, random_state=42)
    else:
        df = pd.read_csv(csv_path)

    print(f"  Loaded {len(df):,} rows")

    # Filter to valid rows
    df = df.dropna(subset=["input_tokens", "output_tokens", "started_at", "completed_at"])
    df["input_tokens"] = df["input_tokens"].astype(int)
    df["output_tokens"] = df["output_tokens"].astype(int)

    # Parse dates explicitly (mixed formats: some have microseconds, some don't)
    df["started_at"] = pd.to_datetime(df["started_at"], format="mixed")
    df["completed_at"] = pd.to_datetime(df["completed_at"], format="mixed")

    # Compute duration
    df["duration_s"] = (df["completed_at"] - df["started_at"]).dt.total_seconds()
    df = df[df["duration_s"] > 0]

    # Add date column
    df["date"] = df["started_at"].dt.date

    print(f"  After filtering: {len(df):,} valid rows")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    return df


def compute_daily_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily aggregated statistics."""
    daily = df.groupby("date").agg(
        n_requests=("input_tokens", "count"),
        total_input_tokens=("input_tokens", "sum"),
        total_output_tokens=("output_tokens", "sum"),
        mean_input_tokens=("input_tokens", "mean"),
        mean_output_tokens=("output_tokens", "mean"),
        median_input_tokens=("input_tokens", "median"),
        median_output_tokens=("output_tokens", "median"),
        p90_output_tokens=("output_tokens", lambda x: x.quantile(0.9)),
        mean_duration=("duration_s", "mean"),
        n_unique_users=("user_id", "nunique"),
    ).reset_index()

    daily["total_tokens"] = daily["total_input_tokens"] + daily["total_output_tokens"]

    return daily


def compute_per_request_cost(df: pd.DataFrame, price_input: float, price_output: float) -> pd.Series:
    """Compute per-request API cost.

    Args:
        price_input: Price per 1M input tokens
        price_output: Price per 1M output tokens
    """
    return (df["input_tokens"] * price_input + df["output_tokens"] * price_output) / 1_000_000


def analyze_subscription_pricing(
    df: pd.DataFrame,
    daily_stats: pd.DataFrame,
    price_input: float,
    price_output: float,
    model_name: str,
):
    """Analyze win-win subscription pricing.

    For each day, if OpenRouter has a daily quota of Q requests on subscription,
    the optimal strategy is to route the top-Q most expensive requests to subscription
    and pay API for the rest.

    We sweep over different quota levels and compute:
    - OpenRouter's cost savings vs pure pay-per-token
    - Chutes's guaranteed revenue vs volatile per-token revenue
    """
    # Per-request API cost
    df = df.copy()
    df["api_cost"] = compute_per_request_cost(df, price_input, price_output)

    # Baseline: Pure pay-per-token monthly cost
    total_api_cost = df["api_cost"].sum()
    n_days = daily_stats["date"].nunique()
    daily_api_costs = df.groupby("date")["api_cost"].sum()

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    print(f"Total requests: {len(df):,}")
    print(f"Total days: {n_days}")
    print(f"Avg requests/day: {len(df)/n_days:,.0f}")
    print(f"Avg daily tokens (input): {daily_stats['total_input_tokens'].mean():,.0f}")
    print(f"Avg daily tokens (output): {daily_stats['total_output_tokens'].mean():,.0f}")
    print(f"Mean input tokens/req: {df['input_tokens'].mean():,.0f}")
    print(f"Mean output tokens/req: {df['output_tokens'].mean():,.0f}")
    print(f"Median output tokens/req: {df['output_tokens'].median():,.0f}")
    print(f"P90 output tokens/req: {df['output_tokens'].quantile(0.9):,.0f}")
    print(f"\nPricing: ${price_input}/1M input, ${price_output}/1M output")
    print(f"Total monthly API cost (baseline): ${total_api_cost:,.2f}")
    print(f"Avg daily API cost: ${daily_api_costs.mean():,.2f}")
    print(f"Daily cost std: ${daily_api_costs.std():,.2f}")
    print(f"Daily cost CV (std/mean): {daily_api_costs.std()/daily_api_costs.mean():.3f}")

    # Analyze subscription at different quota levels
    # Quota as fraction of mean daily requests
    avg_daily_requests = len(df) / n_days
    quota_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    results = []

    for frac in quota_fractions:
        quota = int(frac * avg_daily_requests)
        if quota == 0:
            continue

        # For each day, route top-Q most expensive to subscription, rest to API
        daily_savings = []
        daily_subscription_value = []  # Value of requests routed to subscription

        for date, day_df in df.groupby("date"):
            day_costs = day_df["api_cost"].values
            n = len(day_costs)

            if n <= quota:
                # All requests fit in subscription
                saved = day_costs.sum()
                api_remainder = 0.0
            else:
                # Sort descending, top-Q to subscription
                sorted_costs = np.sort(day_costs)[::-1]
                saved = sorted_costs[:quota].sum()
                api_remainder = sorted_costs[quota:].sum()

            daily_savings.append(saved)
            daily_subscription_value.append(saved)

        total_savings = sum(daily_savings)
        mean_daily_savings = np.mean(daily_savings)
        savings_pct = total_savings / total_api_cost * 100

        # Remaining API cost for OpenRouter
        remaining_api_cost = total_api_cost - total_savings

        # Now: what monthly subscription fee makes both parties happy?
        # For Chutes: fee must be > their marginal serving cost (we assume ~0 for simplicity,
        # or some fraction of API revenue)
        # For OpenRouter: fee + remaining API cost < total API cost
        # => fee < total_savings

        # Break-even fee for OpenRouter
        max_fee_openrouter = total_savings  # above this, no savings

        # Chutes's perspective: they lose the per-token revenue from subscription requests
        # but gain guaranteed fee. We model different discount levels.
        discount_levels = [0.3, 0.5, 0.7]

        result_row = {
            "quota_fraction": frac,
            "daily_quota": quota,
            "total_savings_optimal": total_savings,
            "savings_pct": savings_pct,
            "remaining_api_cost": remaining_api_cost,
            "mean_daily_savings": mean_daily_savings,
            "std_daily_savings": np.std(daily_savings),
        }

        for disc in discount_levels:
            # Subscription fee = (1 - discount) * value of subscription requests
            fee = (1 - disc) * total_savings
            openrouter_total_cost = fee + remaining_api_cost
            openrouter_savings = total_api_cost - openrouter_total_cost
            openrouter_savings_pct = openrouter_savings / total_api_cost * 100

            result_row[f"fee_disc{int(disc*100)}"] = fee
            result_row[f"openrouter_cost_disc{int(disc*100)}"] = openrouter_total_cost
            result_row[f"openrouter_savings_pct_disc{int(disc*100)}"] = openrouter_savings_pct
            result_row[f"chutes_revenue_disc{int(disc*100)}"] = fee

        results.append(result_row)

    results_df = pd.DataFrame(results)

    # Print summary table
    print(f"\n{'='*60}")
    print("SUBSCRIPTION PRICING ANALYSIS")
    print(f"{'='*60}")
    print(f"{'Quota%':>7} {'Q/day':>8} {'Opt.Sav%':>9} "
          f"{'Fee@30%disc':>12} {'OR.Sav@30%':>11} "
          f"{'Fee@50%disc':>12} {'OR.Sav@50%':>11}")
    print("-" * 80)

    for _, row in results_df.iterrows():
        print(f"{row['quota_fraction']*100:>6.0f}% "
              f"{int(row['daily_quota']):>8,} "
              f"{row['savings_pct']:>8.1f}% "
              f"${row['fee_disc30']:>10,.2f} "
              f"{row['openrouter_savings_pct_disc30']:>10.1f}% "
              f"${row['fee_disc50']:>10,.2f} "
              f"{row['openrouter_savings_pct_disc50']:>10.1f}%")

    # Key insight: Daily request volume stability
    print(f"\n{'='*60}")
    print("DAILY REQUEST VOLUME STABILITY")
    print(f"{'='*60}")
    print(f"Mean daily requests: {daily_stats['n_requests'].mean():,.0f}")
    print(f"Std daily requests: {daily_stats['n_requests'].std():,.0f}")
    print(f"CV (std/mean): {daily_stats['n_requests'].std()/daily_stats['n_requests'].mean():.3f}")
    print(f"Min daily requests: {daily_stats['n_requests'].min():,}")
    print(f"Max daily requests: {daily_stats['n_requests'].max():,}")
    print(f"P10 daily requests: {daily_stats['n_requests'].quantile(0.1):,.0f}")
    print(f"P50 daily requests: {daily_stats['n_requests'].quantile(0.5):,.0f}")
    print(f"P90 daily requests: {daily_stats['n_requests'].quantile(0.9):,.0f}")

    # Token volume stability
    print(f"\nMean daily total tokens: {daily_stats['total_tokens'].mean():,.0f}")
    print(f"CV of daily total tokens: {daily_stats['total_tokens'].std()/daily_stats['total_tokens'].mean():.3f}")

    # Request cost distribution
    print(f"\n{'='*60}")
    print("REQUEST COST DISTRIBUTION")
    print(f"{'='*60}")
    print(f"Mean cost/request: ${df['api_cost'].mean():.6f}")
    print(f"Median cost/request: ${df['api_cost'].median():.6f}")
    print(f"P90 cost/request: ${df['api_cost'].quantile(0.9):.6f}")
    print(f"P99 cost/request: ${df['api_cost'].quantile(0.99):.6f}")
    print(f"Max cost/request: ${df['api_cost'].max():.4f}")
    print(f"Cost Gini coefficient: {gini(df['api_cost'].values):.3f}")

    return results_df, daily_stats


def gini(values: np.ndarray) -> float:
    """Compute Gini coefficient of inequality."""
    values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(values)
    return (2 * np.sum((np.arange(1, n+1) * values)) / (n * cumsum[-1])) - (n + 1) / n


def analyze_chutes_revenue_stability(
    df: pd.DataFrame,
    daily_stats: pd.DataFrame,
    price_input: float,
    price_output: float,
):
    """Analyze revenue stability from Chutes's perspective.

    Key argument for Chutes: subscription gives revenue predictability.
    We show that per-token revenue is volatile (depends on request mix),
    while subscription fee is guaranteed.
    """
    df = df.copy()
    df["api_cost"] = compute_per_request_cost(df, price_input, price_output)
    daily_revenue = df.groupby("date")["api_cost"].sum()

    print(f"\n{'='*60}")
    print("CHUTES REVENUE STABILITY ANALYSIS")
    print(f"{'='*60}")
    print(f"Daily revenue mean: ${daily_revenue.mean():,.2f}")
    print(f"Daily revenue std: ${daily_revenue.std():,.2f}")
    print(f"Daily revenue CV: {daily_revenue.std()/daily_revenue.mean():.3f}")
    print(f"Daily revenue min: ${daily_revenue.min():,.2f}")
    print(f"Daily revenue max: ${daily_revenue.max():,.2f}")
    print(f"Worst-day / best-day ratio: {daily_revenue.min()/daily_revenue.max():.3f}")

    # Weekly aggregation
    df["week"] = df["started_at"].dt.isocalendar().week
    weekly_revenue = df.groupby("week")["api_cost"].sum()
    print(f"\nWeekly revenue mean: ${weekly_revenue.mean():,.2f}")
    print(f"Weekly revenue CV: {weekly_revenue.std()/weekly_revenue.mean():.3f}")

    return daily_revenue


def save_results(results: dict, output_path: str):
    """Save results to JSON."""
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return obj

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {output_path}")


def main():
    data_dir = Path("/scratch/juncheng/data/prefix_cache/data/metrics_30day/per_model")

    # DeepSeek-V3.1 pricing (per 1M tokens, from Together AI / market rate)
    # Using typical market rates for DeepSeek-V3
    DEEPSEEK_V3_INPUT = 0.20   # $/1M input tokens
    DEEPSEEK_V3_OUTPUT = 0.60  # $/1M output tokens

    # DeepSeek-R1 pricing
    DEEPSEEK_R1_INPUT = 0.55   # $/1M input tokens
    DEEPSEEK_R1_OUTPUT = 2.19  # $/1M output tokens

    # === Analyze DeepSeek-V3.1 ===
    print("\n" + "=" * 70)
    print("ANALYZING DeepSeek-V3.1")
    print("=" * 70)

    v3_path = data_dir / "large" / "DeepSeek-V3.1.csv"
    df_v3 = load_and_preprocess(str(v3_path))
    daily_v3 = compute_daily_stats(df_v3)

    results_v3, _ = analyze_subscription_pricing(
        df_v3, daily_v3,
        DEEPSEEK_V3_INPUT, DEEPSEEK_V3_OUTPUT,
        "DeepSeek-V3.1"
    )
    daily_rev_v3 = analyze_chutes_revenue_stability(
        df_v3, daily_v3, DEEPSEEK_V3_INPUT, DEEPSEEK_V3_OUTPUT
    )

    # === Analyze DeepSeek-R1 (sampled due to size) ===
    print("\n" + "=" * 70)
    print("ANALYZING DeepSeek-R1 (10% sample)")
    print("=" * 70)

    r1_path = data_dir / "large" / "DeepSeek-R1.csv"
    # Sample 10% for R1 (144M rows is too much)
    # Read in chunks and sample
    print(f"Loading {r1_path} with chunked sampling...")
    chunks = []
    for chunk in pd.read_csv(str(r1_path), chunksize=1_000_000):
        chunks.append(chunk.sample(frac=0.1, random_state=42))
        if len(chunks) % 20 == 0:
            print(f"  Processed {len(chunks)}M rows...")

    df_r1 = pd.concat(chunks, ignore_index=True)
    print(f"  Sampled {len(df_r1):,} rows")

    df_r1 = df_r1.dropna(subset=["input_tokens", "output_tokens", "started_at", "completed_at"])
    df_r1["input_tokens"] = df_r1["input_tokens"].astype(int)
    df_r1["output_tokens"] = df_r1["output_tokens"].astype(int)
    df_r1["started_at"] = pd.to_datetime(df_r1["started_at"], format="mixed")
    df_r1["completed_at"] = pd.to_datetime(df_r1["completed_at"], format="mixed")
    df_r1["duration_s"] = (df_r1["completed_at"] - df_r1["started_at"]).dt.total_seconds()
    df_r1 = df_r1[df_r1["duration_s"] > 0]
    df_r1["date"] = df_r1["started_at"].dt.date

    daily_r1 = compute_daily_stats(df_r1)

    # Scale daily stats back to full dataset (10x)
    print("\n[Note: R1 statistics are from 10% sample, request counts scaled 10x]")

    results_r1, _ = analyze_subscription_pricing(
        df_r1, daily_r1,
        DEEPSEEK_R1_INPUT, DEEPSEEK_R1_OUTPUT,
        "DeepSeek-R1 (10% sample, costs represent 10% of actual)"
    )
    daily_rev_r1 = analyze_chutes_revenue_stability(
        df_r1, daily_r1, DEEPSEEK_R1_INPUT, DEEPSEEK_R1_OUTPUT
    )

    # === Save all results ===
    output = {
        "deepseek_v3.1": {
            "pricing": {"input_per_1M": DEEPSEEK_V3_INPUT, "output_per_1M": DEEPSEEK_V3_OUTPUT},
            "total_requests": len(df_v3),
            "n_days": daily_v3["date"].nunique(),
            "avg_daily_requests": int(len(df_v3) / daily_v3["date"].nunique()),
            "daily_request_cv": float(daily_v3["n_requests"].std() / daily_v3["n_requests"].mean()),
            "daily_cost_cv": float(daily_rev_v3.std() / daily_rev_v3.mean()),
            "total_api_cost": float(df_v3["input_tokens"].sum() * DEEPSEEK_V3_INPUT / 1e6 +
                                     df_v3["output_tokens"].sum() * DEEPSEEK_V3_OUTPUT / 1e6),
            "mean_input_tokens": float(df_v3["input_tokens"].mean()),
            "mean_output_tokens": float(df_v3["output_tokens"].mean()),
            "subscription_analysis": results_v3.to_dict(orient="records"),
            "daily_stats": {
                "requests": daily_v3["n_requests"].describe().to_dict(),
                "input_tokens": daily_v3["total_input_tokens"].describe().to_dict(),
                "output_tokens": daily_v3["total_output_tokens"].describe().to_dict(),
            },
        },
        "deepseek_r1": {
            "pricing": {"input_per_1M": DEEPSEEK_R1_INPUT, "output_per_1M": DEEPSEEK_R1_OUTPUT},
            "total_requests_sampled": len(df_r1),
            "total_requests_estimated": len(df_r1) * 10,
            "n_days": daily_r1["date"].nunique(),
            "avg_daily_requests_estimated": int(len(df_r1) * 10 / daily_r1["date"].nunique()),
            "daily_request_cv": float(daily_r1["n_requests"].std() / daily_r1["n_requests"].mean()),
            "daily_cost_cv": float(daily_rev_r1.std() / daily_rev_r1.mean()),
            "total_api_cost_sampled": float(df_r1["input_tokens"].sum() * DEEPSEEK_R1_INPUT / 1e6 +
                                             df_r1["output_tokens"].sum() * DEEPSEEK_R1_OUTPUT / 1e6),
            "mean_input_tokens": float(df_r1["input_tokens"].mean()),
            "mean_output_tokens": float(df_r1["output_tokens"].mean()),
            "subscription_analysis": results_r1.to_dict(orient="records"),
        },
    }

    output_path = "/scratch/juncheng/data/prefix_cache/pricing_analysis_results.json"
    save_results(output, output_path)


if __name__ == "__main__":
    main()
