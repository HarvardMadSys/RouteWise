"""Offline counterfactual simulation: exclude dominant provider.

Answers Juncheng's Direction 3: "Run without the dominant provider.
If we are much better than baselines, the algorithm works."

Uses block bootstrap on openrouter_auto data to simulate what each
routing policy would have experienced without the dominant provider.

Usage:
    python -m experiment.scripts.simulate.offline_counterfactual \
        --csv experiment/results/phase5_qwen3_7d_clean/run_20260410_171625/evaluation_log.csv \
        --exclude WandB \
        --slo-ms 2000 \
        --n-trials 100 \
        --output-dir experiment/results/counterfactual_qwen3_no_wandb
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path for imports.
_project_root = str(Path(__file__).resolve().parents[3])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from experiment.scripts.simulate.bootstrap import (
    block_bootstrap,
    build_latency_pools,
    get_pool_with_fallback,
)
from experiment.scripts.simulate.policies import (
    RequestResult,
    cheapest_fixed_provider,
    fastest_fixed_provider,
    simulate_fixed_policy,
    simulate_hedge_request,
    simulate_oracle_best_per_window,
    simulate_round_robin,
)
# Import production classes directly via importlib to avoid triggering
# experiment.strategies.__init__.py which pulls in unrelated modules.
import importlib.util as _ilu

def _load_module(name: str, path: str):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    # Register in sys.modules so @dataclass can resolve type hints.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_strategies_dir = os.path.join(_project_root, "experiment", "strategies")
_router_mod = _load_module(
    "online_latency_router",
    os.path.join(_strategies_dir, "online_latency_router.py"),
)
_hedge_mod = _load_module(
    "smart_hedging",
    os.path.join(_strategies_dir, "smart_hedging.py"),
)

OnlineLatencyRouter = _router_mod.OnlineLatencyRouter
SmartHedger = _hedge_mod.SmartHedger
HedgingParams = _hedge_mod.HedgingParams
HedgingStrategy = _hedge_mod.HedgingStrategy
BackupSelectionMethod = _hedge_mod.BackupSelectionMethod
select_backup = _hedge_mod.select_backup


def load_openrouter_auto(
    csv_path: str,
    exclude_providers: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Load openrouter_auto data and extract per-provider costs.

    Returns the full openrouter_auto DataFrame (all requests, including
    those routed to excluded providers) and a cost dict for non-excluded
    providers.

    Args:
        csv_path: Path to evaluation_log.csv.
        exclude_providers: Providers to exclude from the cost dict
            (but their requests are kept in the DataFrame).

    Returns:
        (df, costs) where df has all openrouter_auto successful requests,
        and costs maps non-excluded providers to mean cost per request.
    """
    exclude = exclude_providers or set()

    full_df = pd.read_csv(csv_path)
    auto_df = full_df[full_df["policy"] == "openrouter_auto"].copy()
    auto_df = auto_df[auto_df["status"] == "success"].copy()
    auto_df = auto_df[auto_df["ttft_ms"].notna() & (auto_df["ttft_ms"] > 0)].copy()

    # Compute per-provider mean cost (excluding the dominant provider).
    cost_df = auto_df[~auto_df["actual_provider"].isin(exclude)]
    costs = cost_df.groupby("actual_provider")["cost_usd"].mean().to_dict()

    print(f"Loaded {len(auto_df)} openrouter_auto requests")
    print(f"  Providers: {sorted(auto_df['actual_provider'].unique())}")
    print(f"  Excluded from routing: {sorted(exclude)}")
    print(f"  Remaining providers: {sorted(costs.keys())}")

    return auto_df, costs


def compute_global_ttfts(
    df: pd.DataFrame,
    exclude_providers: set[str],
) -> dict[str, np.ndarray]:
    """Compute all TTFT values per non-excluded provider (for hindsight baselines)."""
    result = {}
    for prov, group in df.groupby("actual_provider"):
        if prov not in exclude_providers:
            result[prov] = group["ttft_ms"].values
    return result


def _split_into_lp_intervals(
    trial_df: pd.DataFrame,
    lp_interval_sec: float = 300.0,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Split trial into LP update intervals.

    Returns list of (interval_start_time, sim_times_array, window_idx_array).
    Within each interval, LP weights are fixed.
    """
    sim_times = trial_df["sim_time"].values
    win_indices = trial_df["window_idx"].values

    intervals = []
    t0 = float(sim_times[0])
    i_start = 0

    for i in range(1, len(sim_times)):
        if sim_times[i] - t0 >= lp_interval_sec:
            intervals.append((t0, sim_times[i_start:i], win_indices[i_start:i]))
            t0 = float(sim_times[i])
            i_start = i

    # Last interval.
    if i_start < len(sim_times):
        intervals.append((t0, sim_times[i_start:], win_indices[i_start:]))

    return intervals


def simulate_lp_mix_trial(
    trial_df: pd.DataFrame,
    pools: dict[int, dict[str, np.ndarray]],
    costs: dict[str, float],
    slo_ms: float,
    rng: np.random.Generator,
) -> list[RequestResult]:
    """Simulate lp_mix policy for one bootstrap trial.

    Uses production OnlineLatencyRouter per-request: LP updates every
    5 min (gated by lp_update_interval), routing uses real SWRR ordering.
    """
    slo_sec = slo_ms / 1000.0
    router = OnlineLatencyRouter(
        costs=costs,
        slo_sec=slo_sec,
        window_sec=900.0,
        weight_smoothing=0.3,
        lp_update_interval=300.0,
        relaxation_factors=[1.2, 1.5, 2.0],
    )

    results = []
    available_providers = sorted(costs.keys())
    sim_times = trial_df["sim_time"].values
    win_indices = trial_df["window_idx"].values
    n = len(trial_df)

    for i in range(n):
        t = float(sim_times[i])
        widx = int(win_indices[i])

        # LP update gated by router's internal lp_update_interval (5 min).
        router.update_lp(current_time=t, force=(i == 0))

        # Per-request SWRR routing (preserves production ordering).
        provider = router.route(current_time=t)
        if provider is None:
            provider = available_providers[i % len(available_providers)]

        pool = get_pool_with_fallback(pools, widx, provider)
        if pool is None or len(pool) == 0:
            for alt in available_providers:
                alt_pool = get_pool_with_fallback(pools, widx, alt)
                if alt_pool is not None and len(alt_pool) > 0:
                    provider = alt
                    pool = alt_pool
                    break
        if pool is None or len(pool) == 0:
            continue

        ttft_ms = float(rng.choice(pool))
        router.add_sample(provider, t, ttft_ms)
        results.append(RequestResult(
            provider=provider,
            ttft_ms=ttft_ms,
            cost_usd=costs.get(provider, 0.0),
            slo_violated=ttft_ms > slo_ms,
        ))

    return results


# Hedge parameter refresh interval in simulated seconds.
# Backup selection and hedge time are recomputed at this granularity
# (much finer than the 300s LP interval, but avoids recomputing
# expensive CDF/survival functions on every single request).
_HEDGE_REFRESH_SEC = 30.0


def simulate_smart_hedge_trial(
    trial_df: pd.DataFrame,
    pools: dict[int, dict[str, np.ndarray]],
    costs: dict[str, float],
    slo_ms: float,
    rng: np.random.Generator,
) -> list[RequestResult]:
    """Simulate smart_hedge policy for one bootstrap trial.

    Per-request SWRR routing. Backup selection and hedge time recomputed
    every 30s of sim_time (vs 300s LP interval). LP updates gated by
    the router's internal 5-min interval.
    """
    slo_sec = slo_ms / 1000.0
    router = OnlineLatencyRouter(
        costs=costs,
        slo_sec=slo_sec,
        window_sec=900.0,
        weight_smoothing=0.3,
        lp_update_interval=300.0,
        relaxation_factors=[1.2, 1.5, 2.0],
    )

    hedging_params = HedgingParams(
        strategy=HedgingStrategy.SMART_ECONOMIC,
        slo_sec=slo_sec,
        cost_ratio=0.05,
        dispatch_overhead_sec=0.05,
        backup_method=BackupSelectionMethod.FASTEST,
    )
    hedger = SmartHedger(hedging_params, costs)

    results = []
    available_providers = sorted(costs.keys())
    sim_times = trial_df["sim_time"].values
    win_indices = trial_df["window_idx"].values
    n = len(trial_df)

    # Hedge parameter cache, refreshed every _HEDGE_REFRESH_SEC.
    hedge_cache: dict[str, tuple[str | None, float]] = {}
    last_hedge_refresh = -float("inf")

    for i in range(n):
        t = float(sim_times[i])
        widx = int(win_indices[i])

        # LP update gated by router's internal interval.
        router.update_lp(current_time=t, force=(i == 0))

        # Refresh hedge params every 30s of sim_time.
        if t - last_hedge_refresh >= _HEDGE_REFRESH_SEC:
            hedge_cache.clear()
            for prov in available_providers:
                backup = select_backup(
                    method=BackupSelectionMethod.FASTEST,
                    profiles=router.profiles,
                    costs=costs,
                    primary=prov,
                    slo_sec=slo_sec,
                    now=t,
                    lp_weights=router.last_lp_weights,
                )
                if backup is not None and backup != prov:
                    h = hedger.compute_hedge_time(
                        prov, backup, router.profiles, t
                    )
                else:
                    backup = None
                    h = float("inf")
                hedge_cache[prov] = (backup, h)
            last_hedge_refresh = t

        # Per-request SWRR routing.
        primary = router.route(current_time=t)
        if primary is None:
            primary = available_providers[i % len(available_providers)]

        primary_pool = get_pool_with_fallback(pools, widx, primary)
        if primary_pool is None or len(primary_pool) == 0:
            for alt in available_providers:
                alt_pool = get_pool_with_fallback(pools, widx, alt)
                if alt_pool is not None and len(alt_pool) > 0:
                    primary = alt
                    primary_pool = alt_pool
                    break
        if primary_pool is None or len(primary_pool) == 0:
            continue

        primary_ttft_ms = float(rng.choice(primary_pool))

        backup, hedge_time = hedge_cache.get(primary, (None, float("inf")))
        backup_pool = (
            get_pool_with_fallback(pools, widx, backup)
            if backup is not None else None
        )

        result = simulate_hedge_request(
            primary_ttft_ms=primary_ttft_ms,
            hedge_time_sec=hedge_time,
            backup_pool=backup_pool,
            primary_provider=primary,
            backup_provider=backup,
            primary_cost=costs.get(primary, 0.0),
            backup_cost=costs.get(backup, 0.0) if backup else 0.0,
            slo_ms=slo_ms,
            dispatch_overhead_sec=0.05,
            rng=rng,
        )

        feed_prov = primary
        if result.hedge_triggered and result.hedge_winner == "backup" and result.hedge_provider:
            feed_prov = result.hedge_provider
        router.add_sample(feed_prov, t, result.ttft_ms)
        results.append(result)

    return results


def compute_metrics(results: list[RequestResult]) -> dict[str, float]:
    """Compute summary metrics from a list of RequestResult."""
    if not results:
        return {
            "n_requests": 0,
            "slo_violation_rate": float("nan"),
            "p50_ms": float("nan"),
            "p99_ms": float("nan"),
            "mean_cost": float("nan"),
            "total_cost": 0.0,
            "hedge_rate": 0.0,
        }

    ttfts = np.array([r.ttft_ms for r in results])
    violations = sum(1 for r in results if r.slo_violated)
    hedges = sum(1 for r in results if r.hedge_triggered)
    total_cost = sum(r.cost_usd for r in results)

    return {
        "n_requests": len(results),
        "slo_violation_rate": violations / len(results),
        "p50_ms": float(np.percentile(ttfts, 50)),
        "p99_ms": float(np.percentile(ttfts, 99)),
        "mean_cost": total_cost / len(results),
        "total_cost": total_cost,
        "hedge_rate": hedges / len(results),
    }


def run_simulation(
    csv_path: str,
    exclude_provider: str,
    slo_ms: float,
    n_trials: int,
    output_dir: str,
    window_sec: int = 900,
    seed: int = 42,
) -> pd.DataFrame:
    """Run the full offline counterfactual simulation.

    Args:
        csv_path: Path to evaluation_log.csv.
        exclude_provider: Provider to exclude (e.g., "WandB").
        slo_ms: SLO threshold in milliseconds.
        n_trials: Number of bootstrap trials.
        output_dir: Directory for output files.
        window_sec: Bootstrap window size in seconds.
        seed: Random seed.

    Returns:
        DataFrame with per-trial, per-policy metrics.
    """
    exclude_set = {exclude_provider}

    # Load data.
    auto_df, costs = load_openrouter_auto(csv_path, exclude_set)
    global_ttfts = compute_global_ttfts(auto_df, exclude_set)

    # Determine fixed baselines.
    cheapest_prov = cheapest_fixed_provider(costs)
    fastest_prov = fastest_fixed_provider(global_ttfts)
    all_providers = sorted(costs.keys())

    print(f"\nSimulation config:")
    print(f"  Excluded: {exclude_provider}")
    print(f"  SLO: {slo_ms} ms")
    print(f"  Trials: {n_trials}")
    print(f"  Window: {window_sec} s")
    print(f"  Cheapest provider: {cheapest_prov} (${costs[cheapest_prov]:.6f}/req)")
    print(f"  Fastest (P50) provider: {fastest_prov}")
    print(f"  All providers: {all_providers}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_trial_results = []
    t_start = time.time()

    for trial_idx, trial_df in enumerate(
        block_bootstrap(auto_df, window_sec, n_trials, seed)
    ):
        t_trial_start = time.time()

        # Build latency pools for this trial (exclude dominant provider).
        pools = build_latency_pools(trial_df, window_sec, exclude_set)
        n_windows = trial_df["window_idx"].nunique()

        # ------------------------------------------------------------------
        # Vectorized simple baselines.
        # Pre-compute per-window resolved pools for fixed providers to
        # avoid repeated get_pool_with_fallback calls.
        # ------------------------------------------------------------------
        unique_windows = sorted(trial_df["window_idx"].unique())
        window_indices = trial_df["window_idx"].values

        def _resolve_pools_for_provider(prov: str) -> dict[int, np.ndarray]:
            resolved = {}
            for w in unique_windows:
                p = get_pool_with_fallback(pools, w, prov)
                if p is not None and len(p) > 0:
                    resolved[w] = p
            return resolved

        def _vectorized_fixed(prov: str, cost: float, rng_f: np.random.Generator):
            resolved = _resolve_pools_for_provider(prov)
            ttfts = np.empty(len(trial_df))
            ttfts[:] = np.nan
            for i, w in enumerate(window_indices):
                pool = resolved.get(w)
                if pool is not None:
                    ttfts[i] = rng_f.choice(pool)
            valid = ~np.isnan(ttfts)
            valid_ttfts = ttfts[valid]
            return [
                RequestResult(
                    provider=prov, ttft_ms=t, cost_usd=cost,
                    slo_violated=t > slo_ms,
                )
                for t in valid_ttfts
            ]

        rng_fixed = np.random.default_rng(seed + trial_idx * 1000)
        cheapest_results = _vectorized_fixed(cheapest_prov, costs[cheapest_prov], rng_fixed)

        rng_fast = np.random.default_rng(seed + trial_idx * 1000 + 1)
        fastest_results = _vectorized_fixed(fastest_prov, costs.get(fastest_prov, 0.0), rng_fast)

        # 3. round_robin: vectorized per window.
        rng_rr = np.random.default_rng(seed + trial_idx * 1000 + 2)
        rr_results = []
        all_resolved = {prov: _resolve_pools_for_provider(prov) for prov in all_providers}
        n_prov = len(all_providers)
        for i, w in enumerate(window_indices):
            prov_idx = i % n_prov
            for offset in range(n_prov):
                prov = all_providers[(prov_idx + offset) % n_prov]
                pool = all_resolved[prov].get(w)
                if pool is not None:
                    ttft = float(rng_rr.choice(pool))
                    rr_results.append(RequestResult(
                        provider=prov, ttft_ms=ttft,
                        cost_usd=costs.get(prov, 0.0),
                        slo_violated=ttft > slo_ms,
                    ))
                    break

        # 4. oracle_best_per_window: pre-compute best provider per window.
        rng_oracle = np.random.default_rng(seed + trial_idx * 1000 + 3)
        oracle_best = {}
        for w in unique_windows:
            best_prov, best_p50 = None, float("inf")
            for prov in all_providers:
                pool = all_resolved[prov].get(w)
                if pool is not None:
                    p50 = float(np.median(pool))
                    if p50 < best_p50:
                        best_p50 = p50
                        best_prov = prov
            if best_prov is not None:
                oracle_best[w] = best_prov
        oracle_results = []
        for w in window_indices:
            prov = oracle_best.get(w)
            if prov is None:
                continue
            pool = all_resolved[prov].get(w)
            if pool is not None:
                ttft = float(rng_oracle.choice(pool))
                oracle_results.append(RequestResult(
                    provider=prov, ttft_ms=ttft,
                    cost_usd=costs.get(prov, 0.0),
                    slo_violated=ttft > slo_ms,
                ))

        # 5. lp_mix_no_dominant
        rng_lp = np.random.default_rng(seed + trial_idx * 1000 + 4)
        lp_results = simulate_lp_mix_trial(trial_df, pools, costs, slo_ms, rng_lp)

        # 6. smart_hedge_no_dominant
        rng_hedge = np.random.default_rng(seed + trial_idx * 1000 + 5)
        hedge_results = simulate_smart_hedge_trial(
            trial_df, pools, costs, slo_ms, rng_hedge
        )

        # Collect metrics.
        policies = {
            "cheapest_fixed": cheapest_results,
            "fastest_fixed": fastest_results,
            "round_robin": rr_results,
            "oracle_per_window": oracle_results,
            "lp_mix": lp_results,
            "smart_hedge": hedge_results,
        }

        for policy_name, policy_results in policies.items():
            metrics = compute_metrics(policy_results)
            metrics["trial"] = trial_idx
            metrics["policy"] = policy_name
            all_trial_results.append(metrics)

        elapsed = time.time() - t_trial_start
        total_elapsed = time.time() - t_start
        eta = (total_elapsed / (trial_idx + 1)) * (n_trials - trial_idx - 1)
        print(
            f"  Trial {trial_idx + 1}/{n_trials}: "
            f"{elapsed:.1f}s "
            f"(ETA: {eta / 60:.1f} min) "
            f"lp_slo={compute_metrics(lp_results)['slo_violation_rate']:.3f} "
            f"hedge_slo={compute_metrics(hedge_results)['slo_violation_rate']:.3f} "
            f"cheap_slo={compute_metrics(cheapest_results)['slo_violation_rate']:.3f}"
        )

    # Build results DataFrame.
    results_df = pd.DataFrame(all_trial_results)

    # Save per-trial results.
    trial_csv = os.path.join(output_dir, "per_trial_results.csv")
    results_df.to_csv(trial_csv, index=False)
    print(f"\nPer-trial results saved to {trial_csv}")

    # Compute and save summary.
    summary = compute_summary(results_df)
    summary_csv = os.path.join(output_dir, "summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"Summary saved to {summary_csv}")

    # Print summary table.
    print_summary(summary, exclude_provider)

    return results_df


def compute_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and 95% CI for each policy across trials."""
    metrics = ["slo_violation_rate", "p50_ms", "p99_ms", "mean_cost", "hedge_rate"]
    rows = []

    for policy, group in results_df.groupby("policy"):
        row = {"policy": policy, "n_trials": len(group)}
        for m in metrics:
            vals = group[m].dropna().values
            if len(vals) == 0:
                row[f"{m}_mean"] = float("nan")
                row[f"{m}_ci_lo"] = float("nan")
                row[f"{m}_ci_hi"] = float("nan")
                continue
            mean = np.mean(vals)
            # 95% CI via percentile bootstrap of means.
            ci_lo = np.percentile(vals, 2.5)
            ci_hi = np.percentile(vals, 97.5)
            row[f"{m}_mean"] = mean
            row[f"{m}_ci_lo"] = ci_lo
            row[f"{m}_ci_hi"] = ci_hi
        rows.append(row)

    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame, excluded: str) -> None:
    """Print a formatted summary table."""
    print(f"\n{'=' * 80}")
    print(f"COUNTERFACTUAL SIMULATION RESULTS (excluded: {excluded})")
    print(f"{'=' * 80}")
    print(
        f"{'Policy':<22} {'SLO Viol%':>10} {'95% CI':>16} "
        f"{'P99 (ms)':>10} {'95% CI':>16} "
        f"{'Cost/req':>10} {'Hedge%':>8}"
    )
    print("-" * 92)

    # Sort by SLO violation rate.
    for _, row in summary.sort_values("slo_violation_rate_mean").iterrows():
        slo = row["slo_violation_rate_mean"] * 100
        slo_lo = row["slo_violation_rate_ci_lo"] * 100
        slo_hi = row["slo_violation_rate_ci_hi"] * 100
        p99 = row["p99_ms_mean"]
        p99_lo = row["p99_ms_ci_lo"]
        p99_hi = row["p99_ms_ci_hi"]
        cost = row["mean_cost_mean"]
        hedge = row.get("hedge_rate_mean", 0) * 100

        print(
            f"{row['policy']:<22} {slo:>9.2f}% [{slo_lo:>5.2f}, {slo_hi:>5.2f}] "
            f"{p99:>9.0f} [{p99_lo:>6.0f}, {p99_hi:>6.0f}] "
            f"${cost:>8.6f} {hedge:>7.1f}%"
        )

    print("=" * 92)


def main():
    parser = argparse.ArgumentParser(
        description="Offline counterfactual simulation: exclude dominant provider"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to evaluation_log.csv",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        required=True,
        help="Provider to exclude (e.g., WandB, Inceptron)",
    )
    parser.add_argument(
        "--slo-ms",
        type=float,
        default=2000.0,
        help="SLO threshold in milliseconds (default: 2000)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of bootstrap trials (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: auto-generated)",
    )
    parser.add_argument(
        "--window-sec",
        type=int,
        default=900,
        help="Bootstrap window size in seconds (default: 900)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        exclude_lower = args.exclude.lower().replace(" ", "_")
        args.output_dir = f"experiment/results/counterfactual_no_{exclude_lower}"

    run_simulation(
        csv_path=args.csv,
        exclude_provider=args.exclude,
        slo_ms=args.slo_ms,
        n_trials=args.n_trials,
        output_dir=args.output_dir,
        window_sec=args.window_sec,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
