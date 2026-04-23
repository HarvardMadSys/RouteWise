#!/usr/bin/env python3
"""Plot ablation results for SMART_ECONOMIC hedging strategy.

Generates:
1. Hedge rate vs cost_ratio for each SLO (grouped by backup method)
2. SLO violation rate vs cost_ratio
3. Pareto frontier: hedge rate vs violation rate across all strategies
4. Cross-backup stability comparison
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd


def load_ablation_results(result_dir: str) -> pd.DataFrame:
    """Load and combine ablation CSVs for all SLOs."""
    dfs = []
    for fname in sorted(os.listdir(result_dir)):
        if fname.startswith("ablation_slo") and fname.endswith(".csv"):
            path = os.path.join(result_dir, fname)
            df = pd.read_csv(path)
            dfs.append(df)
    if not dfs:
        print(f"No ablation files found in {result_dir}")
        sys.exit(1)
    return pd.concat(dfs, ignore_index=True)


def plot_cost_ratio_sweep(df: pd.DataFrame, output_dir: str) -> None:
    """Plot hedge rate and violation rate vs cost_ratio for SMART_ECONOMIC."""
    economic_df = df[df["strategy"].str.startswith("smart_economic")]
    if economic_df.empty:
        print("No smart_economic results found")
        return

    # Extract cost_ratio from strategy label
    economic_df = economic_df.copy()
    economic_df["cost_ratio_val"] = economic_df["strategy"].apply(
        lambda s: float(s.split("_cr")[1]) if "_cr" in s else 0.1
    )

    slos = sorted(economic_df["slo_sec"].unique())
    backup_methods = sorted(economic_df["backup_method"].unique())

    fig, axes = plt.subplots(len(slos), 2, figsize=(14, 4 * len(slos)), squeeze=False)
    fig.suptitle("SMART_ECONOMIC: cost_ratio Sweep", fontsize=14, fontweight="bold")

    for row, slo in enumerate(slos):
        slo_df = economic_df[economic_df["slo_sec"] == slo]

        # Hedge rate
        ax = axes[row, 0]
        for bm in backup_methods:
            bm_df = slo_df[slo_df["backup_method"] == bm].sort_values("cost_ratio_val")
            ax.plot(bm_df["cost_ratio_val"], bm_df["hedge_rate"] * 100, marker="o", label=bm)
        ax.set_xlabel("cost_ratio (C_b / V)")
        ax.set_ylabel("Hedge Rate (%)")
        ax.set_title(f"SLO = {slo}s")
        ax.legend(fontsize=8)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

        # Violation rate
        ax = axes[row, 1]
        for bm in backup_methods:
            bm_df = slo_df[slo_df["backup_method"] == bm].sort_values("cost_ratio_val")
            ax.plot(
                bm_df["cost_ratio_val"], bm_df["slo_violation_rate"] * 100, marker="s", label=bm
            )
        ax.set_xlabel("cost_ratio (C_b / V)")
        ax.set_ylabel("SLO Violation Rate (%)")
        ax.set_title(f"SLO = {slo}s")
        ax.legend(fontsize=8)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cost_ratio_sweep.png"), dpi=150)
    plt.close()
    print("Saved cost_ratio_sweep.png")


def plot_strategy_comparison(df: pd.DataFrame, output_dir: str) -> None:
    """Plot Pareto frontier: hedge rate vs violation rate across all strategies."""
    slos = sorted(df["slo_sec"].unique())

    fig, axes = plt.subplots(1, len(slos), figsize=(5 * len(slos), 5), squeeze=False)
    fig.suptitle(
        "Strategy Comparison: Hedge Rate vs Violation Rate", fontsize=14, fontweight="bold"
    )

    # Use fastest backup for cleaner comparison
    fastest_df = df[df["backup_method"] == "fastest"]

    # Color map for strategy families
    strategy_colors = {
        "never": "gray",
        "always": "red",
        "smart_survival": "blue",
        "smart_residual": "green",
        "percentile_based": "purple",
    }
    ft_colors = {
        "fixed_timeout_0.5": "orange",
        "fixed_timeout_0.7": "darkorange",
        "fixed_timeout_0.9": "brown",
    }

    for col, slo in enumerate(slos):
        ax = axes[0, col]
        slo_df = fastest_df[fastest_df["slo_sec"] == slo]

        # Plot non-economic strategies
        for _, row in slo_df.iterrows():
            s = row["strategy"]
            if s.startswith("smart_economic"):
                continue
            color = strategy_colors.get(s, ft_colors.get(s, "black"))
            ax.scatter(
                row["hedge_rate"] * 100,
                row["slo_violation_rate"] * 100,
                color=color,
                s=80,
                zorder=5,
                edgecolors="black",
                linewidth=0.5,
            )
            ax.annotate(
                s,
                (row["hedge_rate"] * 100, row["slo_violation_rate"] * 100),
                fontsize=6,
                textcoords="offset points",
                xytext=(5, 5),
            )

        # Plot economic strategies as connected line
        econ_df = slo_df[slo_df["strategy"].str.startswith("smart_economic")].copy()
        if not econ_df.empty:
            econ_df["cr"] = econ_df["strategy"].apply(
                lambda s: float(s.split("_cr")[1]) if "_cr" in s else 0.1
            )
            econ_df = econ_df.sort_values("cr")
            ax.plot(
                econ_df["hedge_rate"] * 100,
                econ_df["slo_violation_rate"] * 100,
                "o-",
                color="crimson",
                markersize=6,
                label="smart_economic",
                zorder=10,
            )
            for _, row in econ_df.iterrows():
                ax.annotate(
                    f"cr={row['cr']:.2f}",
                    (row["hedge_rate"] * 100, row["slo_violation_rate"] * 100),
                    fontsize=5,
                    textcoords="offset points",
                    xytext=(5, -8),
                    color="crimson",
                )

        ax.set_xlabel("Hedge Rate (%)")
        ax.set_ylabel("SLO Violation Rate (%)")
        ax.set_title(f"SLO = {slo}s")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "strategy_comparison.png"), dpi=150)
    plt.close()
    print("Saved strategy_comparison.png")


def plot_backup_stability(df: pd.DataFrame, output_dir: str) -> None:
    """Compare hedge rate variance across backup methods for each strategy."""
    slos = sorted(df["slo_sec"].unique())

    # For each strategy, compute hedge rate range across backup methods
    strategies = df["strategy"].unique()

    records = []
    for slo in slos:
        slo_df = df[df["slo_sec"] == slo]
        for s in strategies:
            s_df = slo_df[slo_df["strategy"] == s]
            if len(s_df) < 2:
                continue
            rates = s_df["hedge_rate"].values * 100
            records.append(
                {
                    "strategy": s,
                    "slo_sec": slo,
                    "hedge_rate_min": rates.min(),
                    "hedge_rate_max": rates.max(),
                    "hedge_rate_mean": rates.mean(),
                    "hedge_rate_range": rates.max() - rates.min(),
                }
            )

    if not records:
        return

    range_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Hedge Rate Stability Across Backup Methods", fontsize=14, fontweight="bold")

    # Plot for SLO=2.0 (the most problematic case)
    target_slo = 2.0
    if target_slo not in range_df["slo_sec"].values:
        target_slo = range_df["slo_sec"].iloc[0]

    plot_df = range_df[range_df["slo_sec"] == target_slo].sort_values("hedge_rate_range")

    y_pos = range(len(plot_df))
    ax.barh(
        y_pos,
        plot_df["hedge_rate_range"],
        left=plot_df["hedge_rate_min"],
        height=0.6,
        color="steelblue",
        alpha=0.7,
    )
    ax.scatter(plot_df["hedge_rate_mean"], y_pos, color="red", zorder=5, s=40)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["strategy"], fontsize=8)
    ax.set_xlabel("Hedge Rate (%)")
    ax.set_title(f"SLO = {target_slo}s (bar = range across backup methods, dot = mean)")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "backup_stability.png"), dpi=150)
    plt.close()
    print("Saved backup_stability.png")


def print_summary_table(df: pd.DataFrame) -> None:
    """Print summary table of key metrics."""
    fastest_df = df[df["backup_method"] == "fastest"]

    print("\n" + "=" * 100)
    print("Summary Table (backup=fastest)")
    print("=" * 100)
    print(
        f"{'Strategy':<30} {'SLO':>5} {'Hedge%':>8} {'Viol%':>8} {'AvgCost':>10} "
        f"{'P50ms':>8} {'P90ms':>8} {'P99ms':>8}"
    )
    print("-" * 100)

    for slo in sorted(fastest_df["slo_sec"].unique()):
        slo_df = fastest_df[fastest_df["slo_sec"] == slo]
        for _, row in slo_df.sort_values("strategy").iterrows():
            print(
                f"{row['strategy']:<30} {slo:>5.1f} {row['hedge_rate']*100:>7.1f}% "
                f"{row['slo_violation_rate']*100:>7.1f}% {row['avg_cost']:>10.6f} "
                f"{row['p50_ms']:>7.1f} {row['p90_ms']:>7.1f} {row['p99_ms']:>7.1f}"
            )
        print()


def main():
    """Generate hedging ablation plots from Phase 4 simulation results."""
    result_dir = "experiment/results/latency_phase4_economic"
    output_dir = os.path.join(result_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading results from {result_dir}...")
    df = load_ablation_results(result_dir)
    print(f"  Loaded {len(df)} result rows")

    print_summary_table(df)
    plot_cost_ratio_sweep(df, output_dir)
    plot_strategy_comparison(df, output_dir)
    plot_backup_stability(df, output_dir)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
