"""
Parameterized Win-Win Pricing Model for Subscription-Based LLM Routing.

Two parties:
  - OpenRouter (aggregator/buyer): routes LLM requests to providers
  - Chutes (inference provider/seller): serves open-weight models

Contract: (Q, F) = daily quota Q (token-normalized credits), monthly fee F

Key parameters:
  - V_opt(Q): Optimal monthly savings from routing top-Q requests/day to subscription
              (computable from trace via offline optimal algorithm)
  - V_A(Q) = gamma_A * V_opt(Q): Savings under algorithm A
              gamma_A = savings capture ratio (measures routing quality)
  - alpha_tranche: Chutes's current share of the subscription-covered tranche
              i.e., fraction of the top-Q requests that currently go to Chutes per-token
              alpha=0: none of this traffic currently goes to Chutes (pure incremental)
              alpha=1: all of this traffic already goes to Chutes (pure cannibalization)
  - Delta_K: Incremental capacity cost for Chutes to guarantee the subscription SLA
              (reserved capacity that Chutes must commit beyond current provisioning)

Win-win conditions (Stackelberg participation constraints):
  OpenRouter:  F < V_A(Q)                                  [subscription cheaper than paygo]
  Chutes:      F > alpha_tranche * V_A(Q) + Delta_K        [better than outside option]

  Fee lower bound = alpha_tranche * V_A(Q) + Delta_K
  Fee upper bound = V_A(Q)

Win-win zone exists iff:  (1 - alpha_tranche) * V_A(Q) > Delta_K

Total surplus = (1 - alpha_tranche) * V_A(Q) - Delta_K

Nash bargaining: F* = fee_lower + surplus / 2  (equal split of surplus)

Critical alpha threshold:  alpha_crit = 1 - Delta_K / V_A(Q)
  Zone vanishes when alpha_tranche >= alpha_crit

Key insight: gamma_A determines the surplus frontier of the contract game.
  Better routing (higher gamma) expands the feasible contract set.
"""

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path


# ======================================================================
# Model Definition
# ======================================================================

def compute_optimal_savings(df: pd.DataFrame, quota: int) -> float:
    """Compute S*(Q): optimal monthly savings with daily quota Q.

    For each day, route the top-Q most expensive requests to subscription.
    Returns total avoided API cost over the month.
    """
    total_savings = 0.0
    for _, day_df in df.groupby("date"):
        costs = day_df["api_cost"].values
        if len(costs) <= quota:
            total_savings += costs.sum()
        else:
            top_q = np.partition(costs, -quota)[-quota:]
            total_savings += top_q.sum()
    return total_savings


def compute_greedy_savings(df: pd.DataFrame, quota: int) -> float:
    """Compute S_greedy(Q): savings with naive FCFS/greedy routing.

    Greedy: assign first Q requests of each day to subscription (no intelligence).
    """
    total_savings = 0.0
    for _, day_df in df.groupby("date"):
        costs = day_df["api_cost"].values
        assigned = costs[:min(len(costs), quota)]
        total_savings += assigned.sum()
    return total_savings


def winwin_zone(s_star: float, alpha_tranche: float, delta_k: float) -> dict:
    """Compute win-win zone for given parameters.

    Fee bounds:
      lower = alpha_tranche * S*(Q) + Delta_K    (Chutes's minimum)
      upper = S*(Q)                                (OpenRouter's maximum)

    Zone exists iff upper > lower, i.e., (1 - alpha_tranche) * S*(Q) > Delta_K
    """
    fee_lower = alpha_tranche * s_star + delta_k
    fee_upper = s_star
    zone_exists = fee_upper > fee_lower
    zone_size = max(0, fee_upper - fee_lower)
    total_surplus = max(0, (1 - alpha_tranche) * s_star - delta_k)

    # Nash bargaining solution (equal split of surplus)
    if zone_exists:
        fee_nash = fee_lower + total_surplus / 2
        or_surplus = fee_upper - fee_nash
        chutes_surplus = fee_nash - fee_lower
    else:
        fee_nash = None
        or_surplus = 0
        chutes_surplus = 0

    return {
        "fee_lower": fee_lower,
        "fee_upper": fee_upper,
        "zone_exists": zone_exists,
        "zone_size": zone_size,
        "total_surplus": total_surplus,
        "fee_nash": fee_nash,
        "or_surplus": or_surplus,
        "chutes_surplus": chutes_surplus,
    }


# ======================================================================
# Data Loading
# ======================================================================

def load_trace(csv_path: str, price_in: float, price_out: float) -> pd.DataFrame:
    """Load trace and compute per-request API cost."""
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["input_tokens", "output_tokens", "started_at", "completed_at"])
    df["input_tokens"] = df["input_tokens"].astype(int)
    df["output_tokens"] = df["output_tokens"].astype(int)
    df["started_at"] = pd.to_datetime(df["started_at"], format="mixed")
    df["completed_at"] = pd.to_datetime(df["completed_at"], format="mixed")
    df["duration_s"] = (df["completed_at"] - df["started_at"]).dt.total_seconds()
    df = df[df["duration_s"] > 0]
    df["date"] = df["started_at"].dt.date
    df["api_cost"] = (df["input_tokens"] * price_in + df["output_tokens"] * price_out) / 1_000_000
    print(f"  {len(df):,} valid rows, {df['date'].nunique()} days")
    return df


# ======================================================================
# Analysis
# ======================================================================

def compute_savings_table(df: pd.DataFrame, avg_daily_requests: int) -> dict:
    """Precompute S*(Q) and S_greedy(Q) for a set of quota fractions."""
    quota_fracs = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    table = {}
    for qf in quota_fracs:
        quota = int(qf * avg_daily_requests)
        if quota == 0:
            continue
        s_star = compute_optimal_savings(df, quota)
        s_greedy = compute_greedy_savings(df, quota)
        table[qf] = {
            "quota": quota,
            "s_star": s_star,
            "s_greedy": s_greedy,
        }
        print(f"  Q={qf*100:.0f}% ({quota:,}/day): S*=${s_star:,.0f}, "
              f"S_greedy=${s_greedy:,.0f}, ratio={s_star/s_greedy:.2f}x")
    return table


def sweep_alpha_deltak(s_star: float, c_paygo: float) -> pd.DataFrame:
    """Sweep over alpha_tranche and Delta_K to map the full win-win landscape."""
    alphas = np.arange(0, 1.01, 0.05)
    # Delta_K as fraction of S*(Q)
    dk_fracs = np.arange(0, 0.51, 0.025)

    rows = []
    for alpha in alphas:
        for dk_frac in dk_fracs:
            delta_k = dk_frac * s_star
            ww = winwin_zone(s_star, alpha, delta_k)

            # OpenRouter savings at Nash fee
            if ww["fee_nash"] is not None:
                or_total_spend = ww["fee_nash"] + (c_paygo - s_star)
                or_savings_pct = (c_paygo - or_total_spend) / c_paygo * 100
            else:
                or_total_spend = c_paygo
                or_savings_pct = 0.0

            rows.append({
                "alpha_tranche": round(alpha, 3),
                "delta_k_frac": round(dk_frac, 4),
                "delta_k": delta_k,
                "s_star": s_star,
                "fee_lower": ww["fee_lower"],
                "fee_upper": ww["fee_upper"],
                "zone_exists": ww["zone_exists"],
                "zone_size": ww["zone_size"],
                "total_surplus": ww["total_surplus"],
                "fee_nash": ww["fee_nash"],
                "or_savings_pct": or_savings_pct,
            })

    return pd.DataFrame(rows)


def print_model_summary(
    savings_table: dict,
    c_paygo: float,
    avg_daily_req: int,
    n_days: int,
):
    """Print the formal model and key results."""
    sep = "=" * 72

    print(f"\n{sep}")
    print("PARAMETERIZED WIN-WIN PRICING MODEL")
    print(sep)

    print("""
MODEL SETUP
-----------
  Two parties: OpenRouter (buyer), Chutes (seller)
  Contract: (Q, F) = daily quota Q requests, monthly fee F

  Parameters:
    S*(Q)           = Optimal monthly savings from routing top-Q req/day
                      to subscription (from trace)
    alpha_tranche   = Chutes's current share of the top-Q tranche
                      (outside option for subscription-covered requests)
    Delta_K         = Incremental capacity reservation cost

  WIN-WIN CONDITIONS:
    OpenRouter:  F < S*(Q)
    Chutes:      F > alpha_tranche * S*(Q) + Delta_K

    Zone exists iff:  (1 - alpha_tranche) * S*(Q) > Delta_K
    Critical alpha:   alpha_crit = 1 - Delta_K / S*(Q)

  TOTAL SURPLUS = (1 - alpha_tranche) * S*(Q) - Delta_K
""")

    print(f"{sep}")
    print(f"TRACE-BASED S*(Q)  (DeepSeek-V3.1, Chutes 30-day trace)")
    print(sep)
    print(f"  C_paygo (monthly per-token baseline): ${c_paygo:,.2f}")
    print(f"  Avg daily requests: {avg_daily_req:,}")
    print(f"  Days: {n_days}")
    print()

    print(f"{'Quota%':>7} {'Q/day':>8} {'S*(Q)':>12} {'S*/C_pay':>9} "
          f"{'S_greedy':>12} {'Ratio':>6}")
    print("-" * 60)

    for qf in sorted(savings_table.keys()):
        t = savings_table[qf]
        ratio = t["s_star"] / t["s_greedy"] if t["s_greedy"] > 0 else float("inf")
        print(f"{qf*100:>6.0f}% "
              f"{t['quota']:>8,} "
              f"${t['s_star']:>11,.0f} "
              f"{t['s_star']/c_paygo*100:>8.1f}% "
              f"${t['s_greedy']:>11,.0f} "
              f"{ratio:>5.2f}x")

    # Win-win zone table at Q=50%, Delta_K=0
    s_star_50 = savings_table[0.5]["s_star"]

    print(f"\n{sep}")
    print(f"WIN-WIN ZONE  (Q=50%, Delta_K=0)")
    print(sep)
    print(f"  S*(Q) = ${s_star_50:,.0f}")
    print()

    print(f"{'alpha':>6} {'OutsideOpt':>11} {'Fee_lo':>10} {'Fee_hi':>10} "
          f"{'Zone$':>10} {'F_nash':>10} {'OR.Sav%':>8}")
    print("-" * 72)

    for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        ww = winwin_zone(s_star_50, alpha, 0.0)
        outside = alpha * s_star_50  # consistent: outside option on the tranche

        if ww["fee_nash"] is not None:
            or_total = ww["fee_nash"] + (c_paygo - s_star_50)
            or_sav = (c_paygo - or_total) / c_paygo * 100
            f_nash_str = f"${ww['fee_nash']:>9,.0f}"
            or_sav_str = f"{or_sav:>7.1f}%"
        else:
            f_nash_str = "      N/A"
            or_sav_str = "    N/A"

        print(f"{alpha:>5.1f} "
              f"${outside:>10,.0f} "
              f"${ww['fee_lower']:>9,.0f} "
              f"${ww['fee_upper']:>9,.0f} "
              f"${ww['zone_size']:>9,.0f} "
              f"{f_nash_str} "
              f"{or_sav_str}")

    # Critical alpha
    print(f"\n  alpha_crit = 1 - Delta_K / S*(Q)")
    print(f"  At Delta_K=0: alpha_crit = 1.0 (zone vanishes only at pure cannibalization)")
    for dk_frac in [0.1, 0.2, 0.3]:
        dk = dk_frac * s_star_50
        alpha_crit = 1.0 - dk / s_star_50
        print(f"  At Delta_K={dk_frac*100:.0f}% of S* (${dk:,.0f}): "
              f"alpha_crit = {alpha_crit:.2f}")

    # Sensitivity: alpha vs quota at Delta_K=0
    print(f"\n{sep}")
    print("SENSITIVITY: OR savings (%) at Nash fee  (Delta_K=0)")
    print(sep)

    alphas_show = [0.0, 0.1, 0.3, 0.5]
    header = f"{'Quota%':>7}"
    for a in alphas_show:
        header += f"  a={a:.1f}"
    print(header)
    print("-" * (8 + 9 * len(alphas_show)))

    for qf in sorted(savings_table.keys()):
        s = savings_table[qf]["s_star"]
        line = f"{qf*100:>6.0f}%"
        for a in alphas_show:
            ww = winwin_zone(s, a, 0.0)
            if ww["fee_nash"] is not None:
                or_total = ww["fee_nash"] + (c_paygo - s)
                or_sav = (c_paygo - or_total) / c_paygo * 100
                line += f"  {or_sav:>6.1f}%"
            else:
                line += f"     N/A"
        print(line)


def print_stage1_gamma_bridge_table(savings_table: dict, c_paygo: float) -> None:
    """Print a quota-only bridge table using Stage 1 empirical gamma values.

    This avoids mixing Stage 2 joint-routing gamma values into a quota-only
    contract model. The empirical routing quality is taken from the current
    paper/slide Stage 1 results:
      All-API = 907, Greedy = 691, PD-EMA = 629, LA-EMA = 633, Optimal = 532
    """
    # Stage 1 quota-only empirical results from the current paper/slides.
    all_api_cost = 907.0
    greedy_cost = 691.0
    pd_cost = 629.0
    la_cost = 633.0
    optimal_cost = 532.0

    s_opt_stage1 = all_api_cost - optimal_cost
    s_greedy_stage1 = all_api_cost - greedy_cost
    s_pd_stage1 = all_api_cost - pd_cost
    s_la_stage1 = all_api_cost - la_cost

    gamma_table = {
        "Greedy": s_greedy_stage1 / s_opt_stage1,
        "PD-EMA": s_pd_stage1 / s_opt_stage1,
        "LA-EMA": s_la_stage1 / s_opt_stage1,
        "Optimal": 1.0,
    }

    # Match the main win-win example: Q=50%, alpha_tranche=0.3, Delta_K=0.
    s_opt_q50 = float(savings_table[0.5]["s_star"])
    alpha_tranche = 0.3
    delta_k = 0.0

    sep = "=" * 72
    print(f"\n{sep}")
    print("STAGE 1 GAMMA BRIDGE (Quota-Only Contract, Existing Paper Results)")
    print(sep)
    print(
        "Bridge assumptions: use Stage 1 empirical gamma to scale the "
        "Chutes trace-based quota-only S*(Q)."
    )
    print(
        f"  Q=50% of daily volume, S_opt(Q)=${s_opt_q50:,.0f}, "
        f"alpha_tranche={alpha_tranche:.1f}, Delta_K=${delta_k:,.0f}"
    )
    print()
    print(
        f"{'Algorithm':<8} {'gamma':>7} {'V_A(Q)':>10} {'ZoneWidth':>11} "
        f"{'F_nash':>10} {'OR.Sav%':>9}"
    )
    print("-" * 66)

    for name, gamma in gamma_table.items():
        s_alg = gamma * s_opt_q50
        ww = winwin_zone(s_alg, alpha_tranche, delta_k)
        if ww["fee_nash"] is not None:
            or_total = ww["fee_nash"] + (c_paygo - s_alg)
            or_sav_pct = (c_paygo - or_total) / c_paygo * 100
            fee_nash_str = f"{ww['fee_nash']:>10.0f}"
        else:
            or_sav_pct = 0.0
            fee_nash_str = "       N/A"

        print(
            f"{name:<8} {gamma:>7.3f} {s_alg:>10.0f} {ww['zone_size']:>11.0f} "
            f"{fee_nash_str} {or_sav_pct:>8.1f}%"
        )


def plot_all(
    savings_table: dict,
    sweep_df: pd.DataFrame,
    c_paygo: float,
    output_dir: str,
):
    """Generate all visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ------------------------------------------------------------------
    # Plot 1: S*(Q) vs Greedy
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    quotas = sorted(savings_table.keys())
    s_stars = [savings_table[q]["s_star"] / c_paygo * 100 for q in quotas]
    s_greedys = [savings_table[q]["s_greedy"] / c_paygo * 100 for q in quotas]
    q_pcts = [q * 100 for q in quotas]

    ax.plot(q_pcts, s_stars, "o-", color="C0", linewidth=2, markersize=8,
            label="Optimal (our algorithm)")
    ax.plot(q_pcts, s_greedys, "s--", color="C3", linewidth=2, markersize=8,
            label="Greedy (FCFS)")
    ax.fill_between(q_pcts, s_greedys, s_stars, alpha=0.15, color="C0",
                     label="Smart routing gain")
    ax.set_xlabel("Daily Quota (% of avg daily requests)", fontsize=12)
    ax.set_ylabel("Savings (% of paygo cost)", fontsize=12)
    ax.set_title("(a) Savings: Optimal vs Greedy Routing", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Plot 2: Win-win zone vs alpha (Q=50%, Delta_K=0)
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    s_star_50 = savings_table[0.5]["s_star"]
    alphas_plot = np.linspace(0, 1, 50)
    fee_los = [a * s_star_50 for a in alphas_plot]
    fee_his = [s_star_50] * len(alphas_plot)
    nash_fees = [(a * s_star_50 + s_star_50) / 2 for a in alphas_plot]

    ax.fill_between(alphas_plot, fee_los, fee_his, alpha=0.25, color="C2",
                     label="Win-win zone")
    ax.plot(alphas_plot, fee_los, "-", color="C1", linewidth=2,
            label=r"Fee lower: $\alpha \cdot S^*(Q)$")
    ax.plot(alphas_plot, fee_his, "-", color="C0", linewidth=2,
            label=r"Fee upper: $S^*(Q)$")
    ax.plot(alphas_plot, nash_fees, "--", color="C4", linewidth=2,
            label="Nash bargaining fee")
    ax.set_xlabel(r"$\alpha_{tranche}$ (Chutes current share)", fontsize=12)
    ax.set_ylabel("Monthly subscription fee ($)", fontsize=12)
    ax.set_title(r"(b) Win-Win Zone vs Outside Option ($Q$=50%, $\Delta K$=0)",
                 fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Plot 3: alpha x Delta_K feasibility heatmap
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    pivot = sweep_df.pivot_table(
        index="alpha_tranche", columns="delta_k_frac",
        values="or_savings_pct", aggfunc="first"
    )

    # Create masked array for infeasible regions
    data = pivot.values.copy()
    mask = data <= 0
    data[mask] = np.nan

    im = ax.imshow(
        data, aspect="auto", origin="lower",
        extent=[pivot.columns.min() * 100, pivot.columns.max() * 100,
                pivot.index.min(), pivot.index.max()],
        cmap="YlGn", vmin=0, vmax=50,
    )

    # Draw critical alpha boundary
    dk_fracs_line = np.linspace(0, 0.5, 100)
    alpha_crits = [1.0 - dk for dk in dk_fracs_line]
    ax.plot(dk_fracs_line * 100, alpha_crits, "r-", linewidth=2.5,
            label=r"$\alpha_{crit} = 1 - \Delta K / S^*$")

    ax.set_xlabel(r"$\Delta K$ (% of $S^*(Q)$)", fontsize=12)
    ax.set_ylabel(r"$\alpha_{tranche}$", fontsize=12)
    ax.set_title("(c) Feasibility Map: OR Savings (%) at Nash Fee", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("OpenRouter savings (%)", fontsize=10)

    # Annotate infeasible region
    ax.text(35, 0.85, "No deal\n(infeasible)", fontsize=11, color="red",
            ha="center", va="center", fontweight="bold")
    ax.text(10, 0.15, "Large\nsurplus", fontsize=11, color="darkgreen",
            ha="center", va="center", fontweight="bold")

    # ------------------------------------------------------------------
    # Plot 4: Total surplus heatmap
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    pivot_surplus = sweep_df.pivot_table(
        index="alpha_tranche", columns="delta_k_frac",
        values="total_surplus", aggfunc="first"
    )

    surplus_data = pivot_surplus.values.copy()
    surplus_data[surplus_data <= 0] = np.nan
    surplus_pct = surplus_data / c_paygo * 100

    im2 = ax.imshow(
        surplus_pct, aspect="auto", origin="lower",
        extent=[pivot_surplus.columns.min() * 100, pivot_surplus.columns.max() * 100,
                pivot_surplus.index.min(), pivot_surplus.index.max()],
        cmap="YlOrRd_r", vmin=0, vmax=80,
    )

    ax.plot(dk_fracs_line * 100, alpha_crits, "r-", linewidth=2.5,
            label=r"$\alpha_{crit}$")
    ax.set_xlabel(r"$\Delta K$ (% of $S^*(Q)$)", fontsize=12)
    ax.set_ylabel(r"$\alpha_{tranche}$", fontsize=12)
    ax.set_title("(d) Total Surplus (% of paygo) Available to Split", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")

    cbar2 = plt.colorbar(im2, ax=ax)
    cbar2.set_label("Surplus (% of paygo)", fontsize=10)

    plt.tight_layout(pad=2.0)
    out_path = Path(output_dir) / "winwin_pricing_model.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {out_path}")


def main():
    data_dir = Path("/scratch/juncheng/data/prefix_cache/data/metrics_30day/per_model")
    output_dir = Path.home() / "hybridInference" / "experiment" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    V3_IN, V3_OUT = 0.20, 0.60

    # Load trace
    df = load_trace(str(data_dir / "large" / "DeepSeek-V3.1.csv"), V3_IN, V3_OUT)
    c_paygo = df["api_cost"].sum()
    n_days = df["date"].nunique()
    avg_daily_req = int(len(df) / n_days)

    # Compute S*(Q) for various quotas
    print("\nComputing S*(Q) for each quota level...")
    savings_table = compute_savings_table(df, avg_daily_req)

    # Sweep alpha x Delta_K at Q=50%
    s_star_50 = savings_table[0.5]["s_star"]
    print(f"\nSweeping alpha_tranche x Delta_K (Q=50%, S*=${s_star_50:,.0f})...")
    sweep_df = sweep_alpha_deltak(s_star_50, c_paygo)

    # Print summary
    print_model_summary(savings_table, c_paygo, avg_daily_req, n_days)
    print_stage1_gamma_bridge_table(savings_table, c_paygo)

    # Generate plots
    plot_all(savings_table, sweep_df, c_paygo, str(output_dir))

    # Save results
    out_path = output_dir / "winwin_model_results.json"
    output = {
        "model": "DeepSeek-V3.1",
        "c_paygo": float(c_paygo),
        "n_days": n_days,
        "avg_daily_requests": avg_daily_req,
        "total_requests": len(df),
        "savings_table": {str(k): v for k, v in savings_table.items()},
        "sweep_summary": {
            "s_star_50pct": float(s_star_50),
            "alpha_range": [0, 1],
            "delta_k_frac_range": [0, 0.5],
        },
    }
    with open(str(out_path), "w") as f:
        json.dump(output, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
