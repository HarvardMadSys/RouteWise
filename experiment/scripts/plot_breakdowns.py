"""Generate breakdown figures for the paper.

Figure 1: Cost routing breakdown - how many requests go to each tier (S_Q, S_C, S_A)
          for different algorithms on Stage 2 (FreeInference).

Figure 2: Latency provider distribution - which providers are selected by each policy
          in the 24-hour OpenRouter experiment (Qwen3-235B).
"""

import json
import csv
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Google style: no emoji, clean fonts
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "..", "6991d665791ce21ba05287b8", "images")
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# Figure 1: Cost routing breakdown (Stage 2, FreeInference)
# =============================================================================

def plot_cost_breakdown():
    """Bar chart showing request distribution across S_Q, S_C, S_A for each algorithm."""

    # Load Stage 2 online results (FreeInference, Q=5000, C=8)
    with open(os.path.join(BASE, "results", "online", "stage2_online_results.json")) as f:
        s2 = json.load(f)

    # Load ILP cache to get per-request tier breakdown for ILP-Optimal
    ilp_path = os.path.join(
        BASE, "cache", "ilp",
        "stage2_online_featherless_scale_n371214_d1.0_Q5000_C8_cbc_slo0_pe4f3b39d_s0ca43dae.json"
    )
    with open(ilp_path) as f:
        ilp_data = json.load(f)

    ilp_assignments = ilp_data["assignments"]
    ilp_counts = Counter(ilp_assignments.values())
    ilp_total = sum(ilp_counts.values())

    # Extract routing distributions
    algorithms = []

    # ILP-Optimal
    algorithms.append({
        "name": "ILP-Optimal",
        "sq": ilp_counts.get("daily", 0),
        "sc": ilp_counts.get("concurrency", 0),
        "sa": ilp_counts.get("api", 0),
        "api_cost": s2["Offline-Optimal-ILP"]["costs"]["api"],
    })

    # PD-EMA (PrimalDual-Online-Stage2)
    pd = s2["PrimalDual-Online-Stage2"]
    # PD does not have sq/sc split in the JSON; it only has subscription vs api.
    # But we know from the strategy_stats of LA-PD that it tracks sq_routed / sc_routed.
    # For PD-EMA, subscription = 362981, api = 21190 (but no sq/sc split).
    # Use the ratio from the ILP as an approximation? No -- let's check if there's
    # strategy_stats. There isn't for PD-EMA in this file.
    # We'll show what we have: subscription vs api for PD-EMA.
    algorithms.append({
        "name": "PD-EMA",
        "sq": None,  # not available
        "sc": None,
        "subscription": pd["requests"]["subscription"],
        "sa": pd["requests"]["api"],
        "api_cost": pd["costs"]["api"],
    })

    # LA-PD (has sq/sc split in strategy_stats)
    lapd = s2["LA-PD-Unified-Stage2"]
    stats = lapd.get("strategy_stats", {})
    algorithms.append({
        "name": "PD-Hist",
        "sq": stats.get("sq_routed", 0),
        "sc": stats.get("sc_routed", 0),
        "sa": stats.get("api_routed", 0),
        "api_cost": lapd["costs"]["api"],
    })

    # Greedy
    greedy = s2["Greedy-Online-Stage2"]
    algorithms.append({
        "name": "Greedy",
        "sq": None,
        "sc": None,
        "subscription": greedy["requests"]["subscription"],
        "sa": greedy["requests"]["api"],
        "api_cost": greedy["costs"]["api"],
    })

    # All-API
    allapi = s2["All-API"]
    algorithms.append({
        "name": "All-API",
        "sq": 0,
        "sc": 0,
        "sa": allapi["requests"]["api"],
        "api_cost": allapi["costs"]["api"],
    })

    # For algorithms without sq/sc split, show as single "Subscription" bar
    # We'll use a grouped bar chart with S_Q, S_C, S_A

    names = [a["name"] for a in algorithms]
    total = 371214
    x = np.arange(len(names))
    width = 0.6

    sq_pct = []
    sc_pct = []
    sa_pct = []
    for a in algorithms:
        if a.get("sq") is not None:
            sq_pct.append(a["sq"] / total * 100)
            sc_pct.append(a["sc"] / total * 100)
        else:
            sq_pct.append(a["subscription"] / total * 100)
            sc_pct.append(0)
        sa_pct.append(a["sa"] / total * 100)

    # --- Figure (a): request distribution (stacked bar) ---
    fig1, ax1 = plt.subplots(figsize=(4.5, 3.5))
    bars_sq = ax1.bar(x, sq_pct, width, label="$S_Q$ (quota)", color="#2196F3")
    bars_sc = ax1.bar(x, sc_pct, width, bottom=sq_pct, label="$S_C$ (concurrency)", color="#4CAF50")
    bottom2 = [sq + sc for sq, sc in zip(sq_pct, sc_pct)]
    bars_sa = ax1.bar(x, sa_pct, width, bottom=bottom2, label="$S_A$ (API)", color="#FF9800")

    ax1.set_ylabel("Requests (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.set_ylim(0, 105)

    for i, a in enumerate(algorithms):
        if a.get("sq") is None and a["name"] != "All-API":
            ax1.annotate("*", (x[i], 2), ha="center", fontsize=14, color="gray")

    plt.tight_layout()
    out1 = os.path.join(OUT_DIR, "cost_breakdown_tiers.png")
    plt.savefig(out1)
    plt.close()
    print(f"Saved: {out1}")

    # --- Figure (b): API overflow cost ---
    fig2, ax2 = plt.subplots(figsize=(4.5, 3.5))
    api_costs = [a["api_cost"] for a in algorithms]
    bars = ax2.bar(x, api_costs, width,
                   color=["#1976D2", "#42A5F5", "#64B5F6", "#FFA726", "#FF7043"])
    ax2.set_ylabel("API cost ($)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha="right")

    for bar, cost in zip(bars, api_costs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"${cost:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, "cost_breakdown_api_cost.png")
    plt.savefig(out2)
    plt.close()
    print(f"Saved: {out2}")

    # Print summary for paper text
    print("\n--- Cost Breakdown Summary (FreeInference, Stage 2) ---")
    for a in algorithms:
        if a.get("sq") is not None:
            print(f"  {a['name']:20s}: S_Q={a['sq']:6d} ({a['sq']/total*100:.1f}%), "
                  f"S_C={a['sc']:6d} ({a['sc']/total*100:.1f}%), "
                  f"S_A={a['sa']:6d} ({a['sa']/total*100:.1f}%), "
                  f"API cost=${a['api_cost']:.2f}")
        else:
            sub = a.get("subscription", 0)
            print(f"  {a['name']:20s}: Sub={sub:6d} ({sub/total*100:.1f}%), "
                  f"S_A={a['sa']:6d} ({a['sa']/total*100:.1f}%), "
                  f"API cost=${a['api_cost']:.2f}")


# =============================================================================
# Figure 2: Latency provider distribution (Qwen3-235B, 24-hour)
# =============================================================================

def plot_provider_distribution():
    """Horizontal stacked bar chart showing provider selection per policy."""

    log_path = os.path.join(
        BASE, "results", "phase5_qwen3_235b",
        "run_20260405_073250", "evaluation_log.csv"
    )

    # Parse CSV
    policy_providers = defaultdict(lambda: Counter())
    policy_total = Counter()

    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            policy = row["policy"]
            provider = row["actual_provider"]
            if not provider or row["status"] != "success":
                continue
            policy_providers[policy][provider] += 1
            policy_total[policy] += 1

    # Policies to show (matching paper Table 5)
    policy_display = {
        "openrouter_auto": "OpenRouter Auto",
        "sort_price": "sort=price",
        "sort_throughput": "sort=throughput",
        "sort_latency": "sort=latency",
        "cheapest_fixed": "Cheapest Fixed",
        "fastest_fixed": "Fastest Fixed",
        "lp_mix": "Ours: Latency-Aware",
        "smart_hedge": "Ours: Smart Hedge",
    }

    # Collect all providers, sorted by total usage
    all_providers = Counter()
    for policy in policy_display:
        if policy in policy_providers:
            all_providers.update(policy_providers[policy])

    # Top providers + "Other"
    top_n = 7
    top_providers = [p for p, _ in all_providers.most_common(top_n)]

    # Build data matrix
    policies = list(policy_display.keys())
    data = {}
    for policy in policies:
        if policy not in policy_providers:
            continue
        total = policy_total[policy]
        row = {}
        other = 0
        for provider, count in policy_providers[policy].items():
            if provider in top_providers:
                row[provider] = count / total * 100
            else:
                other += count / total * 100
        row["Other"] = other
        data[policy] = row

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4.5))

    categories = top_providers + ["Other"]
    # Color palette (colorblind-friendly)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#cccccc"]

    y_pos = np.arange(len(data))
    y_labels = []
    left = np.zeros(len(data))

    for idx, cat in enumerate(categories):
        widths = []
        for policy in policies:
            if policy in data:
                widths.append(data[policy].get(cat, 0))
        if not widths:
            continue
        widths = np.array(widths)
        ax.barh(y_pos, widths, left=left, height=0.6,
                label=cat, color=colors[idx % len(colors)])
        left += widths

    for policy in policies:
        if policy in data:
            y_labels.append(policy_display[policy])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Requests (%)")
    ax.set_xlim(0, 100)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.set_title("Provider distribution by routing policy (Qwen3-235B, 24h)")
    ax.invert_yaxis()

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "provider_distribution.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

    # Print summary
    print("\n--- Provider Distribution Summary ---")
    for policy in policies:
        if policy not in data:
            continue
        print(f"\n  {policy_display[policy]}:")
        total = policy_total[policy]
        sorted_providers = sorted(
            policy_providers[policy].items(), key=lambda x: -x[1]
        )
        for prov, count in sorted_providers[:5]:
            print(f"    {prov:20s}: {count:5d} ({count/total*100:.1f}%)")


if __name__ == "__main__":
    plot_cost_breakdown()
    print("\n" + "=" * 60)
    plot_provider_distribution()
