"""Generate publication-quality latency figures for ICML paper."""

import contextlib
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from experiment.scripts.plot.common import apply_style, compute_ecdf

csv.field_size_limit(sys.maxsize)

# Apply publication style
apply_style("paper")

OUTPUT_DIR = Path("ICML2026_HybridInference/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_chutes_data(csv_path: str) -> np.ndarray:
    """Load latency data from Chutes provider CSV."""
    latencies = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] == "success":
                latencies.append(float(row["latency_ms"]) / 1000.0)
    return np.array(latencies)


def load_sglang_data(csv_path: str, model: str = "glm-4.6") -> np.ndarray:
    """Load latency data from SGLang local GPU logs."""
    latencies = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["model_id"] == model and row["provider"] == "sglang":
                latency_ms = row.get("latency_ms", "")
                if latency_ms and latency_ms.strip():
                    with contextlib.suppress(ValueError):
                        latencies.append(float(latency_ms) / 1000.0)
    return np.array(latencies)


def solve_lp_routing(profiles, slo, target_pct=99.0):
    """Solve LP for optimal routing mix given SLO constraint."""
    n = len(profiles)
    c = np.array([p["cost"] for p in profiles])
    A_ub = np.array([[-np.mean(p["latencies"] <= slo) for p in profiles]])
    b_ub = np.array([-target_pct / 100.0])
    A_eq = np.ones((1, n))
    b_eq = np.array([1.0])
    bounds = [(0, 1) for _ in range(n)]

    result = optimize.linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )
    if result.success:
        return result.x, result.fun
    return None, np.inf


def main():
    """Generate latency Pareto frontier figure for ICML paper."""
    # Load data
    chutes_latencies = load_chutes_data(
        "experiment/results/latency/raw/latency_chutes_20260121_125109.csv"
    )
    sglang_latencies = load_sglang_data("data/freeinference_logs.csv", "glm-4.6")

    print(f"Local GPU: {len(sglang_latencies)} samples")
    print(f"Chutes: {len(chutes_latencies)} samples")

    profiles = [
        {"name": "Local GPU", "latencies": sglang_latencies, "cost": 0.0, "color": "#2ecc71"},
        {
            "name": "Subscription",
            "latencies": chutes_latencies,
            "cost": 0.00013,
            "color": "#e74c3c",
        },
    ]

    # Compute Pareto frontier
    max_p99 = max(np.percentile(p["latencies"], 99) for p in profiles)
    slos = np.linspace(5.0, max_p99 * 1.1, 200)
    pareto = []
    for slo in slos:
        pi, cost = solve_lp_routing(profiles, slo)
        if pi is not None:
            pareto.append((slo, cost, pi))

    # === Figure: Combined 3-panel figure ===
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    # Panel (a): CDF comparison
    ax1 = axes[0]
    for p in profiles:
        sorted_vals, ecdf = compute_ecdf(p["latencies"])
        ax1.plot(
            sorted_vals,
            ecdf,
            color=p["color"],
            linewidth=2,
            label=f"{p['name']} (n={len(p['latencies'])})",
        )

    ax1.axvline(x=50, color="gray", linestyle="--", alpha=0.7, linewidth=1)
    ax1.text(51, 0.5, "SLO", fontsize=8, color="gray")

    ax1.set_xlabel("Latency (seconds)")
    ax1.set_ylabel("CDF")
    ax1.set_title("(a) Latency Distribution")
    ax1.legend(loc="lower right", framealpha=0.9)
    ax1.set_xlim(0, 70)
    ax1.set_ylim(0, 1.02)

    # Panel (b): Pareto frontier
    ax2 = axes[1]
    if pareto:
        p_slos = [p[0] for p in pareto]
        p_costs = [p[1] * 1000 for p in pareto]  # $/1000 requests
        ax2.plot(p_slos, p_costs, "b-", linewidth=2.5, label="LP Routing")

        # Baseline lines
        for p in profiles:
            ax2.axhline(
                y=p["cost"] * 1000,
                color=p["color"],
                linestyle="--",
                alpha=0.7,
                linewidth=1.5,
                label=f"{p['name']} only",
            )

    ax2.set_xlabel("Latency SLO (P99, seconds)")
    ax2.set_ylabel("Cost ($/1000 req)")
    ax2.set_title("(b) Cost-Latency Pareto Frontier")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.set_xlim(48, 65)

    # Panel (c): Routing mix
    ax3 = axes[2]
    if pareto:
        routing_matrix = np.array([p[2] for p in pareto])
        ax3.stackplot(
            p_slos,
            routing_matrix.T,
            labels=[p["name"] for p in profiles],
            colors=[p["color"] for p in profiles],
            alpha=0.8,
        )
        ax3.axvline(x=50, color="gray", linestyle="--", alpha=0.7, linewidth=1)
        ax3.axvline(x=55, color="gray", linestyle="--", alpha=0.7, linewidth=1)

    ax3.set_xlabel("Latency SLO (P99, seconds)")
    ax3.set_ylabel("Routing Fraction")
    ax3.set_title("(c) Optimal Provider Mix")
    ax3.legend(loc="upper right", framealpha=0.9)
    ax3.set_xlim(48, 65)
    ax3.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "latency_pareto.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'latency_pareto.png'}")

    # Print statistics for paper
    print("\n" + "=" * 60)
    print("Statistics for Paper")
    print("=" * 60)
    for p in profiles:
        lat = p["latencies"]
        print(f"\n{p['name']}:")
        print(f"  N = {len(lat)}")
        print(f"  P50 = {np.percentile(lat, 50):.1f}s")
        print(f"  P90 = {np.percentile(lat, 90):.1f}s")
        print(f"  P99 = {np.percentile(lat, 99):.1f}s")
        print(f"  Cost = ${p['cost']*1000:.2f}/1000 req")

    print("\n" + "=" * 60)
    print("Key SLO Points")
    print("=" * 60)
    for target in [50, 52, 55, 60]:
        for slo, cost, pi in pareto:
            if abs(slo - target) < 0.5:
                print(f"\nSLO = {target}s:")
                print(f"  Cost = ${cost*1000:.3f}/1000 req")
                for i, frac in enumerate(pi):
                    if frac > 0.01:
                        print(f"  {profiles[i]['name']}: {frac*100:.0f}%")
                break


if __name__ == "__main__":
    main()
