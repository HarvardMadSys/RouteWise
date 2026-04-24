"""Alpha-weighted LP sweep on the sanity-check scenarios (Step 5 focus).

Purpose
-------
The sanity-check step 5 sweep (A's P50 from 100-500 ms, B fixed at 100 ms)
gives us scenarios with analytically obvious ground truth. This script
re-runs step 5 with the alpha-weighted LP at several alpha values to
validate that:

    1. alpha = 0        -> reproduces lp_mix/lp_explorer (100% A).
    2. alpha moderate   -> smooth transition to B as P50(A) grows.
    3. alpha large      -> matches V2's intended "pick the faster provider"
                           behavior (but without V2's measurement noise).

Output
------
outputs/alpha_on_sanity/step5/
    summary.json
    a_share_vs_p50.png        -- one curve per alpha, x = P50(A)
    pareto_cost_vs_p99.png    -- one curve per alpha aggregating all P50 values
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.synthetic_latency.sanity import step5_sweep  # noqa: E402
from rwsim.world import generate_workload  # noqa: E402

# Reuse the helpers we already set up in the alpha sweep runner.
from scripts.experiments.run_alpha_sweep import run_lp_alpha, summarize_runs  # noqa: E402

SEEDS = [42, 43, 44]
N_REQUESTS = 1000
DURATION_SECONDS = 3600.0
SLO_SEC = 2.0

ALPHAS = [0.0, 1e-7, 1e-6, 2e-6, 3e-6, 5e-6, 1e-5, 3e-5, 1e-4]

OUTPUT_DIR = _ROOT / "outputs" / "alpha_on_sanity" / "step5"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = step5_sweep()

    # Each entry keyed by (alpha, p50_a) -> summary dict.
    results: dict[tuple[float, float], dict] = {}
    requests = generate_workload(
        n_requests=N_REQUESTS,
        duration_seconds=DURATION_SECONDS,
        seed=0,
        start_time=0.0,
        arrival_process="poisson",
    )

    for scenario in scenarios:
        # A's P50 is the first provider's true P50
        p50_a = scenario.providers[0].true_p50_ms()

        print(f"\nScenario: {scenario.name} (P50_A = {p50_a:.0f}ms)")
        for alpha in ALPHAS:
            runs = []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                run = run_lp_alpha(
                    scenario=scenario,
                    requests=requests,
                    rng=rng,
                    slo_sec=SLO_SEC,
                    alpha=alpha,
                )
                runs.append(run)
            summary = summarize_runs(alpha, runs, scenario)
            summary["p50_a_ms"] = float(p50_a)
            results[(alpha, p50_a)] = summary
            providers_str = ", ".join(
                f"{k}={v:.0%}" for k, v in summary["provider_fractions"].items()
            )
            print(
                f"  alpha={alpha:.0e}  cost={summary['mean_cost_usd']:.2e}  "
                f"P50={summary['p50_ms']:.0f}ms  P99={summary['p99_ms']:.0f}ms  "
                f"[{providers_str}]"
            )

    # Save JSON
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(
            {
                "alphas": ALPHAS,
                "p50_a_values": sorted({p for _, p in results.keys()}),
                "entries": [
                    {"alpha": a, "p50_a_ms": p, **v}
                    for (a, p), v in sorted(results.items(), key=lambda kv: (kv[0][0], kv[0][1]))
                ],
            },
            f,
            indent=2,
        )

    _plot_a_share_vs_p50(results)
    _plot_pareto(results)
    print(f"\nSaved to {OUTPUT_DIR}/")


def _plot_a_share_vs_p50(results):
    """x = P50(A) swept; one curve per alpha; y = share of A."""
    import matplotlib.pyplot as plt

    alphas = sorted({a for a, _ in results.keys()})
    p50_values = sorted({p for _, p in results.keys()})

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=130)
    colormap = plt.cm.viridis(np.linspace(0, 1, len(alphas)))
    for alpha, color in zip(alphas, colormap):
        shares = [
            results[(alpha, p)]["provider_fractions"].get("A", 0.0)
            for p in p50_values
        ]
        label = f"α=0" if alpha == 0 else f"α={alpha:.0e}"
        ax.plot(p50_values, shares, marker="o", color=color, linewidth=1.8, label=label)

    ax.axvline(x=110, color="gray", linestyle="--", alpha=0.5, label="V2 band edge")
    ax.set_xlabel("P50(A) in ms  (B fixed at P50=100ms)", fontsize=12)
    ax.set_ylabel("Share of traffic to A", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "a_share_vs_p50.png")
    plt.close(fig)


def _plot_pareto(results):
    """Aggregate across all P50(A) values; one point per alpha showing mean cost + P99."""
    import matplotlib.pyplot as plt

    alphas = sorted({a for a, _ in results.keys()})
    p50_values = sorted({p for _, p in results.keys()})

    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
    colormap = plt.cm.viridis(np.linspace(0, 1, len(alphas)))
    for alpha, color in zip(alphas, colormap):
        costs = [results[(alpha, p)]["mean_cost_usd"] for p in p50_values]
        p99s = [results[(alpha, p)]["p99_ms"] for p in p50_values]
        label = "α=0" if alpha == 0 else f"α={alpha:.0e}"
        ax.plot(costs, p99s, marker="o", color=color, linewidth=1.5, label=label, alpha=0.8)

    ax.set_xlabel("Mean cost per request (USD)", fontsize=12)
    ax.set_ylabel("P99 TTFT (ms)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pareto_cost_vs_p99.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
