"""Alpha-weighted LP prototype sweep (Juncheng 2026-04-21 proposal).

Question to answer:
    Can we drop V2 by extending lp_mix's objective to include a latency term?

    Proposed formulation:
        min  sum_j pi_j * (c_j + alpha * p50_j)
        s.t. sum_j pi_j * F_j(SLO) >= 0.99, sum pi_j = 1, pi_j >= 0

Validation criteria:
    1. alpha = 0 reproduces lp_mix / lp_hedge behavior (cost-heavy mix).
    2. Large alpha collapses toward the fastest provider (V2-like behavior).
    3. Intermediate alpha traces a continuous Pareto frontier in (cost, P99).

Usage (from routewise-simulator/ root with .venv active):
    python run_alpha_sweep.py

Output:
    results/alpha_sweep/
        s2_tradeoff/
            summary.json
            pareto_cost_vs_p99.png
            provider_mix_vs_alpha.png
        s3_tail/
            summary.json
            pareto_cost_vs_p99.png
            provider_mix_vs_alpha.png
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from legacy.experiment.data.schema import Request  # noqa: E402
from legacy.experiment.scripts.simulate.synthetic.runner import (  # noqa: E402
    StrategyRun,
    _cheapest_provider_name,
    _costs_dict,
    _make_hedger,
    _probe_providers,
    _sample,
    _warm_up_router,
)
from legacy.experiment.scripts.simulate.synthetic.scenarios import (  # noqa: E402
    ScenarioConfig,
    make_scenarios,
)
from legacy.experiment.scripts.simulate.synthetic.workload import generate_workload  # noqa: E402
from legacy.experiment.strategies.online_latency_router import OnlineLatencyRouter  # noqa: E402
from legacy.experiment.strategies.smart_hedging import (  # noqa: E402
    BackupSelectionMethod,
    select_backup,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Alpha sweep. Units are USD per ms, matching the cost dict (USD per request)
# and P50 (ms). The natural middle range turns out to be around 1e-7..1e-5
# because a typical 200-token request on a $1/M provider costs 2e-4 USD while
# a typical P50 is ~200 ms, so the crossover happens around alpha ~ 1e-6.
ALPHAS = [
    0.0,      # pure cost (reproduces lp_mix)
    1e-8,     # negligible latency weight
    1e-7,
    3e-7,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,     # latency dominates (reproduces V2-like)
    1e-3,
]

SEEDS = [42, 43, 44]
N_REQUESTS = 2000
DURATION_SECONDS = 3600.0
SLO_SEC = 2.0
OUTPUT_ROOT = _ROOT / "results" / "alpha_sweep"

# Which synthetic scenarios to run the sweep on. S2 (cost-latency tradeoff)
# and S3 (tail-heavy) are the most informative; S1 (clear winner) is a sanity
# check.
SCENARIO_IDS = ["s1_dominant", "s2_tradeoff", "s3_tail"]


# ---------------------------------------------------------------------------
# LP+Hedge+Explorer run with alpha (clone of _run_lp_hedge with alpha added)
# ---------------------------------------------------------------------------


def run_lp_alpha(
    scenario: ScenarioConfig,
    requests: list[Request],
    rng: np.random.Generator,
    slo_sec: float,
    alpha: float,
    hedge_as_probe: bool = True,
    probe_rate: float | None = None,
) -> StrategyRun:
    """LP+hedge+explorer with alpha-weighted cost objective.

    When alpha=0 and hedge_as_probe=True, this matches the standard
    ``lp_explorer`` strategy. For alpha>0 the LP objective adds a term
    alpha * p50_j per provider, biasing toward faster providers.
    """
    costs = _costs_dict(scenario)
    pdict = {p.name: p for p in scenario.providers}

    router = OnlineLatencyRouter(
        costs=costs,
        slo_sec=slo_sec,
        lp_update_interval=60.0,
        alpha=alpha,
    )
    hedger = _make_hedger(costs, slo_sec)

    t0 = float(requests[0].timestamp) if requests else 0.0
    _warm_up_router(router, scenario, t0, rng)
    fallback_name = _cheapest_provider_name(scenario)

    ttft_ms, cost_usd, provider_sel, timestamps = [], [], [], []
    hedged_flags: list[bool] = []

    for req in requests:
        t = float(req.timestamp)
        primary_name = router.route(t)
        if primary_name is None:
            primary_name = fallback_name

        backup_name = select_backup(
            BackupSelectionMethod.FASTEST,
            router.profiles,
            costs,
            primary_name,
            slo_sec,
            t,
        )
        if backup_name is None:
            backup_name = primary_name

        p_primary = pdict[primary_name]
        T_primary_ms, _ = _sample(p_primary, req.response_tokens, rng, t)

        p_backup = pdict[backup_name]
        T_backup_ms, _ = _sample(p_backup, req.response_tokens, rng, t)

        result = hedger.simulate_request(
            primary=primary_name,
            profiles=router.profiles,
            now=t,
            T_primary_sec=T_primary_ms / 1000.0,
            err_primary=None,
            T_backup_sec=T_backup_ms / 1000.0,
            err_backup=None,
            backup=backup_name,
        )
        final_ttft_ms = result.final_ttft_sec * 1000.0

        router.add_sample(primary_name, t, T_primary_ms)
        if hedge_as_probe and result.hedged and backup_name != primary_name:
            router.add_sample(backup_name, t, T_backup_ms)
        _probe_providers(router, scenario, primary_name, t, rng, probe_rate=probe_rate)

        c = p_primary.cost_per_token * req.total_tokens
        if result.hedged:
            c += p_backup.cost_per_token * req.total_tokens

        ttft_ms.append(final_ttft_ms)
        cost_usd.append(c)
        provider_sel.append(primary_name)
        timestamps.append(t)
        hedged_flags.append(result.hedged)

    return StrategyRun(
        strategy=f"lp_alpha_{alpha:.0e}",
        ttft_ms=np.array(ttft_ms),
        cost_usd=np.array(cost_usd),
        provider=provider_sel,
        timestamp=np.array(timestamps),
        hedge_triggered=np.array(hedged_flags, dtype=bool),
    )


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _avg(runs: list[StrategyRun], fn) -> float:
    return float(np.mean([fn(r) for r in runs]))


def summarize_runs(
    alpha: float,
    runs: list[StrategyRun],
    scenario: ScenarioConfig,
) -> dict:
    entry: dict = {"alpha": alpha}
    for slo in scenario.slo_thresholds_ms:
        entry[f"slo_violation_rate_{int(slo)}ms"] = _avg(
            runs, lambda r, s=slo: r.slo_violation_rate(s)
        )
    entry["mean_cost_usd"] = _avg(runs, lambda r: r.mean_cost_usd())
    entry["p50_ms"] = _avg(runs, lambda r: r.p50_ms())
    entry["p99_ms"] = _avg(runs, lambda r: r.p99_ms())
    entry["hedge_rate"] = _avg(runs, lambda r: r.hedge_rate())

    frac_lists: dict[str, list[float]] = {}
    for r in runs:
        for pname, frac in r.provider_fractions().items():
            frac_lists.setdefault(pname, []).append(frac)
    entry["provider_fractions"] = {
        p: float(np.mean(fracs)) for p, fracs in sorted(frac_lists.items())
    }
    return entry


# ---------------------------------------------------------------------------
# Plotting (one PNG per figure, no subplots — project convention)
# ---------------------------------------------------------------------------


def plot_pareto(
    scenario_id: str,
    summaries: list[dict],
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    alphas = [s["alpha"] for s in summaries]
    costs = [s["mean_cost_usd"] for s in summaries]
    p99s = [s["p99_ms"] for s in summaries]
    p50s = [s["p50_ms"] for s in summaries]

    # Figure 1: Pareto frontier (cost vs P99)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
    ax.plot(costs, p99s, marker="o", linewidth=1.8, color="tab:green", label="alpha sweep")
    for a, c, p in zip(alphas, costs, p99s):
        ax.annotate(
            f"α={a:.0e}" if a > 0 else "α=0",
            xy=(c, p),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8,
        )
    ax.set_xlabel("Mean cost per request (USD)", fontsize=12)
    ax.set_ylabel("P99 TTFT (ms)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_cost_vs_p99.png")
    plt.close(fig)

    # Figure 2: Cost vs P50 (secondary view)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
    ax.plot(costs, p50s, marker="s", linewidth=1.8, color="tab:blue", label="alpha sweep")
    for a, c, p in zip(alphas, costs, p50s):
        ax.annotate(
            f"α={a:.0e}" if a > 0 else "α=0",
            xy=(c, p),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8,
        )
    ax.set_xlabel("Mean cost per request (USD)", fontsize=12)
    ax.set_ylabel("P50 TTFT (ms)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_cost_vs_p50.png")
    plt.close(fig)

    # Figure 3: Provider mix vs alpha
    all_providers = sorted({
        p for s in summaries for p in s["provider_fractions"]
    })
    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)

    # Plot on log-scale x axis; replace alpha=0 with a tiny placeholder so
    # it renders on the log axis without throwing.
    alphas_plot = [max(a, 1e-10) for a in alphas]
    for pname in all_providers:
        fracs = [s["provider_fractions"].get(pname, 0.0) for s in summaries]
        ax.plot(alphas_plot, fracs, marker="o", linewidth=1.8, label=pname)
    ax.set_xscale("log")
    ax.set_xlabel("alpha (latency weight)", fontsize=12)
    ax.set_ylabel("Provider traffic share", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "provider_mix_vs_alpha.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    scenarios = make_scenarios()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for scenario_id in SCENARIO_IDS:
        scenario = scenarios[scenario_id]
        print(f"\n{'=' * 60}")
        print(f"Scenario: {scenario_id}")
        print(f"  {scenario.description}")

        out_dir = OUTPUT_ROOT / scenario_id
        out_dir.mkdir(parents=True, exist_ok=True)

        requests = generate_workload(
            n_requests=N_REQUESTS,
            duration_seconds=DURATION_SECONDS,
            seed=0,
            start_time=0.0,
            arrival_process="poisson",
        )

        alpha_summaries: list[dict] = []
        for alpha in ALPHAS:
            t0 = time.perf_counter()
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
            elapsed = time.perf_counter() - t0

            summary = summarize_runs(alpha, runs, scenario)
            alpha_summaries.append(summary)

            providers_str = ", ".join(
                f"{k}={v:.0%}" for k, v in summary["provider_fractions"].items()
            )
            print(
                f"  alpha={alpha:.0e}  "
                f"cost={summary['mean_cost_usd']:.3e}  "
                f"P50={summary['p50_ms']:.0f}ms  "
                f"P99={summary['p99_ms']:.0f}ms  "
                f"hedge={summary['hedge_rate']:.1%}  "
                f"[{providers_str}]  ({elapsed:.1f}s)"
            )

        with open(out_dir / "summary.json", "w") as f:
            json.dump({"scenario": scenario_id, "alphas": alpha_summaries}, f, indent=2)

        plot_pareto(scenario_id, alpha_summaries, out_dir)
        print(f"  Saved summary + 3 PNGs to {out_dir}/")

    print(f"\nDone. Results in {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
