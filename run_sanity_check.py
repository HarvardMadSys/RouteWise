"""Orchestrator for the Juncheng sanity-check scenarios.

Usage (from routewise-simulator/ project root, with .venv active):
    python run_sanity_check.py

Output layout:
    results/sanity_check/
        step1_clear_winner/
            summary.json           # all strategies x single scenario
        step2_cost_converge/
            summary.json           # all strategies x 5 cost values
            responsiveness.png     # x=a_cost, y=A_share / cost / P99
        step3_identical/
            summary.json
        step4_band/
            summary.json
        step5_latency_sweep/
            summary.json           # all strategies x 15 P50 values
            responsiveness.png     # MAIN DELIVERABLE
            ground_truth_check.md  # auto-generated pass/fail report

Placement
---------
This file should live at the project root:
    routewise-simulator/run_sanity_check.py

It imports sanity_check from legacy.experiment.scripts.simulate.synthetic, so
sanity_check.py must already exist at
    routewise-simulator/experiment/scripts/simulate/synthetic/sanity_check.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# Add project root so experiment.* imports resolve.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from legacy.experiment.scripts.simulate.synthetic.runner import (  # noqa: E402
    STRATEGIES,
    StrategyRun,
    run_strategy,
)
from legacy.experiment.scripts.simulate.synthetic.sanity_check import (  # noqa: E402
    SanityStep,
    make_sanity_steps,
)
from legacy.experiment.scripts.simulate.synthetic.scenarios import ScenarioConfig  # noqa: E402
from legacy.experiment.scripts.simulate.synthetic.workload import generate_workload  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEEDS = [42, 43, 44]
OUTPUT_ROOT = _ROOT / "results" / "sanity_check"

# Keep scenarios small so the whole sweep runs in a few minutes.
N_REQUESTS = 1000
DURATION_SECONDS = 3600.0

# The set of policies we actually care about for the plots. The runner
# supports more, but too many lines makes the plot unreadable.
PLOT_STRATEGIES = [
    "cheapest_fixed",
    "fastest_fixed",
    "round_robin",
    "lp_mix",
    "lp_hedge",
    "lp_explorer",
    "v2_only",
    "v2_p50_hedge",
    "v2_explorer",
]


# ---------------------------------------------------------------------------
# Summary builders
# ---------------------------------------------------------------------------


def _avg(runs: list[StrategyRun], fn) -> float:
    return float(np.mean([fn(r) for r in runs]))


def build_scenario_summary(
    scenario: ScenarioConfig,
    results: dict[str, list[StrategyRun]],
) -> dict:
    """Reduce per-strategy multi-seed runs to a single-scenario summary."""
    summary: dict = {
        "scenario_name": scenario.name,
        "description": scenario.description,
        "providers": {
            p.name: {
                "cost_per_m_tokens": p.cost_per_token * 1e6,
                "p50_ms": p.true_p50_ms(),
                "p99_ms": p.true_p99_ms(),
            }
            for p in scenario.providers
        },
        "strategies": {},
    }
    for strat, runs in results.items():
        entry: dict = {}
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
        summary["strategies"][strat] = entry
    return summary


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_one_scenario(scenario: ScenarioConfig) -> dict[str, list[StrategyRun]]:
    """Run all strategies x seeds on a single scenario."""
    requests = generate_workload(
        n_requests=N_REQUESTS,
        duration_seconds=DURATION_SECONDS,
        seed=0,
        start_time=0.0,
        arrival_process="poisson",
    )

    results: dict[str, list[StrategyRun]] = {s: [] for s in STRATEGIES}
    for strategy in STRATEGIES:
        for seed in SEEDS:
            run = run_strategy(scenario, requests, strategy, seed=seed)
            results[strategy].append(run)
    return results


def run_step(step: SanityStep, out_dir: Path) -> list[dict]:
    """Run a full sanity step (possibly multiple scenarios)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    step_summaries: list[dict] = []

    for scenario in step.scenarios:
        print(f"  Scenario: {scenario.name}")
        t0 = time.perf_counter()
        results = run_one_scenario(scenario)
        elapsed = time.perf_counter() - t0

        summary = build_scenario_summary(scenario, results)
        step_summaries.append(summary)

        # Brief per-scenario print
        for strat in PLOT_STRATEGIES:
            if strat not in summary["strategies"]:
                continue
            entry = summary["strategies"][strat]
            providers_str = ", ".join(
                f"{k}={v:.0%}" for k, v in entry["provider_fractions"].items()
            )
            print(
                f"    {strat:<22s} SLO@2s={entry['slo_violation_rate_2000ms']:.1%} "
                f"cost={entry['mean_cost_usd']:.2e} "
                f"[{providers_str}]"
            )
        print(f"    (elapsed {elapsed:.1f}s)")

    # Save combined summary for the step
    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {
                "step_id": step.step_id,
                "description": step.description,
                "is_sweep": step.is_sweep,
                "scenarios": step_summaries,
            },
            f,
            indent=2,
        )
    return step_summaries


# ---------------------------------------------------------------------------
# Plotting (step 5 is the critical one)
# ---------------------------------------------------------------------------


def plot_responsiveness(step_summaries: list[dict], out_path: Path, x_label: str) -> None:
    """Plot A-share / cost / P99 vs swept parameter for all strategies.

    Per project convention: one PNG per figure (no subplots), so we
    actually emit three PNGs with a common filename prefix.
    """
    import matplotlib.pyplot as plt

    x_values = []
    for s in step_summaries:
        # Pull the swept parameter out of the first provider's record.
        # For step 2 (cost sweep): A's cost. For step 5: A's P50.
        pa = s["providers"].get("A")
        if pa is None:
            # Identical-scenario step: nothing to sweep.
            return
        x_values.append(pa)
    # Decide which axis to extract based on file name hint in x_label.
    if "cost" in x_label.lower():
        x = [pa["cost_per_m_tokens"] for pa in x_values]
    else:
        x = [pa["p50_ms"] for pa in x_values]

    def _extract(strat: str, field: str) -> list[float]:
        out = []
        for s in step_summaries:
            entry = s["strategies"].get(strat)
            if entry is None:
                out.append(float("nan"))
                continue
            if field == "a_share":
                out.append(entry["provider_fractions"].get("A", 0.0))
            elif field == "cost":
                out.append(entry["mean_cost_usd"])
            elif field == "p99":
                out.append(entry["p99_ms"])
        return out

    prefix = out_path.with_suffix("")  # strip .png; we'll append suffixes below

    for field, ylabel, title in [
        ("a_share", "Share of traffic to A", "Provider-A share"),
        ("cost", "Mean cost per request (USD)", "Cost"),
        ("p99", "P99 TTFT (ms)", "P99 latency"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
        for strat in PLOT_STRATEGIES:
            y = _extract(strat, field)
            ax.plot(x, y, marker="o", label=strat, linewidth=1.5)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        out_file = Path(f"{prefix}_{field}.png")
        fig.savefig(out_file)
        plt.close(fig)
        print(f"    Saved {out_file}")


def generate_ground_truth_report(step5_summaries: list[dict], out_path: Path) -> None:
    """Emit a pass/fail report against the expected behavior in step 5.

    Checks:
      1. A-share monotonically non-increasing in A's P50 for cost-aware policies.
      2. V2 share of A drops to <= 20% once A's P50 > 110ms (band boundary).
      3. LP share of A has no discontinuity > 40% between adjacent points.

    Writes a short Markdown report with pass/fail flags.
    """
    lines = ["# Step 5 Ground-Truth Check", ""]
    xs = [s["providers"]["A"]["p50_ms"] for s in step5_summaries]

    def a_share(strat: str) -> list[float]:
        return [
            s["strategies"][strat]["provider_fractions"].get("A", 0.0)
            for s in step5_summaries
        ]

    # Check 1: monotonicity for cost-aware strategies
    lines.append("## Check 1: A-share monotonic non-increasing")
    for strat in ["cheapest_fixed", "lp_mix", "lp_hedge", "lp_explorer",
                  "v2_only", "v2_p50_hedge", "v2_explorer"]:
        shares = a_share(strat)
        violations = sum(
            1
            for a, b in zip(shares[:-1], shares[1:])
            if b > a + 0.10  # allow 10pp tolerance for noise
        )
        flag = "PASS" if violations == 0 else f"FAIL ({violations} violations)"
        lines.append(f"- {strat}: {flag}")
    lines.append("")

    # Check 2: V2 band boundary
    lines.append("## Check 2: V2 drops A after band boundary (P50 > 110ms)")
    for strat in ["v2_only", "v2_p50_hedge", "v2_explorer"]:
        shares = a_share(strat)
        post_boundary = [s for x, s in zip(xs, shares) if x > 110.0]
        max_post = max(post_boundary, default=0.0)
        flag = "PASS" if max_post <= 0.20 else f"FAIL (max A-share={max_post:.1%})"
        lines.append(f"- {strat}: {flag}")
    lines.append("")

    # Check 3: LP smoothness
    lines.append("## Check 3: LP A-share has no discontinuity > 40pp")
    for strat in ["lp_mix", "lp_hedge", "lp_explorer"]:
        shares = a_share(strat)
        biggest_jump = max(
            (abs(a - b) for a, b in zip(shares[:-1], shares[1:])),
            default=0.0,
        )
        flag = "PASS" if biggest_jump <= 0.40 else f"FAIL (max jump={biggest_jump:.1%})"
        lines.append(f"- {strat}: {flag}")
    lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    steps = make_sanity_steps()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for step in steps:
        print(f"\n{'=' * 60}")
        print(f"Step: {step.step_id}")
        print(f"  {step.description}")

        step_dir = OUTPUT_ROOT / step.step_id
        summaries = run_step(step, step_dir)

        # Emit responsiveness plot for sweeps only
        if step.is_sweep and len(summaries) >= 3:
            x_label = (
                "A cost ($/M tokens)"
                if step.step_id == "step2_cost_converge"
                else "A P50 (ms)"
            )
            plot_responsiveness(summaries, step_dir / "responsiveness.png", x_label)

        # Ground-truth auto-check for step 5 only
        if step.step_id == "step5_latency_sweep":
            generate_ground_truth_report(
                summaries, step_dir / "ground_truth_check.md"
            )

    print(f"\nDone. Results in {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
