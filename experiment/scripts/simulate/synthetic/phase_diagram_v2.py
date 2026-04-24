"""LP vs V2 phase diagram with realistic provider tail structure.

Fixes a limitation of phase_diagram.py: that sweep gave providers B and C
the same tail structure (P99 = 2*P50), so whenever A was clearly fastest,
LP and V2 degenerated to the same choice — most cells collapsed.

This v2 sweep uses a *Llama-like* provider structure:
  - A: fastest P50, variable tail (the Y axis: tail asymmetry)
  - B: moderately slower P50 (set via p50_spread_ab), BEST tail (P99 = 1.3*P50)
  - C: slow P50, moderate tail (P99 = 2*P50)

This structure matches what we actually see in OpenRouter: one provider
is fastest-P50 but has a non-trivial tail, another is slower but stable.
V2's "pick fastest P50" then pays the tail cost; LP's CDF constraint can
diversify to B (slower-P50-but-stable).

Axes of the sweep:
  X: p50_spread_ab = P50_B / P50_A (how much slower B is than A)
     {1.1, 1.3, 1.5, 2.0, 3.0}
  Y: tail_ratio_a = P99_A / P50_A (A's tail asymmetry)
     {1.5, 3.0, 5.0, 8.0, 15.0}

C is always at P50 = P50_A * 3.0 with P99 = P50_C * 2.0, acting as
"slow-but-cheap fallback".

All three providers have the same cost (1e-6 USD/token) to isolate
latency-only trade-offs.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from .providers import LogNormal, SyntheticProvider
from .scenarios import ScenarioConfig
from .runner import run_strategy


# P50 spread between A (fastest) and B (stable backup).
P50_SPREAD_AB = [1.1, 1.3, 1.5, 2.0, 3.0]

# Tail asymmetry of provider A (the fastest).
TAIL_RATIO_A = [1.5, 3.0, 5.0, 8.0, 15.0]

BASE_P50_MS = 200.0
UNIFORM_COST = 1.0e-6

# Strategies to compare in this sweep.
STRATEGIES = ["lp_explorer", "v2_explorer"]


def _lognormal_from_p50_p99(p50_ms: float, p99_ms: float) -> LogNormal:
    mu = math.log(p50_ms)
    if p99_ms <= p50_ms:
        sigma = 0.01
    else:
        sigma = (math.log(p99_ms) - mu) / 2.326
    return LogNormal(mu=mu, sigma=max(sigma, 0.01))


def build_scenario(
    p50_spread_ab: float,
    tail_ratio_a: float,
    name: str | None = None,
) -> ScenarioConfig:
    """Llama-like 3-provider setup:
    A: fastest P50, variable tail.
    B: slower P50 (by factor p50_spread_ab), stable tail (P99 = 1.3*P50).
    C: slow P50 (3x A's), moderate tail (P99 = 2*P50), always cost fallback.
    """
    fast_p50 = BASE_P50_MS
    mid_p50 = BASE_P50_MS * p50_spread_ab
    slow_p50 = BASE_P50_MS * 3.0

    fast_p99 = fast_p50 * tail_ratio_a
    mid_p99 = mid_p50 * 1.3  # B is the stable backup
    slow_p99 = slow_p50 * 2.0

    tps = LogNormal(mu=5.5, sigma=0.3)
    providers = [
        SyntheticProvider(
            name="A",
            cost_per_token=UNIFORM_COST,
            ttft_dist=_lognormal_from_p50_p99(fast_p50, fast_p99),
            tps_dist=tps,
        ),
        SyntheticProvider(
            name="B",
            cost_per_token=UNIFORM_COST,
            ttft_dist=_lognormal_from_p50_p99(mid_p50, mid_p99),
            tps_dist=tps,
        ),
        SyntheticProvider(
            name="C",
            cost_per_token=UNIFORM_COST,
            ttft_dist=_lognormal_from_p50_p99(slow_p50, slow_p99),
            tps_dist=tps,
        ),
    ]

    return ScenarioConfig(
        name=name or f"v2grid_spread{p50_spread_ab:.1f}_tail{tail_ratio_a:.1f}",
        description=(
            f"Llama-like grid: A(P50={fast_p50:.0f}ms,P99={fast_p99:.0f}ms) "
            f"B(P50={mid_p50:.0f},P99={mid_p99:.0f},stable) "
            f"C(P50={slow_p50:.0f},P99={slow_p99:.0f},slow)"
        ),
        providers=providers,
    )


@dataclass
class CellResult:
    p50_spread_ab: float
    tail_ratio_a: float
    strategy: str
    seed: int
    p50_ms: float
    p99_ms: float
    slo_violation_rate_2s: float
    mean_cost_usd: float
    hedge_rate: float
    provider_fractions: dict[str, float]


def run_cell(
    p50_spread_ab: float,
    tail_ratio_a: float,
    seeds: list[int] | None = None,
) -> list[CellResult]:
    if seeds is None:
        seeds = [42, 43, 44]

    scenario = build_scenario(p50_spread_ab, tail_ratio_a)
    from .workload import generate_workload

    requests = generate_workload(
        n_requests=scenario.n_requests,
        duration_seconds=scenario.duration_seconds,
        seed=0,
        start_time=0.0,
        arrival_process=scenario.arrival_process,
    )

    results: list[CellResult] = []
    for strategy in STRATEGIES:
        for seed in seeds:
            run = run_strategy(scenario, requests, strategy, seed=seed)
            ttft = np.asarray(run.ttft_ms)
            valid = ttft[ttft > 0]
            p50 = float(np.percentile(valid, 50)) if len(valid) else -1.0
            p99 = float(np.percentile(valid, 99)) if len(valid) else -1.0
            slo_v = float(np.mean(ttft > 2000.0)) if len(ttft) else 0.0
            cost_mean = float(np.mean(run.cost_usd)) if len(run.cost_usd) else 0.0
            hedge_rate = float(np.mean(run.hedge_triggered)) if len(run.hedge_triggered) else 0.0
            total = len(run.provider) or 1
            fractions = {p: run.provider.count(p) / total for p in sorted(set(run.provider))}
            results.append(
                CellResult(
                    p50_spread_ab=p50_spread_ab,
                    tail_ratio_a=tail_ratio_a,
                    strategy=strategy,
                    seed=seed,
                    p50_ms=p50,
                    p99_ms=p99,
                    slo_violation_rate_2s=slo_v,
                    mean_cost_usd=cost_mean,
                    hedge_rate=hedge_rate,
                    provider_fractions=fractions,
                )
            )
    return results


def run_grid(
    p50_spreads: list[float] | None = None,
    tail_ratios: list[float] | None = None,
    seeds: list[int] | None = None,
) -> list[CellResult]:
    if p50_spreads is None:
        p50_spreads = P50_SPREAD_AB
    if tail_ratios is None:
        tail_ratios = TAIL_RATIO_A

    all_results: list[CellResult] = []
    total = len(p50_spreads) * len(tail_ratios)
    for i, sp in enumerate(p50_spreads):
        for j, tr in enumerate(tail_ratios):
            idx = i * len(tail_ratios) + j + 1
            print(f"[{idx}/{total}] spread_AB={sp:.1f}  tail_A={tr:.1f}")
            all_results.extend(run_cell(sp, tr, seeds=seeds))
    return all_results


def aggregate_cells(
    results: list[CellResult],
) -> dict[tuple[float, float, str], dict[str, float]]:
    bucket: dict[tuple[float, float, str], list[CellResult]] = {}
    for r in results:
        bucket.setdefault(
            (r.p50_spread_ab, r.tail_ratio_a, r.strategy), []
        ).append(r)
    agg: dict[tuple[float, float, str], dict[str, float]] = {}
    for key, rs in bucket.items():
        agg[key] = {
            "p50_ms": float(np.mean([r.p50_ms for r in rs])),
            "p99_ms": float(np.mean([r.p99_ms for r in rs])),
            "slo_violation_rate_2s": float(np.mean([r.slo_violation_rate_2s for r in rs])),
            "mean_cost_usd": float(np.mean([r.mean_cost_usd for r in rs])),
            "hedge_rate": float(np.mean([r.hedge_rate for r in rs])),
        }
    return agg


def compute_winner_matrix(
    agg, p50_spreads, tail_ratios, tie_threshold: float = 0.10
):
    n_sp = len(p50_spreads)
    n_tr = len(tail_ratios)
    magnitude = np.zeros((n_tr, n_sp))
    for i, sp in enumerate(p50_spreads):
        for j, tr in enumerate(tail_ratios):
            lp = agg.get((sp, tr, "lp_explorer"))
            v2 = agg.get((sp, tr, "v2_explorer"))
            if lp is None or v2 is None:
                continue
            lp_p99 = lp["p99_ms"]
            v2_p99 = v2["p99_ms"]
            if lp_p99 <= 0 or v2_p99 <= 0:
                continue
            magnitude[j, i] = (lp_p99 - v2_p99) / min(lp_p99, v2_p99)
    return magnitude
