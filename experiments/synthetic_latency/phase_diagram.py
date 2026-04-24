"""LP vs V2 phase diagram: 2D parameter sweep characterizing winner regime.

Generates a 5x5 grid over (P50 spread, tail asymmetry of fastest provider).
Each cell is a 3-provider synthetic scenario with uniform cost, and compares
lp_hedge_pp (LP + hedge + explorer) against v2_hedge_pp (V2 + hedge +
explorer). Output is a heatmap annotated with production regime anchors.

Why this matters:
  Our synthetic S1-S5 scenarios used hand-crafted provider configurations,
  so we saw that "V2 fails in heavy-tail regimes" and "V2 oscillates when
  providers are similar", but we could not tell *where exactly* the
  boundary is. Production data on Llama-3.3-70B showed V2 winning even
  though we had previously declared V2 dead. The 5x5 grid traces the
  actual switching boundary between LP and V2 as (P50-spread, tail-ratio)
  varies, giving the paper a quantitative "regime map" rather than a
  single winner.

Parameters swept:
  - P50_spread = P50_slow / P50_fast, axis X, 5 values
  - tail_asymmetry = P99_fast / P50_fast (tail of fastest provider),
    axis Y, 5 values

All scenarios:
  - 3 providers with equal cost (isolates latency effect from cost effect)
  - 2000 requests over 1 simulated hour
  - SLO = 2s
  - 3 seeds, averaged
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

from rwsim.runner import run_registered_strategy as run_strategy
from rwsim.world import LogNormal, ScenarioConfig, SyntheticProvider


# ---------------------------------------------------------------------------
# Grid parameterization
# ---------------------------------------------------------------------------

# P50 spread: ratio of slowest to fastest provider.
# 1.0 = all equal, 10.0 = extreme dominance.
P50_SPREADS = [1.1, 1.5, 2.0, 5.0, 10.0]

# Tail asymmetry: P99 / P50 ratio for the FASTEST provider. This is the
# critical dimension because V2 concentrates on fastest P50 and pays the
# cost of whatever tail that provider has.
TAIL_RATIOS = [1.5, 3.0, 5.0, 10.0, 20.0]

# Fixed anchor: fastest provider's P50.
BASE_P50_MS = 200.0

# Provider cost: all identical so this sweep isolates latency trade-offs.
UNIFORM_COST_PER_TOKEN = 1.0e-6

# Algorithms compared per cell.
STRATEGIES = ["lp_explorer", "v2_explorer"]

# Reporting SLO thresholds.
SLO_THRESHOLDS_MS = [1000.0, 2000.0, 3000.0, 5000.0]


# ---------------------------------------------------------------------------
# Scenario builder
# ---------------------------------------------------------------------------


def _lognormal_from_p50_p99(p50_ms: float, p99_ms: float) -> LogNormal:
    """Build a LogNormal with the given P50 and P99 (in ms)."""
    mu = math.log(p50_ms)
    if p99_ms <= p50_ms:
        sigma = 0.01
    else:
        sigma = (math.log(p99_ms) - mu) / 2.326
    return LogNormal(mu=mu, sigma=max(sigma, 0.01))


def build_scenario(
    p50_spread: float,
    tail_ratio: float,
    name: str | None = None,
) -> ScenarioConfig:
    """Construct a 3-provider synthetic scenario at the given grid point.

    Provider layout:
      A (fastest):  P50 = BASE_P50_MS,        P99 = BASE_P50_MS * tail_ratio
      B (middle):   P50 = BASE_P50_MS * sqrt(p50_spread), P99 = 2 * P50
      C (slowest):  P50 = BASE_P50_MS * p50_spread, P99 = 2 * P50

    The fastest provider's tail is the independent variable (Y axis); the
    other two providers have moderate tails (P99/P50 = 2) so they are
    "stable but slower" backups. All three providers have the same cost,
    which isolates the latency effect.
    """
    fast_p50 = BASE_P50_MS
    mid_p50 = BASE_P50_MS * math.sqrt(p50_spread)
    slow_p50 = BASE_P50_MS * p50_spread

    fast_p99 = fast_p50 * tail_ratio
    mid_p99 = mid_p50 * 2.0
    slow_p99 = slow_p50 * 2.0

    tps = LogNormal(mu=5.5, sigma=0.3)  # ~245 tokens/sec
    providers = [
        SyntheticProvider(
            name="A",
            cost_per_token=UNIFORM_COST_PER_TOKEN,
            ttft_dist=_lognormal_from_p50_p99(fast_p50, fast_p99),
            tps_dist=tps,
        ),
        SyntheticProvider(
            name="B",
            cost_per_token=UNIFORM_COST_PER_TOKEN,
            ttft_dist=_lognormal_from_p50_p99(mid_p50, mid_p99),
            tps_dist=tps,
        ),
        SyntheticProvider(
            name="C",
            cost_per_token=UNIFORM_COST_PER_TOKEN,
            ttft_dist=_lognormal_from_p50_p99(slow_p50, slow_p99),
            tps_dist=tps,
        ),
    ]

    return ScenarioConfig(
        name=name or f"grid_p50x{p50_spread:.1f}_tail{tail_ratio:.1f}",
        description=(
            f"3-provider grid cell: P50 spread {p50_spread:.1f}x, "
            f"fastest-provider tail P99/P50 {tail_ratio:.1f}x"
        ),
        providers=providers,
    )


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------


@dataclass
class CellResult:
    """Metrics for a single grid cell."""

    p50_spread: float
    tail_ratio: float
    strategy: str
    seed: int

    n_requests: int
    p50_ms: float
    p99_ms: float
    slo_violation_rate_2s: float
    mean_cost_usd: float
    hedge_rate: float
    provider_fractions: dict[str, float]


def run_cell(
    p50_spread: float,
    tail_ratio: float,
    seeds: list[int] | None = None,
) -> list[CellResult]:
    """Run all strategies for one (p50_spread, tail_ratio) grid cell."""
    if seeds is None:
        seeds = [42, 43, 44]

    # Build scenario and workload once (shared across strategies and seeds).
    scenario = build_scenario(p50_spread, tail_ratio)
    from rwsim.world import generate_workload

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
            ttft_valid = ttft[ttft > 0]
            p50 = float(np.percentile(ttft_valid, 50)) if len(ttft_valid) else -1.0
            p99 = float(np.percentile(ttft_valid, 99)) if len(ttft_valid) else -1.0
            slo_v = float(np.mean(ttft > 2000.0)) if len(ttft) else 0.0
            cost_mean = float(np.mean(run.cost_usd)) if len(run.cost_usd) else 0.0
            hedge_rate = float(np.mean(run.hedge_triggered)) if len(run.hedge_triggered) else 0.0
            total = len(run.provider) or 1
            fractions = {
                p: run.provider.count(p) / total for p in sorted(set(run.provider))
            }
            results.append(
                CellResult(
                    p50_spread=p50_spread,
                    tail_ratio=tail_ratio,
                    strategy=strategy,
                    seed=seed,
                    n_requests=len(ttft),
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
    """Run the full 2D grid sweep."""
    if p50_spreads is None:
        p50_spreads = P50_SPREADS
    if tail_ratios is None:
        tail_ratios = TAIL_RATIOS

    all_results: list[CellResult] = []
    for i, sp in enumerate(p50_spreads):
        for j, tr in enumerate(tail_ratios):
            print(f"[{i * len(tail_ratios) + j + 1}/{len(p50_spreads) * len(tail_ratios)}] "
                  f"p50_spread={sp:.1f}  tail_ratio={tr:.1f}")
            cell = run_cell(sp, tr, seeds=seeds)
            all_results.extend(cell)
    return all_results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_cells(
    results: list[CellResult],
) -> dict[tuple[float, float, str], dict[str, float]]:
    """Average seed-level metrics per (p50_spread, tail_ratio, strategy)."""
    bucket: dict[tuple[float, float, str], list[CellResult]] = {}
    for r in results:
        key = (r.p50_spread, r.tail_ratio, r.strategy)
        bucket.setdefault(key, []).append(r)
    agg: dict[tuple[float, float, str], dict[str, float]] = {}
    for key, rs in bucket.items():
        agg[key] = {
            "p50_ms": float(np.mean([r.p50_ms for r in rs])),
            "p99_ms": float(np.mean([r.p99_ms for r in rs])),
            "slo_violation_rate_2s": float(
                np.mean([r.slo_violation_rate_2s for r in rs])
            ),
            "mean_cost_usd": float(np.mean([r.mean_cost_usd for r in rs])),
            "hedge_rate": float(np.mean([r.hedge_rate for r in rs])),
        }
    return agg


def compute_winner_matrix(
    agg: dict[tuple[float, float, str], dict[str, float]],
    p50_spreads: list[float],
    tail_ratios: list[float],
    tie_threshold: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a winner label matrix and a magnitude matrix.

    Winner label encoding (for heatmap):
      +2  V2 crushes LP         (V2 P99 > 20% better)
      +1  V2 wins                (V2 P99 10-20% better)
       0  tie                    (within tie_threshold)
      -1  LP wins                (LP P99 10-20% better)
      -2  LP crushes V2

    Magnitude = (LP.P99 - V2.P99) / min(LP.P99, V2.P99)
      positive: V2 is faster (lower P99)
      negative: LP is faster
    """
    n_sp = len(p50_spreads)
    n_tr = len(tail_ratios)
    winner = np.zeros((n_tr, n_sp))  # rows: tail (y), cols: spread (x)
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
            rel = (lp_p99 - v2_p99) / min(lp_p99, v2_p99)
            magnitude[j, i] = rel
            if abs(rel) < tie_threshold:
                winner[j, i] = 0
            elif rel > 0.5:
                winner[j, i] = 2  # V2 crushes LP
            elif rel > tie_threshold:
                winner[j, i] = 1  # V2 wins
            elif rel < -0.5:
                winner[j, i] = -2  # LP crushes V2
            else:
                winner[j, i] = -1  # LP wins
    return winner, magnitude
