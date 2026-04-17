"""Synthetic scenario definitions for controlled routing strategy evaluation.

Five scenarios cover the key failure modes identified in the LP analysis:
  S1 – Dominant provider (LP over-diversifies away from the obvious winner)
  S2 – Cost/latency trade-off (answer depends on SLO stringency)
  S3 – Tail-heavy provider (good P50 but catastrophic P99)
  S4 – Distribution shift (provider degrades mid-simulation)
  S5 – Similar providers (cheapest in near-best-P50 band should win)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .providers import LogNormal, ShiftingProvider, SyntheticProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Shared tokens-per-second distribution (P50 ≈ 245 tps).
_TPS = LogNormal(mu=5.5, sigma=0.3)


def _ln_p50_sigma(p50_ms: float, sigma: float = 0.5) -> LogNormal:
    """LogNormal with given P50 and fixed sigma."""
    return LogNormal(mu=math.log(p50_ms), sigma=sigma)


def _ln_p50_p99(p50_ms: float, p99_ms: float) -> LogNormal:
    """LogNormal with given P50 and P99 in ms."""
    mu = math.log(p50_ms)
    sigma = (math.log(p99_ms) - mu) / 2.326
    return LogNormal(mu=mu, sigma=max(sigma, 0.01))


# ---------------------------------------------------------------------------
# ScenarioConfig
# ---------------------------------------------------------------------------


@dataclass
class ScenarioConfig:
    """Configuration for one synthetic simulation scenario."""

    name: str
    description: str
    providers: list[SyntheticProvider]

    # Workload parameters.
    n_requests: int = 2000
    duration_seconds: float = 3600.0     # 1 hour
    arrival_process: str = "poisson"

    # SLO thresholds used for reporting (ms).
    slo_thresholds_ms: list[float] = field(
        default_factory=lambda: [1000.0, 2000.0, 3000.0, 5000.0]
    )

    @property
    def primary_slo_ms(self) -> float:
        """SLO used to configure routers (2 s)."""
        return 2000.0

    @property
    def provider_names(self) -> list[str]:
        return [p.name for p in self.providers]


# ---------------------------------------------------------------------------
# Scenario factory
# ---------------------------------------------------------------------------


def make_scenarios() -> dict[str, ScenarioConfig]:
    """Create all five synthetic evaluation scenarios.

    Returns a dict keyed by scenario ID (e.g. ``"s1_dominant"``).
    """
    return {
        # ------------------------------------------------------------------
        # S1: Dominant provider — B is cheapest AND fastest.
        # Correct answer: 100 % to B.
        # LP risk: spreads traffic to A to meet F(SLO) >= 0.99.
        # ------------------------------------------------------------------
        "s1_dominant": ScenarioConfig(
            name="s1_dominant",
            description=(
                "B dominates: P50=100 ms ($0.5/M). "
                "A is slow (P50=1100 ms) and expensive ($2/M)."
            ),
            providers=[
                SyntheticProvider(
                    name="A",
                    cost_per_token=2.0e-6,
                    ttft_dist=_ln_p50_p99(1100, 7000),
                    tps_dist=_TPS,
                ),
                SyntheticProvider(
                    name="B",
                    cost_per_token=0.5e-6,
                    ttft_dist=_ln_p50_sigma(100),
                    tps_dist=_TPS,
                ),
            ],
        ),

        # ------------------------------------------------------------------
        # S2: Cost/latency trade-off.
        # At SLO=1 s: B wins (never violates). At SLO=5 s: A wins (cheaper).
        # ------------------------------------------------------------------
        "s2_tradeoff": ScenarioConfig(
            name="s2_tradeoff",
            description=(
                "A: cheap ($0.5/M) + slow (P50=500 ms, P99=1600 ms). "
                "B: expensive ($2/M) + fast (P50=100 ms)."
            ),
            providers=[
                SyntheticProvider(
                    name="A",
                    cost_per_token=0.5e-6,
                    ttft_dist=_ln_p50_p99(500, 1600),
                    tps_dist=_TPS,
                ),
                SyntheticProvider(
                    name="B",
                    cost_per_token=2.0e-6,
                    ttft_dist=_ln_p50_sigma(100),
                    tps_dist=_TPS,
                ),
            ],
        ),

        # ------------------------------------------------------------------
        # S3: Tail-heavy vs. stable.
        # A looks great on P50 but has catastrophic P99 — hedging should help.
        # B is slower but consistent.
        # ------------------------------------------------------------------
        "s3_tail": ScenarioConfig(
            name="s3_tail",
            description=(
                "A: P50=100 ms but P99=5 s (heavy tail). "
                "B: P50=300 ms, P99=1 s (stable). Both cost $1/M."
            ),
            providers=[
                SyntheticProvider(
                    name="A",
                    cost_per_token=1.0e-6,
                    ttft_dist=_ln_p50_p99(100, 5000),
                    tps_dist=_TPS,
                ),
                SyntheticProvider(
                    name="B",
                    cost_per_token=1.0e-6,
                    ttft_dist=_ln_p50_p99(300, 1000),
                    tps_dist=_TPS,
                ),
            ],
        ),

        # ------------------------------------------------------------------
        # S4: Distribution shift at t = 30 min.
        # A starts fast (P50=100 ms), degrades to P50=1000 ms at t=1800 s.
        # B stays constant at P50=300 ms. Both cost $1/M.
        # Correct answer: route to A first, switch to B after shift.
        # ------------------------------------------------------------------
        "s4_shift": ScenarioConfig(
            name="s4_shift",
            description=(
                "A: fast (P50=100 ms) until t=30 min, then slow (P50=1000 ms). "
                "B: constant P50=300 ms. Both cost $1/M."
            ),
            providers=[
                ShiftingProvider(
                    name="A",
                    cost_per_token=1.0e-6,
                    ttft_dist=_ln_p50_sigma(100),
                    tps_dist=_TPS,
                    shift_time=1800.0,
                    ttft_dist_after=_ln_p50_sigma(1000),
                ),
                SyntheticProvider(
                    name="B",
                    cost_per_token=1.0e-6,
                    ttft_dist=_ln_p50_sigma(300),
                    tps_dist=_TPS,
                ),
            ],
        ),

        # ------------------------------------------------------------------
        # S5: Three similar providers — C is cheapest within the P50 band.
        # V2 should collapse to C. LP may diversify across all three.
        #
        # P50 spread is kept within V2's default 10 % near-best band so
        # all three are eligible and the cheapest (C) should win.
        # ------------------------------------------------------------------
        "s5_similar": ScenarioConfig(
            name="s5_similar",
            description=(
                "A: P50=200 ms $1/M. B: P50=207 ms $0.8/M. C: P50=215 ms $0.5/M. "
                "All within V2's 10 % near-best band; C is cheapest."
            ),
            providers=[
                SyntheticProvider(
                    name="A",
                    cost_per_token=1.0e-6,
                    ttft_dist=_ln_p50_sigma(200),
                    tps_dist=_TPS,
                ),
                SyntheticProvider(
                    name="B",
                    cost_per_token=0.8e-6,
                    ttft_dist=_ln_p50_sigma(207),
                    tps_dist=_TPS,
                ),
                SyntheticProvider(
                    name="C",
                    cost_per_token=0.5e-6,
                    ttft_dist=_ln_p50_sigma(215),
                    tps_dist=_TPS,
                ),
            ],
        ),
    }
