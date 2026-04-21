"""Sanity-check scenarios for the synthetic simulator (Juncheng 4/20 request).

Motivation
----------
The existing S1-S5 scenarios produce results that are hard to attribute to
specific algorithmic behavior because the ground truth under multi-provider
heterogeneity is ambiguous. Juncheng's 4/20 meeting guidance: start with
scenarios whose correct behavior is analytically obvious, then gradually
increase complexity so any deviation from expected behavior isolates a bug.

Five progressive steps are defined. Each has an explicit "ground truth"
docstring describing the expected routing behavior. Step 5 is a parameter
sweep that produces the key responsiveness curve for the paper's appendix.

Placement
---------
This file should live at:
    routewise-simulator/experiment/scripts/simulate/synthetic/sanity_check.py

It is imported by the project-root script run_sanity_check.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .providers import LogNormal, SyntheticProvider
from .scenarios import ScenarioConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TPS = LogNormal(mu=5.5, sigma=0.3)  # shared tokens-per-second distribution


def _ln_p50_sigma(p50_ms: float, sigma: float = 0.5) -> LogNormal:
    return LogNormal(mu=math.log(p50_ms), sigma=sigma)


def _make_provider(
    name: str,
    cost_per_m_tokens: float,
    p50_ms: float,
    sigma: float = 0.5,
) -> SyntheticProvider:
    """Build a SyntheticProvider from human-readable params.

    Args:
        name: Provider label.
        cost_per_m_tokens: Cost in USD per million tokens.
        p50_ms: Target median TTFT in milliseconds.
        sigma: LogNormal sigma. 0.5 => P99 ~3.4 * P50 (moderate spread).
    """
    return SyntheticProvider(
        name=name,
        cost_per_token=cost_per_m_tokens * 1e-6,
        ttft_dist=_ln_p50_sigma(p50_ms, sigma=sigma),
        tps_dist=_TPS,
    )


# ---------------------------------------------------------------------------
# Step 1: Clear winner
# ---------------------------------------------------------------------------


def step1_clear_winner() -> ScenarioConfig:
    """A: expensive + slow; B: cheap + fast.

    Ground truth:
        All cost-aware policies (cheapest_fixed, lp_mix, lp_hedge,
        lp_explorer, v2_p50_hedge, v2_explorer) should route ~100% to B.
        fastest_fixed and oracle_per_window should also pick B.
        round_robin is the control: forced 50/50, expected to lose.

    If any non-baseline policy sends >5% traffic to A: simulator or algo bug.
    """
    return ScenarioConfig(
        name="step1_clear_winner",
        description="A: $2/M P50=1000ms; B: $0.5/M P50=100ms. B dominates both axes.",
        providers=[
            _make_provider("A", cost_per_m_tokens=2.0, p50_ms=1000),
            _make_provider("B", cost_per_m_tokens=0.5, p50_ms=100),
        ],
    )


# ---------------------------------------------------------------------------
# Step 2: Converge cost (same P50 difference, costs approach)
# ---------------------------------------------------------------------------


def step2_cost_converge(a_cost_per_m: float) -> ScenarioConfig:
    """Sweep A's cost from $2/M down to $0.5/M while keeping A slow.

    A remains slower than B (P50=1000 vs 100 ms). Only A's price changes.

    Ground truth:
        Because B is always faster AND a_cost_per_m >= 0.5, B remains
        cost-eligible under both LP's F(SLO) constraint and V2's P50 band.
        All cost-aware policies should STAY 100% on B across the sweep.

        This tests: does the router chase transient cost signals even when
        the other axis (latency) clearly favors B?
    """
    return ScenarioConfig(
        name=f"step2_cost_converge_a{a_cost_per_m:.2f}",
        description=(
            f"A: ${a_cost_per_m}/M P50=1000ms; B: $0.5/M P50=100ms. "
            "B always faster; traffic should stay on B."
        ),
        providers=[
            _make_provider("A", cost_per_m_tokens=a_cost_per_m, p50_ms=1000),
            _make_provider("B", cost_per_m_tokens=0.5, p50_ms=100),
        ],
    )


def step2_sweep() -> list[ScenarioConfig]:
    """Return the Step 2 cost sweep (5 points)."""
    return [step2_cost_converge(c) for c in [2.0, 1.5, 1.0, 0.75, 0.5]]


# ---------------------------------------------------------------------------
# Step 3: Identical providers
# ---------------------------------------------------------------------------


def step3_identical() -> ScenarioConfig:
    """A and B are identical in every way.

    Ground truth:
        No strategy can prefer one over the other. Traffic split is
        arbitrary but should not cause instability (no hedge thrashing,
        no cost explosion). round_robin should be at the median.

        Expected: provider fractions near 50/50 or 100/0; cost and P99
        should match cheapest_fixed exactly.
    """
    return ScenarioConfig(
        name="step3_identical",
        description="A and B: $1/M, P50=300ms, identical distributions.",
        providers=[
            _make_provider("A", cost_per_m_tokens=1.0, p50_ms=300),
            _make_provider("B", cost_per_m_tokens=1.0, p50_ms=300),
        ],
    )


# ---------------------------------------------------------------------------
# Step 4: Band scenario (V2 eligibility boundary)
# ---------------------------------------------------------------------------


def step4_band() -> ScenarioConfig:
    """A: slightly slower but cheaper; B: fast and expensive.

    A's P50=110ms is exactly at the edge of V2's 10% band ([100, 110]ms).

    Ground truth:
        - cheapest_fixed => 100% A (A is cheapest)
        - lp_mix => 100% A (A meets F(SLO) @2s with margin; cheaper)
        - v2_only => 100% A (A is in band [100, 110] and cheaper)
        - fastest_fixed => 100% B (B has lower P50)
        - oracle_per_window => 100% B (P50-optimal)

    If V2 routes to B despite A being in its band: V2 pre-filter or
    cheapest-in-band logic has a bug.
    """
    return ScenarioConfig(
        name="step4_band",
        description=(
            "A: $0.5/M P50=110ms (edge of V2 10% band). "
            "B: $2/M P50=100ms. Both meet SLO=2s comfortably."
        ),
        providers=[
            _make_provider("A", cost_per_m_tokens=0.5, p50_ms=110),
            _make_provider("B", cost_per_m_tokens=2.0, p50_ms=100),
        ],
    )


# ---------------------------------------------------------------------------
# Step 5: Progressive latency sweep (main deliverable)
# ---------------------------------------------------------------------------


def step5_latency_sweep(a_p50_ms: float) -> ScenarioConfig:
    """Parameterize A's P50 from 100ms (identical to B) to 500ms (way slower).

    B is fixed at $2/M, P50=100ms. A is fixed at $0.5/M, swept P50.

    Ground truth progression (SLO = 2000ms):
        a_p50=100ms  -> LP/V2: 100% A (A in band, cheaper; B is P50 winner
                        by tie but A wins by cost)
        a_p50=105ms  -> LP/V2: 100% A (A still in band)
        a_p50=110ms  -> V2: 100% A (boundary)
        a_p50=111ms  -> V2: 100% B (A just out of band) — SHARP JUMP
        a_p50=110ms  -> LP: smooth transition (no discontinuity)
        a_p50=500ms  -> LP: 100% B or mixed (A too slow to satisfy 99% F(SLO))
        a_p50=500ms  -> V2: 100% B (A far out of band)

    Key validation: V2 share of A should have a sharp step down at the
    band boundary; LP share of A should decrease smoothly.
    """
    return ScenarioConfig(
        name=f"step5_sweep_a{int(a_p50_ms)}",
        description=(
            f"A: $0.5/M P50={a_p50_ms:.0f}ms; B: $2/M P50=100ms. "
            "Sweep A's P50 to trace responsiveness curve."
        ),
        providers=[
            _make_provider("A", cost_per_m_tokens=0.5, p50_ms=a_p50_ms),
            _make_provider("B", cost_per_m_tokens=2.0, p50_ms=100),
        ],
    )


def step5_sweep() -> list[ScenarioConfig]:
    """Return the Step 5 P50 sweep points (fine-grained around the band edge).

    Dense sampling near P50(A) = 110ms (V2 band boundary) so the expected
    discontinuity is visible; coarser beyond.
    """
    p50_values = [
        100, 102, 105, 108, 109, 110, 111, 112, 115, 120, 130, 150,
        200, 300, 500,
    ]
    return [step5_latency_sweep(p) for p in p50_values]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class SanityStep:
    """One sanity-check step: may contain one scenario or a sweep list."""

    step_id: str
    description: str
    scenarios: list[ScenarioConfig]
    is_sweep: bool = False


def make_sanity_steps() -> list[SanityStep]:
    """Return the 5 progressive sanity-check steps."""
    return [
        SanityStep(
            step_id="step1_clear_winner",
            description="Clear winner: B dominates both cost and latency.",
            scenarios=[step1_clear_winner()],
        ),
        SanityStep(
            step_id="step2_cost_converge",
            description="Sweep A's cost toward B's; B always faster.",
            scenarios=step2_sweep(),
            is_sweep=True,
        ),
        SanityStep(
            step_id="step3_identical",
            description="A and B are identical.",
            scenarios=[step3_identical()],
        ),
        SanityStep(
            step_id="step4_band",
            description="A at V2's band boundary, cheaper than B.",
            scenarios=[step4_band()],
        ),
        SanityStep(
            step_id="step5_latency_sweep",
            description="Sweep A's P50 from 100 to 500ms (band boundary at 110).",
            scenarios=step5_sweep(),
            is_sweep=True,
        ),
    ]
