"""Pytest coverage for the eval-grid factory.

Covers the structural invariants (no router execution) plus the surface
contract: variant naming matches what ``lp_budget_eval`` runs, workload
mapping resolves, and the new ``Uniform`` / ``Normal`` distributions
flow through ``provider_p95_at`` without crashing.
"""

from __future__ import annotations

import pytest

from experiments.tiered_capacity.eval_grid import (
    LatencyFamily,
    PAPER_GRID_VARIANTS,
    PAPER_HEDGE_MODES,
    PAPER_P_VALUES,
    PAPER_WORKLOADS,
    ProviderSetup,
    WORKLOAD_DATASET_IDS,
    assert_grid_invariants,
    make_eval_grid_scenarios,
    make_scenario,
    make_ttft_distribution,
    runner_dataset_id,
    variant_name,
)
from experiments.tiered_capacity.lp_budget_eval import (
    MAIN_VARIANTS,
    TRACE_WORKLOAD_DATASETS,
)
from rwsim.policies.latency_routers.tiered_filters import provider_p95_at
from rwsim.world.distributions import LogNormal, Normal, Uniform


# ---------------------------------------------------------------------------
# Structural grid invariants
# ---------------------------------------------------------------------------


def test_grid_has_twelve_scenarios():
    """4 stages (Stage 0 / 1 / 2 / 3) × 3 distributions = 12 scenarios."""
    grid = make_eval_grid_scenarios()
    assert len(grid) == 12
    expected_names = {
        f"grid_{stage.value}_{family.value}"
        for stage in ProviderSetup
        for family in LatencyFamily
    }
    assert set(grid) == expected_names


def test_grid_structural_invariants_pass():
    # Should not raise.
    assert_grid_invariants()


def test_stage0_same_latency_distinct_costs():
    """Stage 0 (Juncheng directive 2026-04-28): identical latency,
    different costs. Cost router should always pick the cheapest."""
    for family in LatencyFamily:
        scenario = make_scenario(ProviderSetup.SAME_LATENCY, family)
        p50s = {p.ttft_dist.p50() for p in scenario.providers}
        costs = {p.cost_per_token for p in scenario.providers}
        assert len(p50s) == 1, f"Stage 0 / {family} should have one P50, got {p50s}"
        assert len(costs) >= 2, (
            f"Stage 0 / {family} should have multiple costs, got {costs}"
        )


def test_stage1_same_cost():
    for family in LatencyFamily:
        scenario = make_scenario(ProviderSetup.SAME_COST, family)
        costs = {p.cost_per_token for p in scenario.providers}
        assert len(costs) == 1, f"Stage 1 / {family} should have one cost, got {costs}"


def test_stage2_distinct_costs():
    for family in LatencyFamily:
        scenario = make_scenario(ProviderSetup.COST_LATENCY_TRADEOFF, family)
        costs = {p.cost_per_token for p in scenario.providers}
        assert len(costs) >= 3


def test_stage3_has_all_three_tiers():
    from rwsim.world.providers import ProviderTier

    for family in LatencyFamily:
        scenario = make_scenario(ProviderSetup.JOINT_PROVIDER, family)
        tiers = {p.tier for p in scenario.providers}
        assert {ProviderTier.S_A, ProviderTier.S_Q, ProviderTier.S_C}.issubset(tiers)


# ---------------------------------------------------------------------------
# Variant-name contract: must match the runnable ids in lp_budget_eval.
# ---------------------------------------------------------------------------


def test_paper_grid_variants_match_runner_main_variants():
    main_variant_set = set(MAIN_VARIANTS)
    for variant in PAPER_GRID_VARIANTS:
        assert variant in main_variant_set, (
            f"Variant {variant!r} from PAPER_GRID_VARIANTS is not in "
            f"lp_budget_eval.MAIN_VARIANTS — runner won't accept it."
        )


def test_variant_name_format():
    assert variant_name(0.0, "no_hedge") == "budget_range_p0"
    assert variant_name(0.25, "hedge") == "budget_range_p25_hedge"
    assert variant_name(0.5, "explorer") == "budget_range_p50_explorer"
    assert variant_name(1.0, "no_hedge") == "budget_range_p100"


def test_variant_name_rejects_unknown_mode():
    with pytest.raises(ValueError):
        variant_name(0.5, "wat")


def test_paper_grid_variants_count():
    """5 p × 2 hedge modes (no_hedge, hedge) = 10 paper-grid variants.

    Explorer is intentionally excluded — see PAPER_HEDGE_MODES docstring.
    """
    assert PAPER_HEDGE_MODES == ("no_hedge", "hedge")
    assert len(PAPER_GRID_VARIANTS) == len(PAPER_P_VALUES) * len(PAPER_HEDGE_MODES) == 10


def test_paper_grid_variants_excludes_explorer():
    """Explorer must not leak back into the simulator paper grid."""
    explorer_in_grid = [v for v in PAPER_GRID_VARIANTS if "explorer" in v]
    assert explorer_in_grid == [], (
        f"Explorer variants leaked into PAPER_GRID_VARIANTS: {explorer_in_grid}. "
        "Per the 2026-04-28 design directive (Juncheng), explorer adds no signal "
        "in the stationary simulator under the active-system probing assumption."
    )


def test_lp_comparison_variants_runnable():
    """LP_COMPARISON_VARIANTS = 2 original_lp + 10 budget_range_p*; every
    entry must be a runnable id known to lp_budget_eval."""
    from experiments.tiered_capacity.eval_grid import LP_COMPARISON_VARIANTS

    assert len(LP_COMPARISON_VARIANTS) == 12
    main_set = set(MAIN_VARIANTS)
    for v in LP_COMPARISON_VARIANTS:
        assert v in main_set, f"{v!r} is not in lp_budget_eval.MAIN_VARIANTS"

    # Composition: original_lp x 2 (no_hedge, hedge) + paper grid x 10
    assert LP_COMPARISON_VARIANTS[:2] == ("original_lp", "original_lp_hedge")
    assert LP_COMPARISON_VARIANTS[2:] == PAPER_GRID_VARIANTS


def test_shadow_price_ablation_excluded_from_main_variants():
    """``original_lp_rawcost*`` is an ablation, not a main variant.

    Pinning this prevents future "just add it to MAIN_VARIANTS for
    convenience" regressions. The rawcost variants must only be reachable
    through ``--include-shadow-price-ablation`` so default legacy and
    paper-grid runs don't get their main result polluted with what is
    explicitly the algorithm with shadow pricing turned off.
    """
    from experiments.tiered_capacity.lp_budget_eval import (
        SHADOW_PRICE_ABLATION_VARIANTS,
    )

    rawcost_in_main = [v for v in MAIN_VARIANTS if "rawcost" in v]
    assert rawcost_in_main == [], (
        f"rawcost variants leaked into MAIN_VARIANTS: {rawcost_in_main}. "
        "They belong in SHADOW_PRICE_ABLATION_VARIANTS only."
    )

    # The ablation list keeps all three hedge modes runnable.
    # Whether explorer is included in paper *figures* is a separate
    # question (see PAPER_HEDGE_MODES); SHADOW_PRICE_ABLATION_VARIANTS
    # stays as the full set so ad-hoc rawcost+explorer ablations remain
    # available without code changes.
    assert SHADOW_PRICE_ABLATION_VARIANTS == [
        "original_lp_rawcost",
        "original_lp_rawcost_hedge",
        "original_lp_rawcost_explorer",
    ]


# ---------------------------------------------------------------------------
# Workload mapping: paper id <-> runner dataset id.
# ---------------------------------------------------------------------------


def test_workload_mapping_covers_all_paper_workloads():
    for workload in PAPER_WORKLOADS:
        assert workload in WORKLOAD_DATASET_IDS, (
            f"Paper workload {workload!r} has no entry in WORKLOAD_DATASET_IDS."
        )


def test_workload_mapping_resolves_to_runner_datasets():
    for paper_id in PAPER_WORKLOADS:
        runner_id = runner_dataset_id(paper_id)
        assert runner_id in TRACE_WORKLOAD_DATASETS, (
            f"Paper workload {paper_id!r} maps to {runner_id!r}, which is "
            f"not in lp_budget_eval.TRACE_WORKLOAD_DATASETS={TRACE_WORKLOAD_DATASETS!r}."
        )


def test_workload_mapping_rejects_unknown_id():
    with pytest.raises(ValueError):
        runner_dataset_id("not_a_workload")


# ---------------------------------------------------------------------------
# Distribution-agnostic P95 lookup: tiered SLO filter must accept the new
# Uniform / Normal families without crashing.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "family",
    [LatencyFamily.UNIFORM, LatencyFamily.NORMAL, LatencyFamily.HEAVY_TAIL],
)
def test_provider_p95_works_for_all_families(family):
    """provider_p95_at used to crash on non-LogNormal because it read
    .mu/.sigma directly. The fix uses dist.p95() — this test pins it."""
    scenario = make_scenario(ProviderSetup.JOINT_PROVIDER, family)
    for provider in scenario.providers:
        p95 = provider_p95_at(provider, now=0.0)
        # Sanity: P95 should be a finite, positive number larger than P50.
        assert p95 > 0.0
        assert p95 >= provider.true_p50_ms(0.0)


# ---------------------------------------------------------------------------
# Distribution shape ordering: heavy_tail tail >= normal tail >= uniform tail.
# ---------------------------------------------------------------------------


def test_tail_strictly_grows_across_families():
    p50 = 300.0
    uniform_p99 = make_ttft_distribution(LatencyFamily.UNIFORM, p50).p99()
    normal_p99 = make_ttft_distribution(LatencyFamily.NORMAL, p50).p99()
    heavy_p99 = make_ttft_distribution(LatencyFamily.HEAVY_TAIL, p50).p99()
    assert uniform_p99 < normal_p99 < heavy_p99


def test_normal_clipping_negligible_at_grid_p50_floor():
    """At the smallest P50 used in the grid (100 ms) and the default
    Normal shape (sigma/mean = 0.3), the 3-sigma lower bound stays well
    above MIN_LATENCY_MS so analytical moments remain trustworthy."""
    dist = make_ttft_distribution(LatencyFamily.NORMAL, 100.0)
    assert isinstance(dist, Normal)
    three_sigma_floor = dist.mean_ms - 3.0 * dist.sigma
    assert three_sigma_floor > 1.0


# ---------------------------------------------------------------------------
# Quantile API: open interval (0, 1) — endpoints rejected.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dist",
    [Uniform(100.0, 200.0), Normal(150.0, 30.0), LogNormal(5.0, 0.5)],
)
@pytest.mark.parametrize("bad_q", [0.0, 1.0, -0.1, 1.1])
def test_quantile_rejects_endpoints_and_out_of_range(dist, bad_q):
    """Quantile boundary contract: (0, 1) strict so callers don't
    silently propagate ``±inf`` or ``0`` from Normal/LogNormal."""
    with pytest.raises(ValueError):
        dist.quantile(bad_q)


@pytest.mark.parametrize(
    "dist",
    [Uniform(100.0, 200.0), Normal(150.0, 30.0), LogNormal(5.0, 0.5)],
)
def test_quantile_interior_matches_named_accessors(dist):
    """quantile(0.5)/0.95/0.99 must agree with p50/p95/p99 — the named
    accessors are the canonical fast paths."""
    assert dist.quantile(0.50) == pytest.approx(dist.p50(), rel=1e-9)
    assert dist.quantile(0.95) == pytest.approx(dist.p95(), rel=1e-9)
    assert dist.quantile(0.99) == pytest.approx(dist.p99(), rel=1e-9)


# ---------------------------------------------------------------------------
# LatencyDistribution Protocol: all three concrete classes match.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dist",
    [Uniform(100.0, 200.0), Normal(150.0, 30.0), LogNormal(5.0, 0.5)],
)
def test_concrete_distributions_satisfy_protocol(dist):
    from rwsim.world.distributions import LatencyDistribution

    # runtime_checkable Protocol — isinstance is structural.
    assert isinstance(dist, LatencyDistribution)
