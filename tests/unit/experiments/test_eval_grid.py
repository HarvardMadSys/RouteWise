"""Pytest coverage for the eval-grid factory."""

from __future__ import annotations

import pytest

from experiments.simulation.eval_grid import (
    LatencyFamily,
    PAPER_GRID_VARIANTS,
    PAPER_WORKLOADS,
    ProviderSetup,
    WORKLOAD_DATASET_IDS,
    assert_grid_invariants,
    make_eval_grid_scenarios,
    make_scenario,
    make_stage3_capacity_scenarios,
    make_ttft_distribution,
    runner_dataset_id,
    stage3_capacity_scenario_name,
)
from experiments.simulation.lp_budget_eval import MAIN_VARIANTS, TRACE_WORKLOAD_DATASETS
from rwsim.world.distributions import LatencyDistribution, LogNormal, Normal, Uniform
from rwsim.world.providers import ProviderTier


def test_grid_has_twelve_scenarios():
    """4 stages x 3 distributions = 12 scenarios."""
    grid = make_eval_grid_scenarios()
    assert len(grid) == 12
    expected_names = {
        f"grid_{stage.value}_{family.value}"
        for stage in ProviderSetup
        for family in LatencyFamily
    }
    assert set(grid) == expected_names


def test_grid_structural_invariants_pass():
    assert_grid_invariants()


def test_stage0_same_latency_distinct_costs():
    for family in LatencyFamily:
        scenario = make_scenario(ProviderSetup.SAME_LATENCY, family)
        p50s = {p.ttft_dist.p50() for p in scenario.providers}
        costs = {p.cost_per_token for p in scenario.providers}
        assert len(p50s) == 1
        assert len(costs) >= 2


def test_stage1_same_cost():
    for family in LatencyFamily:
        scenario = make_scenario(ProviderSetup.SAME_COST, family)
        costs = {p.cost_per_token for p in scenario.providers}
        assert len(costs) == 1


def test_stage2_distinct_costs():
    for family in LatencyFamily:
        scenario = make_scenario(ProviderSetup.COST_LATENCY_TRADEOFF, family)
        costs = {p.cost_per_token for p in scenario.providers}
        assert len(costs) >= 3


def test_stage3_has_all_three_tiers():
    for family in LatencyFamily:
        scenario = make_scenario(ProviderSetup.JOINT_PROVIDER, family)
        tiers = {p.tier for p in scenario.providers}
        assert {ProviderTier.S_A, ProviderTier.S_Q, ProviderTier.S_C}.issubset(tiers)


def test_stage3_capacity_overrides_set_quota_and_concurrency():
    scenario = make_scenario(
        ProviderSetup.JOINT_PROVIDER,
        LatencyFamily.HEAVY_TAIL,
        stage3_quota_size=5000,
        stage3_concurrency_limit=8,
    )
    assert scenario.name == "grid_joint_provider_heavy_tail_q5000_c8"

    quota_provider = next(p for p in scenario.providers if p.tier is ProviderTier.S_Q)
    concurrency_provider = next(
        p for p in scenario.providers if p.tier is ProviderTier.S_C
    )
    assert quota_provider.quota is not None
    assert quota_provider.quota.size == 5000
    assert concurrency_provider.concurrency is not None
    assert concurrency_provider.concurrency.limit == 8


def test_stage3_capacity_overrides_reject_non_stage3():
    with pytest.raises(ValueError):
        make_scenario(
            ProviderSetup.SAME_LATENCY,
            LatencyFamily.HEAVY_TAIL,
            stage3_quota_size=5000,
        )


def test_stage3_capacity_scenarios_cover_cross_product():
    scenarios = make_stage3_capacity_scenarios(
        quota_sizes=[1000, 5000],
        concurrency_limits=[2, 4],
        families=[LatencyFamily.HEAVY_TAIL],
    )
    expected = {
        stage3_capacity_scenario_name(
            LatencyFamily.HEAVY_TAIL,
            quota_size=quota,
            concurrency_limit=concurrency,
        )
        for quota in [1000, 5000]
        for concurrency in [2, 4]
    }
    assert set(scenarios) == expected


def test_paper_grid_variants_match_runner_main_variants():
    assert PAPER_GRID_VARIANTS == (
        "greedy_cost",
        "greedy_latency",
        "random",
        "ablation_lp_only",
        "ablation_lp_hedging",
        "routewise",
    )
    assert set(PAPER_GRID_VARIANTS).issubset(set(MAIN_VARIANTS))


def test_workload_mapping_covers_all_paper_workloads():
    for workload in PAPER_WORKLOADS:
        assert workload in WORKLOAD_DATASET_IDS


def test_workload_mapping_resolves_to_runner_datasets():
    for paper_id in PAPER_WORKLOADS:
        runner_id = runner_dataset_id(paper_id)
        assert runner_id in TRACE_WORKLOAD_DATASETS


def test_workload_mapping_rejects_unknown_id():
    with pytest.raises(ValueError):
        runner_dataset_id("not_a_workload")


def test_active_provider_distribution_honours_drift():
    fast = LogNormal(mu=4.6, sigma=0.3)
    slow = LogNormal(mu=6.9, sigma=0.3)
    shift = 3600.0
    scenario = make_scenario(ProviderSetup.JOINT_PROVIDER, LatencyFamily.HEAVY_TAIL)
    provider = scenario.providers[0]
    provider.ttft_dist = fast
    provider.shift_time = shift
    provider.ttft_dist_after = slow

    assert provider.true_p99_ms(shift - 1.0) == pytest.approx(fast.p99())
    assert provider.true_p99_ms(shift + 1.0) == pytest.approx(slow.p99())


def test_tail_strictly_grows_across_families():
    p50 = 300.0
    uniform_p99 = make_ttft_distribution(LatencyFamily.UNIFORM, p50).p99()
    normal_p99 = make_ttft_distribution(LatencyFamily.NORMAL, p50).p99()
    heavy_p99 = make_ttft_distribution(LatencyFamily.HEAVY_TAIL, p50).p99()
    assert uniform_p99 < normal_p99 < heavy_p99


@pytest.mark.parametrize(
    "dist",
    [Uniform(100.0, 200.0), Normal(150.0, 30.0), LogNormal(5.0, 0.5)],
)
@pytest.mark.parametrize("bad_q", [0.0, 1.0, -0.1, 1.1])
def test_quantile_rejects_endpoints_and_out_of_range(dist, bad_q):
    with pytest.raises(ValueError):
        dist.quantile(bad_q)


@pytest.mark.parametrize(
    "dist",
    [Uniform(100.0, 200.0), Normal(150.0, 30.0), LogNormal(5.0, 0.5)],
)
def test_quantile_interior_matches_named_accessors(dist):
    assert dist.quantile(0.50) == pytest.approx(dist.p50(), rel=1e-9)
    assert dist.quantile(0.95) == pytest.approx(dist.p95(), rel=1e-9)
    assert dist.quantile(0.99) == pytest.approx(dist.p99(), rel=1e-9)


@pytest.mark.parametrize(
    "dist",
    [Uniform(100.0, 200.0), Normal(150.0, 30.0), LogNormal(5.0, 0.5)],
)
def test_concrete_distributions_satisfy_protocol(dist):
    assert isinstance(dist, LatencyDistribution)
