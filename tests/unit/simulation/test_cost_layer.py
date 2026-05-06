"""Tests for the section-based cost-layer simulator module."""

from __future__ import annotations

import json

import pytest

from experiments.simulation import common, cost_layer
from experiments.subscriptions import load_subscription_plans
from routewise_cli.main import main as routewise_main
from rwsim.schemas import Request
from rwsim.world.capacity import ProviderTier


def test_cost_layer_scenarios_match_section_contract():
    scenarios = cost_layer.make_scenarios()

    assert tuple(scenarios) == (
        "cost_layer_uniform",
        "cost_layer_normal",
        "cost_layer_heavy_tail",
        "cost_layer_real_world",
        "quota__plan=chutes__n=1",
        "quota__plan=chutes__n=2",
        "quota__plan=chutes__n=3",
        "quota__plan=chutes__n=4",
        "quota__plan=chutes__n=5",
        "quota__plan=chutes__n=6",
        "quota__plan=chutes__n=8",
        "cost_layer_concurrency_c1",
        "cost_layer_concurrency_c2",
        "cost_layer_concurrency_c3",
        "cost_layer_concurrency_c4",
    )
    assert "quota" in cost_layer.list_scenarios()
    assert "cost_layer_quota_q1" not in cost_layer.list_scenarios()


def test_cost_layer_make_scenario_rebuilds_real_world_by_name():
    left = cost_layer.make_scenario("cost_layer_real_world")
    right = cost_layer.make_scenario("cost_layer_real_world")

    assert [provider.name for provider in left.providers] == [
        provider.name for provider in right.providers
    ]


def test_cost_layer_on_demand_providers_hold_latency_constant_and_vary_cost():
    scenario = cost_layer.make_scenarios()["cost_layer_normal"]

    assert [provider.tier for provider in scenario.providers] == [ProviderTier.S_A] * 3
    assert [provider.true_p50_ms() for provider in scenario.providers] == [300.0, 300.0, 300.0]
    assert [provider.cost_per_token for provider in scenario.providers] == [1e-6, 2e-6, 4e-6]
    assert [provider.effective_input_cost_per_token for provider in scenario.providers] == [
        1e-6,
        2e-6,
        4e-6,
    ]
    assert [provider.effective_output_cost_per_token for provider in scenario.providers] == [
        5e-6,
        10e-6,
        20e-6,
    ]


def test_cost_layer_real_world_uses_one_pooled_latency_distribution():
    scenario = cost_layer.make_scenarios()["cost_layer_real_world"]

    assert [provider.tier for provider in scenario.providers] == [ProviderTier.S_A] * 3
    assert [provider.cost_per_token for provider in scenario.providers] == [1e-6, 2e-6, 4e-6]
    assert [provider.effective_output_cost_per_token for provider in scenario.providers] == [
        5e-6,
        10e-6,
        20e-6,
    ]
    assert len({id(provider.ttft_dist) for provider in scenario.providers}) == 1
    assert scenario.providers[0].ttft_dist.label == "qwen3_24h/rw8_pooled"
    assert 500.0 < scenario.providers[0].true_p50_ms() < 1500.0
    assert scenario.providers[0].true_p99_ms() > scenario.providers[0].true_p50_ms()


def test_quota_scenario_allows_explicit_exploratory_subscription_counts():
    scenarios = cost_layer.make_scenarios(
        subscription_counts=(40,),
    )
    scenario = scenarios["quota__plan=chutes__n=40"]

    assert scenario.metadata["subscription_count"] == 40
    assert scenario.providers[0].quota.size == 200000


def test_quota_scenario_can_use_real_world_latency_family():
    scenario = cost_layer.make_scenario(
        "quota",
        subscription_plan="chutes",
        subscription_count=2,
        quota_latency_family="real_world",
    )

    assert scenario.name == "quota__plan=chutes__n=2__latency=real_world"
    assert scenario.metadata["latency_family"] == "real_world"
    assert len({id(provider.ttft_dist) for provider in scenario.providers}) == 1
    assert scenario.providers[0].ttft_dist.label == "qwen3_24h/rw8_pooled"


def test_cost_layer_policy_surface_disables_explorer_and_greedy_latency():
    policies = cost_layer.policies_for_section((0.0, 0.75, 1.0))
    presets = common.make_routewise_presets(p_values=(0.0, 0.75, 1.0), include_hedging=False)

    assert policies == (
        "greedy_cost",
        "random",
        "offline",
        "ablation_lp_only_p0",
        "ablation_lp_only_p75",
        "ablation_lp_only_p100",
    )
    assert "greedy_latency" not in policies
    assert "routewise" not in presets
    assert presets["ablation_lp_only_p75"]["params"] == {
        "hedging": False,
        "explorer": False,
        "p": 0.75,
        "cost_envelope": common.WORKLOAD_COST_ENVELOPE,
    }


def test_workload_cost_envelope_uses_cheapest_api_request_cost():
    scenario = cost_layer.make_scenario(
        "quota",
        subscription_plan="chutes",
        subscription_count=1,
    )
    requests = [
        Request(id=1, timestamp=0.0, request_tokens=1, response_tokens=1, total_tokens=2),
        Request(
            id=2,
            timestamp=1.0,
            request_tokens=1000,
            response_tokens=1000,
            total_tokens=2000,
        ),
    ]

    L, U = common.workload_cost_envelope(
        scenario.providers,
        requests,
        lower_percentile=0.0,
        upper_percentile=100.0,
    )

    assert pytest.approx(6e-6) == L
    assert pytest.approx(0.006) == U


def test_offline_cost_baseline_uses_cheapest_api_when_no_capacity_provider():
    scenario = cost_layer.make_scenarios()["cost_layer_uniform"]
    requests = common.load_workload(max_requests=3)

    run = cost_layer.run_offline_policy(scenario, requests, seed=42)

    assert run.policy == "offline"
    assert run.provider_fractions() == {"api_cheap": 1.0}
    assert run.mean_cost_usd() == sum(
        request.request_tokens * 1e-6 + request.response_tokens * 5e-6 for request in requests
    ) / len(requests)


def test_offline_cost_baseline_uses_quota_for_highest_cost_requests():
    scenario = cost_layer.make_scenario(
        "quota",
        subscription_plan="chutes",
        subscription_count=1,
    )
    requests = common.load_workload(max_requests=5)

    run = cost_layer.run_offline_policy(scenario, requests, seed=42)

    assert run.policy == "offline"
    assert run.provider_fractions() == {"chutes_quota": 1.0}
    assert run.mean_cost_usd() == 0.0


def test_offline_cost_baseline_can_use_concurrency_capacity():
    scenario = cost_layer.make_scenarios()["cost_layer_concurrency_c1"]
    requests = common.load_workload(max_requests=5)

    run = cost_layer.run_offline_policy(scenario, requests, seed=42)

    assert run.policy == "offline"
    assert run.provider_fractions() == {"concurrency_1": 1.0}
    assert run.mean_cost_usd() == 0.0


def test_routewise_simulator_list_only_registers_runnable_sections(capsys):
    assert routewise_main(["simulator", "list"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["sections"][0]["name"] == "cost-layer"
    assert payload["sections"][0]["description"] == "paper §3.2 — same latency / different cost"
    assert "cost_layer_uniform" in payload["sections"][0]["scenarios"]
    assert "cost_layer_real_world" in payload["sections"][0]["scenarios"]
    assert "quota" in payload["sections"][0]["scenarios"]
    assert "cost_layer_quota_q1" not in payload["sections"][0]["scenarios"]
    assert "offline" in payload["sections"][0]["policies"]
    assert "ablation_lp_only_p75" in payload["sections"][0]["policies"]


def test_subscription_plan_loader_validates_and_exposes_chutes():
    plan = load_subscription_plans()["chutes"]

    assert plan.display_name == "Chutes"
    assert plan.monthly_fee_usd == 20.0
    assert plan.quota_windows[0].quota_requests == 5000
    assert plan.quota_windows[0].quota_window_sec == 86400
    assert plan.subscription_counts == (1, 2, 3, 4, 5, 6, 8)


def test_subscription_plan_loader_exposes_minimax_tiers_with_quota_windows():
    plans = load_subscription_plans()

    assert plans["minimax_subscription_starter"].monthly_fee_usd == 10.0
    assert plans["minimax_subscription_plus"].monthly_fee_usd == 20.0
    assert plans["minimax_subscription_max"].monthly_fee_usd == 50.0
    assert [window.name for window in plans["minimax_subscription_plus"].quota_windows] == [
        "five_hour",
        "weekly_allowance",
    ]
    assert [
        window.quota_requests for window in plans["minimax_subscription_plus"].quota_windows
    ] == [4500, 45000]


def test_subscription_plan_loader_exposes_featherless_premium_concurrency():
    plans = load_subscription_plans()
    plan = plans["featherless_premium"]

    assert "featherless_scale" not in plans
    assert plan.tier == "concurrency"
    assert plan.monthly_fee_usd == 25.0
    assert plan.concurrency_allotment == 4
    assert dict(plan.model_concurrency_costs_by_class) == {
        "le_15b": 1,
        "24_34b": 2,
        "ge_70b": 4,
    }
    assert plan.resolve_model_class("sharegpt") == "ge_70b"
    sharegpt = plan.resolve_model_class_with_cost("sharegpt")
    assert sharegpt is not None
    assert sharegpt.model_class == "ge_70b"
    assert sharegpt.cost == 4
    assert sharegpt.matched_via == "override"
    assert plan.concurrency_cost_for_model("sharegpt") == sharegpt.cost
    assert plan.resolve_model_class("qwen3-coder-30b") == "24_34b"
    override = plan.resolve_model_class_with_cost("qwen3-coder-30b")
    assert override is not None
    assert override.model_class == "24_34b"
    assert override.cost == 2
    assert override.matched_via == "override"
    assert plan.concurrency_cost_for_model("qwen3-coder-30b") == override.cost
    assert plan.resolve_model_class("unknown-model") is None
    assert plan.resolve_model_class_with_cost("unknown-model") is None
    with pytest.raises(ValueError, match="no concurrency model class resolved"):
        plan.concurrency_cost_for_model("unknown-model")
    assert plan.subscription_counts == (1, 2, 3, 4)
    assert plan.eligible_sections == ("cost_layer_concurrency",)


def test_subscription_plan_loader_rejects_missing_quota_size(tmp_path):
    path = tmp_path / "plans.yaml"
    path.write_text(
        """
plans:
  bad:
    monthly_fee_usd: 10
    quota_windows:
      - name: daily
        quota_window_sec: 86400
    subscription_counts: [1]
    eligible_sections: [cost_layer_quota]
    cost_claim_allowed: true
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="quota_requests"):
        load_subscription_plans(path)


def test_subscription_plan_loader_rejects_cost_claim_without_fee(tmp_path):
    path = tmp_path / "plans.yaml"
    path.write_text(
        """
plans:
  bad:
    monthly_fee_usd: null
    quota_windows:
      - name: daily
        quota_requests: 5000
        quota_window_sec: 86400
    subscription_counts: [1]
    eligible_sections: [cost_layer_quota]
    cost_claim_allowed: true
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="monthly_fee_usd"):
        load_subscription_plans(path)


def test_subscription_plan_loader_rejects_concurrency_plan_without_allotment(tmp_path):
    path = tmp_path / "plans.yaml"
    path.write_text(
        """
plans:
  bad:
    tier: concurrency
    monthly_fee_usd: 25
    model_concurrency_costs_by_class:
      ge_70b: 4
    subscription_counts: [1]
    eligible_sections: [cost_layer_concurrency]
    cost_claim_allowed: true
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="concurrency_allotment"):
        load_subscription_plans(path)


def test_subscription_plan_loader_rejects_unknown_override_model_class(tmp_path):
    path = tmp_path / "plans.yaml"
    path.write_text(
        """
plans:
  bad:
    tier: concurrency
    monthly_fee_usd: 25
    concurrency_allotment: 4
    model_concurrency_costs_by_class:
      ge_70b: 4
    model_class_overrides:
      sharegpt: missing
    subscription_counts: [1]
    eligible_sections: [cost_layer_concurrency]
    cost_claim_allowed: true
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model_class_overrides"):
        load_subscription_plans(path)


def test_make_quota_provider_aggregates_subscription_count_into_one_quota():
    plan = load_subscription_plans()["chutes"]

    provider = common.make_quota_provider(
        "chutes_quota",
        plan=plan,
        subscription_count=2,
    )

    assert provider.tier == ProviderTier.S_Q
    assert provider.quota is not None
    assert provider.quota.size == 10000
    assert provider.quota.window_sec == 86400


def test_make_quota_provider_rejects_subscription_count_without_plan():
    with pytest.raises(ValueError, match="subscription_count requires plan"):
        common.make_quota_provider(
            "manual_quota",
            quota_size=2000,
            subscription_count=2,
        )


def test_minimax_quota_plan_uses_composite_quota_state():
    scenario = cost_layer.make_scenario(
        "quota",
        subscription_plan="minimax_subscription_plus",
        subscription_count=2,
    )

    provider = scenario.providers[0]
    assert provider.name == "minimax_subscription_plus_quota"
    assert provider.tier == ProviderTier.S_Q
    assert provider.quota is not None
    assert [window.size for window in provider.quota.windows] == [9000, 90000]
    assert [window.window_sec for window in provider.quota.windows] == [18000, 604800]
    assert scenario.metadata["quota_windows"] == [
        {
            "name": "five_hour",
            "quota_requests": 4500,
            "quota_window_sec": 18000,
            "aggregate_quota_requests": 9000,
        },
        {
            "name": "weekly_allowance",
            "quota_requests": 45000,
            "quota_window_sec": 604800,
            "aggregate_quota_requests": 90000,
        },
    ]


def test_quota_saturated_flag_is_window_based():
    plan = load_subscription_plans()["chutes"]
    within_quota = [
        Request(id=index, timestamp=0.0, request_tokens=1, response_tokens=1, total_tokens=2)
        for index in range(5000)
    ]
    over_quota = [
        Request(id=index, timestamp=0.0, request_tokens=1, response_tokens=1, total_tokens=2)
        for index in range(5001)
    ]

    assert common.quota_saturated_in_trace(
        plan,
        subscription_count=1,
        requests=within_quota,
    )
    assert not common.quota_saturated_in_trace(
        plan,
        subscription_count=1,
        requests=over_quota,
    )


def test_subscription_summary_adds_fixed_fee_only_at_section_layer():
    scenario = cost_layer.make_scenario(
        "quota",
        subscription_plan="chutes",
        subscription_count=2,
    )
    requests = [
        Request(id=0, timestamp=0.0, request_tokens=1, response_tokens=1, total_tokens=2),
        Request(
            id=1,
            timestamp=86400.0,
            request_tokens=10,
            response_tokens=10,
            total_tokens=20,
        ),
    ]
    run = cost_layer.run_offline_policy(scenario, requests, seed=42)

    row = common.summarize_runs(
        scenario=scenario,
        policy="offline",
        seeds=(42,),
        runs=[run],
        requests=requests,
    )

    assert row["api_cost_usd"] == 0.0
    assert row["run_count"] == 1
    assert row["subscription_fixed_cost_usd"] == pytest.approx(20.0 * 2 / 30.0)
    assert row["subscription_fixed_cost_usd_per_run"] == pytest.approx(
        row["subscription_fixed_cost_usd"]
    )
    assert row["total_cost_usd"] == pytest.approx(
        row["api_cost_usd"] + row["subscription_fixed_cost_usd"]
    )
    assert row["total_cost_usd_per_run"] == pytest.approx(row["total_cost_usd"])
    assert row["mean_api_cost_usd"] == 0.0
    assert row["mean_total_cost_usd"] == pytest.approx(row["total_cost_usd"] / 2)
    assert row["trace_paper_grade"] is False
    assert row["quota_saturated_in_trace"] is True

    two_seed_row = common.summarize_runs(
        scenario=scenario,
        policy="offline",
        seeds=(42, 43),
        runs=[run, run],
        requests=requests,
    )
    assert two_seed_row["run_count"] == 2
    assert two_seed_row["subscription_fixed_cost_usd"] == pytest.approx(
        2 * row["subscription_fixed_cost_usd"]
    )
    assert two_seed_row["subscription_fixed_cost_usd_per_run"] == pytest.approx(
        row["subscription_fixed_cost_usd_per_run"]
    )
    assert two_seed_row["total_cost_usd"] == pytest.approx(
        two_seed_row["api_cost_usd"]
        + two_seed_row["subscription_fixed_cost_usd"]
    )
    assert two_seed_row["total_cost_usd_per_run"] == pytest.approx(
        row["total_cost_usd_per_run"]
    )
    assert two_seed_row["mean_total_cost_usd"] == pytest.approx(
        two_seed_row["total_cost_usd"] / 4
    )


@pytest.mark.slow
@pytest.mark.integration
def test_cost_layer_parallel_run_section_matches_serial(tmp_path):
    scenario = cost_layer.make_scenario("cost_layer_uniform")
    policies = ("greedy_cost", "random")
    seeds = (42, 43)
    presets = common.make_routewise_presets(p_values=())

    serial_rows = common.run_section(
        section_name=cost_layer.SECTION_NAME,
        scenarios={scenario.name: scenario},
        policies=policies,
        presets=presets,
        seeds=seeds,
        workload_dataset="burstgpt",
        max_requests=100,
        output_dir=tmp_path / "serial",
        jobs=1,
    )
    parallel_rows = common.run_section(
        section_name=cost_layer.SECTION_NAME,
        scenarios={scenario.name: scenario},
        policies=policies,
        presets=presets,
        seeds=seeds,
        workload_dataset="burstgpt",
        max_requests=100,
        output_dir=tmp_path / "parallel",
        jobs=2,
        parallel_cell_runner=cost_layer.run_cost_layer_cell,
    )

    _assert_rows_close(parallel_rows, serial_rows)
    assert (tmp_path / "parallel" / "ttft_histograms.json").exists()
    assert (tmp_path / "parallel" / "ttft_histograms_by_seed.json").exists()
    per_seed = json.loads((tmp_path / "parallel" / "ttft_histograms_by_seed.json").read_text())
    assert len(per_seed) == len(policies) * len(seeds)


@pytest.mark.slow
@pytest.mark.integration
def test_cost_layer_cli_accepts_jobs(tmp_path):
    output_dir = tmp_path / "cli"

    assert routewise_main(
        [
            "simulator",
            "cost-layer",
            "--scenario",
            "cost_layer_uniform",
            "--workload",
            "burstgpt",
            "--max-requests",
            "100",
            "--policy",
            "greedy_cost",
            "--policy",
            "random",
            "--jobs",
            "2",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["jobs"] == 2
    assert metadata["execution_mode"] == "parallel"
    assert metadata["processed_requests_per_cell"] == 100


def _assert_rows_close(actual: list[dict], expected: list[dict]) -> None:
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected, strict=True):
        assert actual_row.keys() == expected_row.keys()
        for key, expected_value in expected_row.items():
            actual_value = actual_row[key]
            if isinstance(expected_value, float):
                assert actual_value == pytest.approx(expected_value, rel=1e-12, abs=1e-12)
            else:
                assert actual_value == expected_value
