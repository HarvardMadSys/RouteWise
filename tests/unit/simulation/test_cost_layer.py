"""Tests for the section-based cost-layer simulator module."""

from __future__ import annotations

import json

from experiments.simulation import common, cost_layer
from routewise_cli.main import main as routewise_main
from rwsim.world.capacity import ProviderTier


def test_cost_layer_scenarios_match_section_contract():
    scenarios = cost_layer.make_scenarios()

    assert tuple(scenarios) == (
        "cost_layer_uniform",
        "cost_layer_normal",
        "cost_layer_heavy_tail",
        "cost_layer_real_world",
        "cost_layer_quota_q1",
        "cost_layer_quota_q2",
        "cost_layer_quota_q3",
        "cost_layer_quota_q4",
        "cost_layer_concurrency_c1",
        "cost_layer_concurrency_c2",
        "cost_layer_concurrency_c3",
        "cost_layer_concurrency_c4",
    )


def test_cost_layer_on_demand_providers_hold_latency_constant_and_vary_cost():
    scenario = cost_layer.make_scenarios()["cost_layer_normal"]

    assert [provider.tier for provider in scenario.providers] == [ProviderTier.S_A] * 3
    assert [provider.true_p50_ms() for provider in scenario.providers] == [300.0, 300.0, 300.0]
    assert [provider.cost_per_token for provider in scenario.providers] == [1e-6, 2e-6, 4e-6]


def test_cost_layer_real_world_uses_one_pooled_latency_distribution():
    scenario = cost_layer.make_scenarios()["cost_layer_real_world"]

    assert [provider.tier for provider in scenario.providers] == [ProviderTier.S_A] * 3
    assert [provider.cost_per_token for provider in scenario.providers] == [1e-6, 2e-6, 4e-6]
    assert len({id(provider.ttft_dist) for provider in scenario.providers}) == 1
    assert scenario.providers[0].ttft_dist.label == "qwen3_24h/rw8_pooled"
    assert 500.0 < scenario.providers[0].true_p50_ms() < 1500.0
    assert scenario.providers[0].true_p99_ms() > scenario.providers[0].true_p50_ms()


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
    }


def test_offline_cost_baseline_uses_cheapest_api_when_no_capacity_provider():
    scenario = cost_layer.make_scenarios()["cost_layer_uniform"]
    requests = common.load_workload(max_requests=3)

    run = cost_layer.run_offline_policy(scenario, requests, seed=42)

    assert run.policy == "offline"
    assert run.provider_fractions() == {"api_cheap": 1.0}
    assert run.mean_cost_usd() == sum(
        request.total_tokens * 1e-6 for request in requests
    ) / len(requests)


def test_offline_cost_baseline_uses_quota_for_highest_cost_requests():
    scenario = cost_layer.make_scenarios()["cost_layer_quota_q1"]
    requests = common.load_workload(max_requests=5)

    run = cost_layer.run_offline_policy(scenario, requests, seed=42)

    assert run.policy == "offline"
    assert run.provider_fractions() == {"quota_1": 1.0}
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
    assert "offline" in payload["sections"][0]["policies"]
    assert "ablation_lp_only_p75" in payload["sections"][0]["policies"]
