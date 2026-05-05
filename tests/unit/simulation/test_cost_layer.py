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


def test_cost_layer_policy_surface_disables_explorer_and_greedy_latency():
    policies = cost_layer.policies_for_section((0.0, 0.75, 1.0))
    presets = common.make_routewise_presets(p_values=(0.0, 0.75, 1.0), include_hedging=False)

    assert policies == (
        "greedy_cost",
        "random",
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


def test_routewise_simulator_list_only_registers_runnable_sections(capsys):
    assert routewise_main(["simulator", "list"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["sections"][0]["name"] == "cost-layer"
    assert payload["sections"][0]["description"] == "paper §3.2 — same latency / different cost"
    assert "cost_layer_uniform" in payload["sections"][0]["scenarios"]
    assert "ablation_lp_only_p75" in payload["sections"][0]["policies"]
