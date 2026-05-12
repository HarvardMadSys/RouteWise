"""Tests for the §3 end-to-end simulator section."""

from __future__ import annotations

import csv
import json

import pytest

from experiments.simulation import end_to_end
from routewise_cli.main import main as routewise_main
from rwsim.world.capacity import ProviderTier


def test_end_to_end_scenarios_match_section_contract():
    assert end_to_end.list_scenarios() == (
        "end_to_end_rw3",
        "end_to_end_3sa_cost_tiers",
        "end_to_end_rw8",
    )


def test_end_to_end_rw3_uses_one_api_plus_quota_and_concurrency():
    scenario = end_to_end.make_scenario("end_to_end_rw3")

    assert scenario.metadata["public_scenario"] == "end_to_end"
    assert scenario.metadata["real_world_pool"] == "rw3"
    assert scenario.metadata["api_provider_count"] == 1
    assert scenario.metadata["subscription_plan"] == "chutes"
    assert scenario.metadata["subscription_count"] == 14
    assert scenario.metadata["concurrency_plan"] == "featherless_premium"
    assert scenario.metadata["concurrency_count"] == 8
    assert scenario.metadata["model"] == "qwen3-235b"
    assert scenario.metadata["model_class"] == "ge_70b"
    assert scenario.metadata["effective_concurrency_limit"] == 8
    assert scenario.metadata["slo_ms"] == pytest.approx(5000.0)
    assert [provider.tier for provider in scenario.providers] == [
        ProviderTier.S_Q,
        ProviderTier.S_C,
        ProviderTier.S_A,
    ]
    assert scenario.providers[0].ttft_dist.label == "minimax_m25_subscriptions/chutes"
    assert scenario.providers[1].ttft_dist.label == "minimax_m25_subscriptions/featherless"
    assert scenario.providers[2].name == "api_WandB"
    assert scenario.providers[2].ttft_dist.label == "qwen3_24h/WandB"


def test_end_to_end_cost_tiered_scenario_uses_synthetic_prices_and_real_latency():
    scenario = end_to_end.make_scenario("end_to_end_3sa_cost_tiers")

    assert scenario.metadata["real_world_pool"] == "rw8"
    assert scenario.metadata["api_provider_count"] == 3
    assert scenario.metadata["api_price_source"] == "cost_layer_synthetic_tiers"
    assert scenario.metadata["api_latency_provider_names"] == [
        "SiliconFlow",
        "Google",
        "WandB",
    ]
    assert len(scenario.providers) == 5
    assert [provider.tier for provider in scenario.providers] == [
        ProviderTier.S_Q,
        ProviderTier.S_C,
        ProviderTier.S_A,
        ProviderTier.S_A,
        ProviderTier.S_A,
    ]
    assert [provider.name for provider in scenario.providers[2:]] == [
        "api_A_slow_SiliconFlow",
        "api_B_mid_Google",
        "api_C_fast_WandB",
    ]
    assert [
        provider.effective_input_cost_per_token * 1_000_000
        for provider in scenario.providers[2:]
    ] == [1.0, 2.0, 4.0]
    assert [
        provider.effective_output_cost_per_token * 1_000_000
        for provider in scenario.providers[2:]
    ] == [5.0, 10.0, 20.0]
    assert [provider.ttft_dist.label for provider in scenario.providers[2:]] == [
        "qwen3_24h/SiliconFlow",
        "qwen3_24h/Google",
        "qwen3_24h/WandB",
    ]


def test_end_to_end_rw8_uses_full_api_pool_plus_capacity_tiers():
    scenario = end_to_end.make_scenario("end_to_end_rw8")

    assert scenario.metadata["real_world_pool"] == "minimax_m25_rw8"
    assert scenario.metadata["api_provider_count"] == 8
    assert scenario.metadata["api_price_source"] == "metadata_openrouter_price"
    assert len(scenario.providers) == 10
    assert [provider.tier for provider in scenario.providers[:2]] == [
        ProviderTier.S_Q,
        ProviderTier.S_C,
    ]
    assert [provider.tier for provider in scenario.providers[2:]] == [ProviderTier.S_A] * 8
    assert scenario.metadata["api_provider_names"] == [
        "api_Inceptron",
        "api_Friendli",
        "api_DeepInfra",
        "api_SambaNova",
        "api_Venice",
        "api_AtlasCloud",
        "api_Chutes",
        "api_SiliconFlow",
    ]
    assert [provider.ttft_dist.label for provider in scenario.providers[2:]] == [
        "minimax_m25_openrouter_24h/Inceptron",
        "minimax_m25_openrouter_24h/Friendli",
        "minimax_m25_openrouter_24h/DeepInfra",
        "minimax_m25_openrouter_24h/SambaNova",
        "minimax_m25_openrouter_24h/Venice",
        "minimax_m25_openrouter_24h/AtlasCloud",
        "minimax_m25_openrouter_24h/Chutes",
        "minimax_m25_openrouter_24h/SiliconFlow",
    ]
    assert [
        provider.effective_input_cost_per_token * 1_000_000
        for provider in scenario.providers[2:]
    ] == pytest.approx(
        [
            0.24,
            0.30,
            0.15,
            0.30,
            0.34,
            0.295,
            0.15,
            0.30,
        ]
    )
    assert [
        provider.effective_output_cost_per_token * 1_000_000
        for provider in scenario.providers[2:]
    ] == pytest.approx(
        [
            0.90,
            1.20,
            1.15,
            1.20,
            1.19,
            1.20,
            1.20,
            1.20,
        ]
    )


def test_end_to_end_prefix_cache_sets_api_cached_input_prices():
    scenario = end_to_end.make_scenario(
        "end_to_end_rw8",
        prefix_cache_enabled=True,
        cached_input_price_fraction=0.2,
    )

    assert scenario.metadata["prefix_cache_enabled"] is True
    assert scenario.metadata["cached_input_price_fraction"] is None
    assert (
        scenario.metadata["cached_input_price_source"]
        == "metadata_openrouter_input_cache_read"
    )
    assert scenario.providers[0].cached_input_cost_per_token is None
    assert scenario.providers[1].cached_input_cost_per_token is None
    cached_prices = [
        (
            None
            if provider.cached_input_cost_per_token is None
            else provider.cached_input_cost_per_token * 1_000_000
        )
        for provider in scenario.providers[2:]
    ]
    assert cached_prices[:3] == pytest.approx([0.030, 0.060, 0.030])
    assert cached_prices[3] is None
    assert cached_prices[4:] == pytest.approx([0.040, 0.060, 0.075, 0.030])


def test_end_to_end_policy_surface_covers_no_hedge_and_hedging_p_sweep():
    policies = end_to_end.policies_for_section((0.0, 0.75))
    presets = end_to_end.make_policy_presets((0.0, 0.75))

    assert policies == (
        "greedy_cost",
        "greedy_latency",
        "random",
        "ablation_lp_only_p0",
        "ablation_lp_only_p75",
        "ablation_lp_hedging_p0",
        "ablation_lp_hedging_p75",
    )
    assert presets["ablation_lp_only_p75"]["params"]["hedging"] is False
    assert presets["ablation_lp_only_p75"]["params"]["explorer"] is False
    assert presets["ablation_lp_only_p75"]["params"]["latency_profile_mode"] == "configured"
    assert presets["ablation_lp_hedging_p75"]["params"]["hedging"] == "probability_target"
    assert presets["ablation_lp_hedging_p75"]["params"]["explorer"] is False
    assert presets["ablation_lp_hedging_p75"]["params"]["latency_profile_mode"] == "configured"
    assert presets["ablation_lp_hedging_p75"]["params"]["slo_ms"] == pytest.approx(5000.0)


def test_end_to_end_cli_writes_plot_ready_metrics_to_json_and_csv(tmp_path):
    output_dir = tmp_path / "end_to_end"

    assert routewise_main(
        [
            "simulator",
            "end-to-end",
            "--scenario",
            "end_to_end_rw3",
            "--p",
            "0.75",
            "--policy",
            "ablation_lp_only_p75",
            "--policy",
            "ablation_lp_hedging_p75",
            "--seed",
            "42",
            "--max-requests",
            "12",
            "--prefix-cache-enabled",
            "--cached-input-price-fraction",
            "0.2",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    rows = json.loads((output_dir / "summary.json").read_text())
    assert [row["policy"] for row in rows] == [
        "ablation_lp_only_p75",
        "ablation_lp_hedging_p75",
    ]

    baseline, hedging_row = rows
    assert baseline["real_world_pool"] == "rw3"
    assert baseline["api_provider_count"] == 1
    assert baseline["routewise_p"] == 0.75
    assert baseline["hedging_enabled"] is False
    assert baseline["explorer_enabled"] is False
    assert baseline["latency_profile_mode"] == "configured"
    assert baseline["prefix_cache_enabled"] is True
    assert baseline["cached_input_price_fraction"] == 0.2
    assert "simulated_429_rate" not in baseline
    assert hedging_row["routewise_p"] == 0.75
    assert hedging_row["hedging_enabled"] is True
    assert hedging_row["explorer_enabled"] is False
    assert hedging_row["slo_ms"] == 5000.0

    with (output_dir / "summary.csv").open() as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 2
    assert csv_rows[1]["policy"] == "ablation_lp_hedging_p75"
    assert csv_rows[1]["real_world_pool"] == "rw3"
    assert csv_rows[1]["routewise_p"] == "0.75"
    assert csv_rows[1]["hedging_enabled"] == "True"
    assert csv_rows[1]["prefix_cache_enabled"] == "True"
    assert csv_rows[1]["cached_input_price_fraction"] == "0.2"
