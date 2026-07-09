"""Tests for the section-based latency-layer simulator module."""

from __future__ import annotations

import csv
import json

import pytest

from experiments.simulation import latency_layer
from routewise.capacity import ProviderTier
from routewise.const import DEFAULT_PRIMARY_SLO_MS
from routewise_cli.main import main as routewise_main


def test_latency_layer_scenarios_match_section_contract():
    assert latency_layer.list_scenarios() == (
        "latency_layer_uniform_no_overlap",
        "latency_layer_uniform_half_overlap",
        "latency_layer_normal_no_overlap",
        "latency_layer_normal_half_overlap",
        "latency_layer_heavy_tail_no_overlap",
        "latency_layer_heavy_tail_half_overlap",
        "latency_layer_real_world",
    )


def test_latency_layer_policy_set_has_single_routewise_alpha_value():
    assert latency_layer.policies_for_section() == (
        "random",
        "greedy_latency",
        "ablation_lp_only_alpha75",
    )


def test_latency_layer_synthetic_provider_invariants_and_band_targets():
    scenarios = latency_layer.make_scenarios()

    for family in ("uniform", "normal", "heavy_tail"):
        no_overlap = scenarios[f"latency_layer_{family}_no_overlap"]
        half_overlap = scenarios[f"latency_layer_{family}_half_overlap"]

        for scenario in (no_overlap, half_overlap):
            assert [provider.tier for provider in scenario.providers] == [ProviderTier.S_A] * 3
            assert [provider.name for provider in scenario.providers] == [
                "api_fast",
                "api_medium",
                "api_slow",
            ]
            assert [provider.true_mean_ms() for provider in scenario.providers] == pytest.approx(
                [100.0, 300.0, 1000.0]
            )
            assert scenario.primary_slo_ms == pytest.approx(DEFAULT_PRIMARY_SLO_MS)
            assert scenario.metadata["mean_anchors_ms"] == [100.0, 300.0, 1000.0]
            assert len({provider.effective_input_cost_per_token for provider in scenario.providers}) == 1
            assert len({provider.effective_output_cost_per_token for provider in scenario.providers}) == 1
            assert "realised_tvo_fast_medium" not in scenario.metadata

        assert no_overlap.metadata["target_band_coverage_fast_medium"] == 0.0
        assert no_overlap.metadata["realised_band_coverage_fast_medium"] == pytest.approx(0.0)
        assert half_overlap.metadata["target_band_coverage_fast_medium"] == 0.5
        assert half_overlap.metadata["realised_band_coverage_fast_medium"] == pytest.approx(0.5)


def test_latency_layer_real_world_is_single_untargeted_scenario():
    scenario = latency_layer.make_scenario("latency_layer_real_world")

    assert scenario.metadata["latency_family"] == "real_world"
    assert scenario.metadata["overlap_label"] is None
    assert scenario.metadata["target_band_coverage_fast_medium"] is None
    assert "latency_layer_real_world_no_overlap" not in latency_layer.list_scenarios()
    assert "latency_layer_real_world_half_overlap" not in latency_layer.list_scenarios()


def test_latency_layer_cli_writes_band_metadata_to_json_and_csv(tmp_path):
    output_dir = tmp_path / "latency-layer"

    assert routewise_main(
        [
            "simulator",
            "latency-layer",
            "--scenario",
            "latency_layer_uniform_half_overlap",
            "--policy",
            "greedy_latency",
            "--seed",
            "42",
            "--max-requests",
            "10",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    rows = json.loads((output_dir / "summary.json").read_text())
    assert len(rows) == 1
    assert rows[0]["target_band_coverage_fast_medium"] == 0.5
    assert rows[0]["realised_band_coverage_fast_medium"] == pytest.approx(0.5)
    assert "realised_tvo_fast_medium" not in rows[0]

    with (output_dir / "summary.csv").open() as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 1
    assert csv_rows[0]["target_band_coverage_fast_medium"] == "0.5"
    assert "realised_tvo_fast_medium" not in csv_rows[0]
