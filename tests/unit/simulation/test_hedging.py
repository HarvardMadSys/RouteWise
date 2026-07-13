"""Tests for the §2.2 hedging simulator section."""

from __future__ import annotations

import csv
import json

import pytest

from experiments.simulation import hedging
from routewise_cli.main import main as routewise_main


def test_hedging_scenarios_match_section_contract():
    assert hedging.list_scenarios() == (
        "hedging_heavy_tail",
        "hedging_real_world_rw3",
        "hedging_real_world_rw8",
        "hedging_real_world_rw8_slo4000",
    )


def test_hedging_policy_set_uses_hedging_only_routewise():
    assert hedging.policies_for_section() == (
        "ablation_lp_only_alpha75",
        "ablation_lp_hedging_alpha75",
    )

    presets = hedging.make_policy_presets()
    assert presets["ablation_lp_only_alpha75"]["params"]["hedging"] is False
    assert presets["ablation_lp_only_alpha75"]["params"]["explorer"] is False
    assert presets["ablation_lp_only_alpha75"]["params"]["latency_profile_mode"] == "configured"
    assert presets["ablation_lp_hedging_alpha75"]["params"]["hedging"] == "probability_target"
    assert presets["ablation_lp_hedging_alpha75"]["params"]["explorer"] is False
    assert presets["ablation_lp_hedging_alpha75"]["params"]["latency_profile_mode"] == "configured"


def test_hedging_scenarios_reuse_latency_layer_with_section_slos():
    heavy = hedging.make_scenario("hedging_heavy_tail")
    rw3 = hedging.make_scenario("hedging_real_world_rw3")
    rw8 = hedging.make_scenario("hedging_real_world_rw8")
    rw8_slo4000 = hedging.make_scenario("hedging_real_world_rw8_slo4000")

    assert heavy.metadata["source_latency_scenario"] == "latency_layer_heavy_tail_half_overlap"
    assert heavy.metadata["latency_family"] == "heavy_tail"
    assert heavy.metadata["latency_anchor_kind"] == "mean"
    assert heavy.metadata["latency_anchor_ms"] == [100.0, 300.0, 1000.0]
    assert heavy.primary_slo_ms == pytest.approx(500.0)
    assert heavy.metadata["slo_ms"] == pytest.approx(500.0)
    assert heavy.metadata["backup_selection"] == "probability_target_non_primary"
    assert heavy.metadata["latency_profile_source"] == "configured_distribution"

    assert rw3.metadata["source_latency_scenario"] == "latency_layer_real_world"
    assert rw3.metadata["latency_family"] == "real_world"
    assert rw3.primary_slo_ms == pytest.approx(2000.0)
    assert rw3.metadata["slo_ms"] == pytest.approx(2000.0)

    assert rw8.metadata["source_latency_scenario"] is None
    assert rw8.metadata["latency_family"] == "real_world"
    assert rw8.metadata["real_world_pool"] == "rw8"
    assert len(rw8.providers) == 8
    assert len({provider.effective_input_cost_per_token for provider in rw8.providers}) == 1
    assert len({provider.effective_output_cost_per_token for provider in rw8.providers}) == 1
    assert rw8.primary_slo_ms == pytest.approx(2000.0)

    assert rw8_slo4000.metadata["source_latency_scenario"] is None
    assert rw8_slo4000.metadata["real_world_pool"] == "rw8"
    assert len(rw8_slo4000.providers) == 8
    assert rw8_slo4000.primary_slo_ms == pytest.approx(4000.0)
    assert rw8_slo4000.metadata["slo_ms"] == pytest.approx(4000.0)


def test_hedging_cli_writes_plot_ready_metrics_to_json_and_csv(tmp_path, require_burstgpt_data):
    output_dir = tmp_path / "hedging"

    assert (
        routewise_main(
            [
                "simulator",
                "hedging",
                "--scenario",
                "hedging_heavy_tail",
                "--seed",
                "42",
                "--max-requests",
                "12",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    rows = json.loads((output_dir / "summary.json").read_text())
    assert [row["policy"] for row in rows] == [
        "ablation_lp_only_alpha75",
        "ablation_lp_hedging_alpha75",
    ]

    baseline, hedging_row = rows
    assert baseline["hedging_policy_mode"] == "lp_only"
    assert baseline["backup_selection"] == "none"
    assert baseline["hedge_rate"] == 0.0
    assert hedging_row["hedging_policy_mode"] == "routewise_hedging"
    assert hedging_row["backup_selection"] == "probability_target_non_primary"
    assert hedging_row["learns_from_backup"] is False
    assert hedging_row["latency_profile_mode"] == "configured"
    assert "real_world_pool" in hedging_row

    for row in rows:
        assert row["slo_ms"] == 500.0
        assert "hedge_rate" in row
        assert "p99_ms" in row
        assert "mean_ttft_ms" in row
        assert "p50_ms" in row
        assert "cost_multiplier_vs_lp_only" in row
        assert row["latency_anchor_kind"] == "mean"
        assert row["latency_anchor_ms"] == [100.0, 300.0, 1000.0]

    with (output_dir / "summary.csv").open() as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 2
    assert csv_rows[1]["policy"] == "ablation_lp_hedging_alpha75"
    assert csv_rows[1]["backup_selection"] == "probability_target_non_primary"
    assert csv_rows[1]["latency_profile_mode"] == "configured"
    assert csv_rows[1]["slo_ms"] == "500.0"
    assert csv_rows[1]["latency_anchor_kind"] == "mean"
    assert csv_rows[1]["latency_anchor_ms"] == "[100.0, 300.0, 1000.0]"
