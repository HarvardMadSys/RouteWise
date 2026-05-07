"""Unit tests for the effective-cost ablation harness."""

from __future__ import annotations

import pytest

from experiments.ablations.effective_cost import harness
from experiments.ablations.effective_cost.presets import (
    DEFAULT_P_VALUES,
    DEFAULT_QUOTA_CURVES,
    ablation_policy_name,
    make_ablation_presets,
    parse_ablation_policy_name,
)
from experiments.simulation import common
from rwsim.world.capacity import ProviderTier


def test_default_phase_a_scenario_is_locked_to_qstar_16_heavy_tail() -> None:
    scenarios = harness.make_scenarios()

    assert tuple(scenarios) == ("quota__plan=chutes__n=16",)
    scenario = scenarios["quota__plan=chutes__n=16"]
    assert scenario.metadata["public_scenario"] == "quota"
    assert scenario.metadata["subscription_plan"] == "chutes"
    assert scenario.metadata["subscription_count"] == 16
    assert scenario.metadata["latency_family"] == "heavy_tail"
    assert {provider.tier for provider in scenario.providers} == {
        ProviderTier.S_Q,
        ProviderTier.S_A,
    }


def test_repeated_qstar_expands_one_scenario_per_q() -> None:
    q_values = (2, 4, 8, 12, 16)

    scenarios = harness.make_scenarios(qstar=q_values)
    presets = make_ablation_presets(curves=DEFAULT_QUOTA_CURVES, p_values=(0.0,))

    assert tuple(scenarios) == tuple(f"quota__plan=chutes__n={value}" for value in q_values)
    assert len(scenarios) * len(presets) * 1 == 20


def test_cli_repeated_qstar_and_p_zero_builds_qsweep_grid(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run_section(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(harness.common, "run_section", fake_run_section)

    assert (
        harness.main(
            [
                "--curve",
                "exp_lu",
                "--curve",
                "linear_lu",
                "--curve",
                "constant_l",
                "--curve",
                "constant_u",
                "--qstar",
                "2",
                "--qstar",
                "4",
                "--qstar",
                "8",
                "--qstar",
                "12",
                "--qstar",
                "16",
                "--p",
                "0",
                "--seed",
                "42",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )

    assert tuple(captured["scenarios"]) == (
        "quota__plan=chutes__n=2",
        "quota__plan=chutes__n=4",
        "quota__plan=chutes__n=8",
        "quota__plan=chutes__n=12",
        "quota__plan=chutes__n=16",
    )
    assert len(captured["policies"]) == 4
    assert captured["seeds"] == (42,)
    assert all(parse_ablation_policy_name(policy)[2] == 0.0 for policy in captured["policies"])
    assert len(captured["scenarios"]) * len(captured["policies"]) * len(captured["seeds"]) == 20


def test_policies_for_phase_default_curve_grid() -> None:
    policies = harness.policies_for_phase()

    assert policies == tuple(
        ablation_policy_name(curve, p=DEFAULT_P_VALUES[0]) for curve in DEFAULT_QUOTA_CURVES
    )


def test_presets_are_ablation_local_and_use_workload_envelope_sentinel() -> None:
    presets = make_ablation_presets(curves=("exp_lu",), p_values=(0.5,))
    preset = presets["effective_cost__q=exp_lu__c=legacy_linear_u__p50"]

    assert preset["policy"] == "LPOnlyAblationPolicy"
    assert preset["params"]["cost_envelope"] == common.WORKLOAD_COST_ENVELOPE


def test_policy_name_roundtrip() -> None:
    policy = ablation_policy_name("linear_lu", p=0.25)

    assert parse_ablation_policy_name(policy) == (
        "linear_lu",
        "legacy_linear_u",
        0.25,
    )


@pytest.mark.parametrize("phase", ["concurrency", "joint"])
def test_non_quota_phases_are_deferred(phase: str) -> None:
    with pytest.raises(ValueError, match="deferred"):
        harness.make_scenarios(phase=phase)
