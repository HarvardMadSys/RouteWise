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
