"""Unit tests for the effective-cost ablation harness."""

from __future__ import annotations

import pytest

from experiments.ablations.effective_cost import harness
from experiments.ablations.effective_cost.presets import (
    CONCURRENCY_ONLY_QUOTA_CURVE,
    DEFAULT_CONCURRENCY_CURVES,
    DEFAULT_P_VALUES,
    DEFAULT_QUOTA_CURVES,
    ablation_policy_name,
    make_ablation_presets,
    make_concurrency_ablation_presets,
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
    presets = make_ablation_presets(curves=DEFAULT_QUOTA_CURVES, alpha_values=(0.0,))

    assert tuple(scenarios) == tuple(f"quota__plan=chutes__n={value}" for value in q_values)
    assert len(scenarios) * len(presets) * 1 == 20


def test_repeated_concurrency_count_expands_one_scenario_per_n() -> None:
    counts = (6, 8, 10, 11, 12, 13, 14, 16)

    scenarios = harness.make_scenarios(
        phase=harness.PHASE_CONCURRENCY,
        concurrency_count=counts,
    )
    presets = make_concurrency_ablation_presets(
        concurrency_curves=DEFAULT_CONCURRENCY_CURVES,
        alpha_values=(0.0,),
    )

    assert tuple(scenarios) == tuple(
        f"concurrency__plan=featherless_premium__n={value}__model=sharegpt" for value in counts
    )
    assert len(scenarios) * len(presets) * 1 == 40
    scenario = scenarios["concurrency__plan=featherless_premium__n=12__model=sharegpt"]
    assert scenario.metadata["public_scenario"] == "concurrency"
    assert scenario.metadata["concurrency_plan"] == "featherless_premium"
    assert scenario.metadata["model"] == "sharegpt"
    assert scenario.metadata["model_class"] == "ge_70b"
    assert scenario.metadata["model_concurrency_cost"] == 4
    assert {provider.tier for provider in scenario.providers} == {
        ProviderTier.S_C,
        ProviderTier.S_A,
    }


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
    assert captured["retain_records"] is False
    assert all(parse_ablation_policy_name(policy)[2] == 0.0 for policy in captured["policies"])
    assert len(captured["scenarios"]) * len(captured["policies"]) * len(captured["seeds"]) == 20


def test_cli_phase_b_concurrency_grid_and_p_zero(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run_section(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(harness.common, "run_section", fake_run_section)

    assert (
        harness.main(
            [
                "--phase",
                "concurrency",
                "--concurrency-plan",
                "featherless_premium",
                "--model",
                "sharegpt",
                "--concurrency-count",
                "6",
                "--concurrency-count",
                "8",
                "--concurrency-count",
                "10",
                "--concurrency-count",
                "11",
                "--concurrency-count",
                "12",
                "--concurrency-count",
                "13",
                "--concurrency-count",
                "14",
                "--concurrency-count",
                "16",
                "--concurrency-curve",
                "util_linear_u",
                "--concurrency-curve",
                "exp_lu",
                "--concurrency-curve",
                "linear_lu",
                "--concurrency-curve",
                "constant_l",
                "--concurrency-curve",
                "constant_u",
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
        "concurrency__plan=featherless_premium__n=6__model=sharegpt",
        "concurrency__plan=featherless_premium__n=8__model=sharegpt",
        "concurrency__plan=featherless_premium__n=10__model=sharegpt",
        "concurrency__plan=featherless_premium__n=11__model=sharegpt",
        "concurrency__plan=featherless_premium__n=12__model=sharegpt",
        "concurrency__plan=featherless_premium__n=13__model=sharegpt",
        "concurrency__plan=featherless_premium__n=14__model=sharegpt",
        "concurrency__plan=featherless_premium__n=16__model=sharegpt",
    )
    parsed = [parse_ablation_policy_name(policy) for policy in captured["policies"]]
    assert [item[1] for item in parsed] == list(DEFAULT_CONCURRENCY_CURVES)
    assert {item[0] for item in parsed} == {CONCURRENCY_ONLY_QUOTA_CURVE}
    assert {item[2] for item in parsed} == {0.0}
    assert captured["seeds"] == (42,)
    assert captured["retain_records"] is False
    assert len(captured["scenarios"]) * len(captured["policies"]) * len(captured["seeds"]) == 40


def test_policies_for_phase_default_curve_grid() -> None:
    policies = harness.policies_for_phase()

    assert policies == tuple(
        ablation_policy_name(curve, p=DEFAULT_P_VALUES[0]) for curve in DEFAULT_QUOTA_CURVES
    )


def test_policies_for_phase_b_default_concurrency_curve_grid() -> None:
    policies = harness.policies_for_phase(phase=harness.PHASE_CONCURRENCY)

    assert policies == tuple(
        ablation_policy_name(
            CONCURRENCY_ONLY_QUOTA_CURVE,
            concurrency_curve=curve,
            p=DEFAULT_P_VALUES[0],
        )
        for curve in DEFAULT_CONCURRENCY_CURVES
    )


def test_presets_are_ablation_local_and_use_workload_envelope_sentinel() -> None:
    presets = make_ablation_presets(curves=("exp_lu",), alpha_values=(0.5,))
    preset = presets["effective_cost__q=exp_lu__c=constant_l__alpha50"]

    assert preset["policy"] == "LPOnlyAblationPolicy"
    assert preset["params"]["cost_envelope"] == common.WORKLOAD_COST_ENVELOPE


def test_phase_b_presets_sweep_concurrency_curve_only() -> None:
    presets = make_concurrency_ablation_presets(
        concurrency_curves=("util_linear_u", "exp_lu"),
        alpha_values=(0.0,),
    )

    assert tuple(presets) == (
        "effective_cost__q=exp_lu__c=util_linear_u__alpha0",
        "effective_cost__q=exp_lu__c=exp_lu__alpha0",
    )
    for preset in presets.values():
        assert preset["policy"] == "LPOnlyAblationPolicy"
        assert preset["params"]["quota_curve"] == CONCURRENCY_ONLY_QUOTA_CURVE
        assert preset["params"]["cost_envelope"] == common.WORKLOAD_COST_ENVELOPE


def test_policy_name_roundtrip() -> None:
    policy = ablation_policy_name("linear_lu", p=0.25)

    assert parse_ablation_policy_name(policy) == (
        "linear_lu",
        "constant_l",
        0.25,
    )


@pytest.mark.parametrize("phase", ["joint"])
def test_deferred_phases_raise(phase: str) -> None:
    with pytest.raises(ValueError, match="deferred"):
        harness.make_scenarios(phase=phase)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"qstar": (2, 2)}, "qstar sweep values must be unique"),
        ({"qstar": (0,)}, "qstar must be > 0"),
        (
            {"phase": harness.PHASE_CONCURRENCY, "concurrency_count": (8, 8)},
            "concurrency_count sweep values must be unique",
        ),
        (
            {"phase": harness.PHASE_CONCURRENCY, "concurrency_count": (0,)},
            "concurrency_count must be > 0",
        ),
    ],
)
def test_rejects_invalid_sweep_counts(kwargs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        harness.make_scenarios(**kwargs)
