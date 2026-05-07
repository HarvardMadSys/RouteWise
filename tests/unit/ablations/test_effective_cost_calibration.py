"""Unit tests for effective-cost envelope calibration ablations."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.ablations.effective_cost_calibration import harness
from experiments.ablations.effective_cost_calibration.envelope import (
    API_REFERENCES,
    DEFAULT_API_REFERENCE,
    DEFAULT_PERCENTILE_ENVELOPE,
    PERCENTILE_ENVELOPES,
    EnvelopeSpec,
    percentile_bounds,
    workload_api_reference_costs,
    workload_cost_envelope,
)
from experiments.simulation.common import make_api_provider, make_quota_provider
from routewise_cli.main import ABLATION_COMMANDS
from rwsim.schemas import Request
from rwsim.world.scenarios import ScenarioConfig


def _requests() -> list[Request]:
    return [
        Request(
            id=1,
            timestamp=0.0,
            request_tokens=100,
            response_tokens=100,
            total_tokens=200,
        ),
        Request(
            id=2,
            timestamp=1.0,
            request_tokens=200,
            response_tokens=200,
            total_tokens=400,
        ),
    ]


def _providers():
    return [
        make_api_provider(
            "api_cheap",
            cost_per_million_tokens=1.0,
            latency_family="heavy_tail",
        ),
        make_api_provider(
            "api_mid",
            cost_per_million_tokens=2.0,
            latency_family="heavy_tail",
        ),
        make_api_provider(
            "api_expensive",
            cost_per_million_tokens=4.0,
            latency_family="heavy_tail",
        ),
    ]


def test_reference_costs_use_selected_api_reference() -> None:
    providers = _providers()
    requests = _requests()

    assert workload_api_reference_costs(
        providers, requests, api_reference="cheapest_api"
    ) == pytest.approx([0.0006, 0.0012])
    assert workload_api_reference_costs(
        providers, requests, api_reference="median_api"
    ) == pytest.approx([0.0012, 0.0024])
    assert workload_api_reference_costs(
        providers, requests, api_reference="mean_api"
    ) == pytest.approx([0.0014, 0.0028])
    assert workload_api_reference_costs(
        providers, requests, api_reference="max_api"
    ) == pytest.approx([0.0024, 0.0048])


def test_percentile_envelope_names_map_to_bounds() -> None:
    assert percentile_bounds("p05_p95") == (5.0, 95.0)
    assert percentile_bounds("p10_p90") == (10.0, 90.0)
    assert percentile_bounds("p25_p75") == (25.0, 75.0)
    assert percentile_bounds("min_max") == (0.0, 100.0)


def test_workload_cost_envelope_applies_reference_and_percentile_pair() -> None:
    providers = _providers()
    requests = _requests()
    spec = EnvelopeSpec(api_reference="cheapest_api", percentile_envelope="p10_p90")

    L, U = workload_cost_envelope(providers, requests, spec=spec)

    values = np.asarray([0.0006, 0.0012], dtype=float)
    assert pytest.approx(float(np.percentile(values, 10.0))) == L
    assert pytest.approx(float(np.percentile(values, 90.0))) == U


def test_calibration_specs_default_runs_two_one_dimensional_sweeps() -> None:
    specs = harness.calibration_specs()

    assert len(specs) == len(PERCENTILE_ENVELOPES) + len(API_REFERENCES) - 1
    assert specs[0] == EnvelopeSpec(
        api_reference=DEFAULT_API_REFERENCE,
        percentile_envelope="p05_p95",
    )
    assert specs.count(
        EnvelopeSpec(
            api_reference=DEFAULT_API_REFERENCE,
            percentile_envelope=DEFAULT_PERCENTILE_ENVELOPE,
        )
    ) == 1
    assert specs[-1] == EnvelopeSpec(
        api_reference="max_api",
        percentile_envelope=DEFAULT_PERCENTILE_ENVELOPE,
    )


def test_calibration_specs_allow_custom_fixed_axis_for_one_dimensional_sweeps() -> None:
    assert harness.calibration_specs(
        sweep="percentile",
        api_references=("mean_api",),
        percentile_envelopes=("p10_p90", "min_max"),
    ) == (
        EnvelopeSpec(api_reference="mean_api", percentile_envelope="p10_p90"),
        EnvelopeSpec(api_reference="mean_api", percentile_envelope="min_max"),
    )
    assert harness.calibration_specs(
        sweep="reference",
        api_references=("cheapest_api", "max_api"),
        percentile_envelopes=("p25_p75",),
    ) == (
        EnvelopeSpec(api_reference="cheapest_api", percentile_envelope="p25_p75"),
        EnvelopeSpec(api_reference="max_api", percentile_envelope="p25_p75"),
    )


def test_calibration_policy_name_encodes_reference_envelope_curve_and_p() -> None:
    spec = EnvelopeSpec(api_reference="mean_api", percentile_envelope="p25_p75")

    assert harness.calibration_policy_name(spec, quota_curve="exp_lu", p=0.5) == (
        "effective_cost_calibration__ref=mean_api__env=p25_p75__q=exp_lu__p50"
    )


def test_build_calibration_policy_materializes_envelope() -> None:
    scenario = ScenarioConfig(
        name="test",
        description="test",
        providers=[
            make_quota_provider("quota", quota_size=10),
            *_providers(),
        ],
    )
    spec = EnvelopeSpec(api_reference="max_api", percentile_envelope="min_max")
    presets = harness.make_calibration_presets(specs=(spec,), p_values=(0.5,))
    policy_name = next(iter(presets))

    policy = harness.build_calibration_policy(
        policy_name,
        presets=presets,
        scenario=scenario,
        requests=_requests(),
        seed=42,
    )

    assert policy.cost_envelope == pytest.approx((0.0024, 0.0048))


def test_cli_default_grid_uses_q16_and_seven_calibrations(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run_section(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(harness.common, "run_section", fake_run_section)

    assert (
        harness.main(
            [
                "--max-requests",
                "10",
                "--seed",
                "42",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )

    assert tuple(captured["scenarios"]) == ("quota__plan=chutes__n=16",)
    assert len(captured["policies"]) == 7
    assert captured["workload_dataset"] == "burstgpt"
    assert captured["retain_records"] is False


def test_routewise_cli_registers_effective_cost_calibration() -> None:
    assert (
        ABLATION_COMMANDS["effective-cost-calibration"]
        == "experiments.ablations.effective_cost_calibration.harness"
    )
