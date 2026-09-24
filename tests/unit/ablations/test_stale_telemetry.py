"""Unit tests for the stale-telemetry robustness experiment."""

from __future__ import annotations

import csv
import json

import pytest

from experiments.ablations.stale_telemetry import harness
from experiments.ablations.stale_telemetry.harness import (
    SpikeTrainConfig,
    build_spike_schedule,
    classify_phase,
    parse_spike_artifact_label,
    select_spiked_provider,
    spike_artifact_label,
    spike_onsets,
    timeseries_bin,
    timeseries_bin_start_min,
)
from experiments.ablations.stale_telemetry.policy import StaleBeliefRouteWisePolicy
from experiments.ablations.stale_telemetry.presets import (
    make_presets,
    parse_policy_name,
    policy_name,
)
from llm_routewise.schemas import Request
from llm_routewise.sim.engine.state import SimulationState
from llm_routewise.sim.policies.routewise import RouteWisePolicy
from llm_routewise.sim.world.distributions import ScaledDistribution


def _requests(count: int, interval_sec: float) -> list[Request]:
    return [
        Request(
            id=index,
            timestamp=index * interval_sec,
            request_tokens=100,
            response_tokens=50,
            total_tokens=150,
        )
        for index in range(count)
    ]


# 30 minutes of trace, one spike per 10 minutes lasting 2 minutes, first at 5 min.
_CONFIG = SpikeTrainConfig(period_min=10.0, warmup_min=5.0, pre_onset_min=2.0)
_REQUESTS = _requests(count=181, interval_sec=10.0)


class TestLabels:
    def test_round_trip(self) -> None:
        for magnitude, duration in ((2.0, 10.0), (2.5, 0.25), (5.0, 30.0)):
            label = spike_artifact_label(magnitude, duration)
            assert parse_spike_artifact_label(label) == (magnitude, duration)

    def test_default_label(self) -> None:
        assert spike_artifact_label(3.0, 10.0) == "spike__mag=3__dur=10m"

    def test_rejects_foreign_labels(self) -> None:
        with pytest.raises(ValueError, match="scenario label"):
            parse_spike_artifact_label("rw8_shift__period=static__mag=3")


class TestSpikeTrain:
    def test_onsets_start_after_warmup_and_repeat(self) -> None:
        assert spike_onsets(
            anchor_sec=0.0, end_sec=7000.0, warmup_sec=1800.0, period_sec=3600.0
        ) == (
            1800.0,
            5400.0,
        )
        assert (
            spike_onsets(anchor_sec=0.0, end_sec=100.0, warmup_sec=1800.0, period_sec=3600.0) == ()
        )

    def test_schedule_alternates_degraded_and_baseline(self) -> None:
        base = harness.hedging.make_scenario("hedging_real_world_rw3").providers[0].ttft_dist
        schedule = build_spike_schedule(
            base,
            onsets=(1800.0, 5400.0),
            duration_sec=600.0,
            magnitude=3.0,
        )
        assert [entry[0] for entry in schedule] == [1800.0, 2400.0, 5400.0, 6000.0]
        assert isinstance(schedule[0][1], ScaledDistribution)
        assert schedule[0][1].scale == 3.0
        assert schedule[1][1] is base
        assert isinstance(schedule[2][1], ScaledDistribution)
        assert schedule[3][1] is base

    def test_scenario_spikes_only_the_fastest_provider(self) -> None:
        scenario = harness.make_scenario(
            spike_artifact_label(3.0, 2.0),
            requests=_REQUESTS,
            config=_CONFIG,
        )
        spiked = next(
            p for p in scenario.providers if p.name == scenario.metadata["spiked_provider"]
        )
        others = [p for p in scenario.providers if p is not spiked]
        baseline_mean = spiked.ttft_dist.mean()

        assert spiked.name == "api_fast"
        assert baseline_mean == min(p.ttft_dist.mean() for p in scenario.providers)
        assert scenario.metadata["spike_first_onset_sec"] == pytest.approx(300.0)
        assert scenario.metadata["spike_episode_count"] == 3
        assert scenario.metadata["spike_magnitude"] == 3.0
        assert scenario.metadata["spike_duration_min"] == 2.0
        assert scenario.metadata["real_world_pool"] == "rw3"
        assert scenario.metadata["public_scenario"] == "stale_telemetry"
        assert scenario.primary_slo_ms == pytest.approx(2000.0)
        # Degraded inside [onset, onset + duration), baseline elsewhere.
        assert spiked.true_mean_ms(299.0) == pytest.approx(baseline_mean)
        assert spiked.true_mean_ms(300.0) == pytest.approx(3.0 * baseline_mean)
        assert spiked.true_mean_ms(419.0) == pytest.approx(3.0 * baseline_mean)
        assert spiked.true_mean_ms(420.0) == pytest.approx(baseline_mean)
        assert spiked.true_mean_ms(900.0) == pytest.approx(3.0 * baseline_mean)
        for provider in others:
            assert provider.ttft_shift_schedule is None
            assert provider.true_mean_ms(300.0) == pytest.approx(provider.ttft_dist.mean())
        assert len({p.effective_input_cost_per_token for p in scenario.providers}) == 1

    def test_spiked_provider_override(self) -> None:
        scenario = harness.make_scenario(
            spike_artifact_label(2.0, 2.0),
            requests=_REQUESTS,
            config=SpikeTrainConfig(
                period_min=10.0,
                warmup_min=5.0,
                pre_onset_min=2.0,
                spiked_provider="api_slow",
            ),
        )
        assert scenario.metadata["spiked_provider"] == "api_slow"
        with pytest.raises(ValueError, match="not in scenario"):
            select_spiked_provider(scenario.providers, "nope")

    def test_grid_and_validation(self) -> None:
        scenarios = harness.make_scenarios(
            magnitudes=(2.0, 3.0),
            duration_minutes=(2.0,),
            requests=_REQUESTS,
            config=_CONFIG,
        )
        assert list(scenarios) == ["spike__mag=2__dur=2m", "spike__mag=3__dur=2m"]
        with pytest.raises(ValueError, match="magnitude must be > 1"):
            _CONFIG.validate_spike(1.0, 2.0)
        with pytest.raises(ValueError, match="spike duration must be in"):
            _CONFIG.validate_spike(3.0, 10.0)
        with pytest.raises(ValueError, match="pre_onset_min must leave"):
            _CONFIG.validate_spike(3.0, 9.0)
        with pytest.raises(ValueError, match="pre_onset_min must be in"):
            SpikeTrainConfig(period_min=10.0, pre_onset_min=10.0)
        with pytest.raises(ValueError, match="unknown provider pool"):
            SpikeTrainConfig(pool="rw99")
        with pytest.raises(ValueError, match="non-empty workload"):
            harness.make_scenario(spike_artifact_label(3.0, 2.0), requests=[], config=_CONFIG)


class TestPhases:
    def test_classify_phase_cycles(self) -> None:
        kwargs = {"first_onset_sec": 300.0, "period_sec": 600.0, "duration_sec": 120.0}
        assert classify_phase(0.0, **kwargs) == "baseline"
        assert classify_phase(299.9, **kwargs) == "baseline"
        assert classify_phase(300.0, **kwargs) == "spike"
        assert classify_phase(419.9, **kwargs) == "spike"
        assert classify_phase(420.0, **kwargs) == "post_spike"
        assert classify_phase(539.9, **kwargs) == "post_spike"
        assert classify_phase(540.0, **kwargs) == "baseline"
        assert classify_phase(899.9, **kwargs) == "baseline"
        assert classify_phase(900.0, **kwargs) == "spike"

    def test_timeseries_bins_align_on_onset_and_wrap(self) -> None:
        kwargs = {"first_onset_sec": 300.0, "period_sec": 600.0, "pre_onset_sec": 120.0}
        assert timeseries_bin(100.0, **kwargs) is None
        assert timeseries_bin(180.0, **kwargs) == 0
        assert timeseries_bin(300.0, **kwargs) == 2
        assert timeseries_bin(779.9, **kwargs) == 9
        assert timeseries_bin(780.0, **kwargs) == 0
        assert timeseries_bin_start_min(0, pre_onset_min=2.0) == pytest.approx(-2.0)
        assert timeseries_bin_start_min(2, pre_onset_min=2.0) == pytest.approx(0.0)


class TestStaleBelief:
    def test_frozen_view_ignores_the_spike_while_fresh_view_tracks_it(self) -> None:
        scenario = harness.make_scenario(
            spike_artifact_label(3.0, 2.0),
            requests=_REQUESTS,
            config=_CONFIG,
        )
        state = SimulationState.from_providers({p.name: p for p in scenario.providers}, now=330.0)
        spiked = state.providers["api_fast"]
        request = _REQUESTS[0]
        kwargs = {
            "hedging": False,
            "explorer": False,
            "slo_ms": 2000.0,
            "cost_envelope": (1e-6, 1e-3),
        }
        stale = StaleBeliefRouteWisePolicy(**kwargs)
        fresh = RouteWisePolicy(latency_profile_mode="configured", **kwargs)

        stale_view = stale._view(spiked, request, state)
        fresh_view = fresh._view(spiked, request, state)
        baseline_mean = spiked.ttft_dist.mean()
        assert stale_view.prior_ttft_mean_ms(330.0) == pytest.approx(baseline_mean)
        assert fresh_view.prior_ttft_mean_ms(330.0) == pytest.approx(3.0 * baseline_mean)
        assert stale_view.prior_ttft_cdf(2000.0, 330.0) == pytest.approx(
            spiked.ttft_dist.cdf(2000.0)
        )
        assert fresh_view.prior_ttft_cdf(2000.0, 330.0) < stale_view.prior_ttft_cdf(2000.0, 330.0)
        # Outside the spike both views agree.
        assert fresh_view.prior_ttft_mean_ms(200.0) == pytest.approx(baseline_mean)

    def test_stale_policy_requires_configured_mode(self) -> None:
        with pytest.raises(ValueError, match="configured"):
            StaleBeliefRouteWisePolicy(
                latency_profile_mode="observed",
                cost_envelope=(1e-6, 1e-3),
            )


class TestPresets:
    def test_names_and_params(self) -> None:
        presets = make_presets()
        assert tuple(presets) == (
            "routewise_lp_stale",
            "routewise_hedge_stale",
            "routewise_lp_observed15m",
            "routewise_hedge_observed15m",
            "routewise_lp_fresh",
            "routewise_hedge_fresh",
        )
        stale = presets["routewise_hedge_stale"]
        assert stale["policy"] == "StaleBeliefRouteWisePolicy"
        assert stale["params"]["hedging"] == "probability_target"
        assert stale["params"]["latency_profile_mode"] == "configured"
        assert stale["params"]["explorer"] is False
        observed = presets["routewise_lp_observed15m"]
        assert observed["policy"] == "RouteWisePolicy"
        assert observed["params"]["hedging"] is False
        assert observed["params"]["latency_profile_mode"] == "observed"
        assert observed["params"]["profile_window_sec"] == 900.0
        assert observed["params"]["explorer"] is True
        fresh = presets["routewise_hedge_fresh"]
        assert fresh["policy"] == "RouteWisePolicy"
        assert fresh["params"]["latency_profile_mode"] == "configured"
        assert "profile_window_sec" not in fresh["params"]

    def test_policy_name_round_trip(self) -> None:
        for family, mode, window in (
            ("lp", "stale", None),
            ("hedge", "fresh", None),
            ("hedge", "observed", 2.5),
        ):
            parsed = parse_policy_name(policy_name(family, mode, window))
            assert parsed == {"family": family, "belief_mode": mode, "window_min": window}
        with pytest.raises(ValueError, match="policy name"):
            parse_policy_name("ablation_lp_only_alpha75")


class TestTracking:
    def test_phase_metrics_partition_the_trace(self) -> None:
        scenario = harness.make_scenario(
            spike_artifact_label(3.0, 2.0),
            requests=_REQUESTS,
            config=_CONFIG,
        )
        presets = make_presets(output_predictor=None)
        run = harness.run_stale_telemetry_policy(
            scenario,
            _REQUESTS,
            "routewise_lp_stale",
            presets=presets,
            seed=42,
            retain_records=False,
        )
        extra = run.extra_metrics
        n_by_phase = {phase: extra[f"{phase}_n_requests"] for phase in harness.PHASES}
        assert sum(n_by_phase.values()) == len(_REQUESTS)
        # Three 2-minute spikes at 10 s spacing -> 12 requests each.
        assert n_by_phase["spike"] == 36
        assert n_by_phase["post_spike"] == 36
        # Stale LP keeps routing everything to the spiked (baseline-fastest) provider.
        assert extra["spike_spiked_provider_final_share"] == 1.0
        assert extra["spike_hedge_rate"] == 0.0
        assert extra["spike_slo_violation_rate"] > extra["baseline_slo_violation_rate"]
        ts_counts = {key: value for key, value in extra.items() if key.startswith("ts|n|")}
        # Bins cover [-2, 8) minutes of every 10-minute cycle from 3 min onwards.
        assert len(ts_counts) == 10
        assert sum(ts_counts.values()) == sum(1 for r in _REQUESTS if r.timestamp >= 180.0)
        assert "profile_mean_fallback_rate" not in extra
        assert run.records == []


def test_cli_writes_summary_and_timeseries(tmp_path) -> None:
    output_dir = tmp_path / "stale_telemetry"
    assert (
        harness.main(
            [
                "--workload",
                "smoke",
                # Truncation keeps the loader on the direct JSONL path so the
                # test never writes a pickle cache next to the tracked fixture.
                "--max-requests",
                "120",
                "--magnitude",
                "3",
                "--spike-duration-min",
                "0.25",
                "--period-min",
                "1",
                "--warmup-min",
                "0.5",
                "--pre-onset-min",
                "0.25",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    rows = json.loads((output_dir / "summary.json").read_text())
    assert [row["policy"] for row in rows] == list(make_presets())
    by_policy = {row["policy"]: row for row in rows}
    for row in rows:
        assert row["scenario"] == "spike__mag=3__dur=0p25m"
        assert row["spiked_provider"] == "api_fast"
        assert row["spike_episode_count"] == 3
        assert row["spike_n_requests"] > 0
        assert not any(key.startswith("ts|") for key in row)
    stale_lp = by_policy["routewise_lp_stale"]
    assert stale_lp["spike_spiked_provider_final_share"] == 1.0
    assert stale_lp["spike_slo_violation_delta_vs_stale_lp_pp"] == 0.0
    assert stale_lp["spike_cost_multiplier_vs_stale_lp"] == 1.0
    assert by_policy["routewise_hedge_stale"]["spike_hedge_rate"] > 0.0
    assert by_policy["routewise_lp_fresh"]["spike_spiked_provider_final_share"] == 0.0
    assert by_policy["routewise_lp_observed15m"]["profile_mean_fallback_rate"] is not None

    with (output_dir / "summary.csv").open() as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 6
    assert csv_rows[0]["belief_mode"] == "stale"
    assert "spike_p99_ms" in csv_rows[0]

    with (output_dir / "spike_timeseries.csv").open() as handle:
        ts_rows = list(csv.DictReader(handle))
    # One-minute period with a 0.25-minute lead -> a single bin per policy.
    assert len(ts_rows) == 6
    assert ts_rows[0]["minutes_since_onset"] == "-0.25"
    assert float(ts_rows[0]["n"]) > 0

    from plots.ablations import plot_stale_telemetry

    assert plot_stale_telemetry.main(["--input-dir", str(output_dir)]) == 0
    figures = output_dir / "figures"
    assert (figures / "stale_telemetry_ts_slo__spike__mag=3__dur=0p25m.pdf").exists()
    assert (figures / "stale_telemetry_spike_slo_vs_magnitude__dur=0.25m.png").exists()
    assert (output_dir / "stale_telemetry_spike_summary.csv").exists()
