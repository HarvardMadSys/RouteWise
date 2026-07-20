"""Unit tests for the latency-profile window ablation building blocks."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.ablations.profile_window import harness
from experiments.ablations.profile_window.harness import (
    apply_square_wave,
    build_square_wave_schedule,
    parse_shift_artifact_label,
    shift_artifact_label,
)
from experiments.ablations.profile_window.presets import (
    make_profile_window_presets,
    parse_profile_window_policy_name,
    profile_window_policy_name,
)
from llm_routewise.core.latency_profile import RollingLatencyProfile
from llm_routewise.sim.world.distributions import LogNormal, ScaledDistribution
from llm_routewise.sim.world.providers import Provider


def _provider(name: str = "p") -> Provider:
    dist = LogNormal(mu=5.0, sigma=0.5)
    return Provider(
        name=name,
        cost_per_token=1e-6,
        ttft_dist=dist,
        tps_dist=LogNormal(mu=3.0, sigma=0.2),
    )


class TestScaledDistribution:
    def test_moments_scale_linearly(self) -> None:
        base = LogNormal(mu=5.0, sigma=0.5)
        scaled = ScaledDistribution(base=base, scale=3.0)
        assert scaled.mean() == pytest.approx(3.0 * base.mean())
        assert scaled.std() == pytest.approx(3.0 * base.std())
        assert scaled.p50() == pytest.approx(3.0 * base.p50())
        assert scaled.p99() == pytest.approx(3.0 * base.p99())
        assert scaled.quantile(0.9) == pytest.approx(3.0 * base.quantile(0.9))

    def test_cdf_matches_scaled_base(self) -> None:
        base = LogNormal(mu=5.0, sigma=0.5)
        scaled = ScaledDistribution(base=base, scale=3.0)
        for value in (50.0, 150.0, 600.0):
            assert scaled.cdf(3.0 * value) == pytest.approx(base.cdf(value))

    def test_samples_are_scaled(self) -> None:
        base = LogNormal(mu=5.0, sigma=0.5)
        scaled = ScaledDistribution(base=base, scale=3.0)
        draws = scaled.sample(np.random.default_rng(0), 2000)
        assert draws.mean() == pytest.approx(scaled.mean(), rel=0.1)

    @pytest.mark.parametrize("scale", [0.0, -1.0, float("inf")])
    def test_rejects_invalid_scale(self, scale: float) -> None:
        with pytest.raises(ValueError, match="scale"):
            ScaledDistribution(base=LogNormal(mu=5.0, sigma=0.5), scale=scale)


class TestShiftSchedule:
    def test_schedule_boundaries(self) -> None:
        provider = _provider()
        base = provider.ttft_dist
        degraded = ScaledDistribution(base=base, scale=3.0)
        provider.ttft_shift_schedule = ((100.0, degraded), (200.0, base), (300.0, degraded))
        assert provider._active_ttft_dist(0.0) is base
        assert provider._active_ttft_dist(99.9) is base
        assert provider._active_ttft_dist(100.0) is degraded
        assert provider._active_ttft_dist(250.0) is base
        assert provider._active_ttft_dist(1e12) is degraded

    def test_schedule_conflicts_with_legacy_shift(self) -> None:
        dist = LogNormal(mu=5.0, sigma=0.5)
        with pytest.raises(ValueError, match="not both"):
            Provider(
                name="x",
                cost_per_token=0.0,
                ttft_dist=dist,
                tps_dist=dist,
                shift_time=1.0,
                ttft_dist_after=dist,
                ttft_shift_schedule=((1.0, dist),),
            )

    def test_schedule_must_be_sorted(self) -> None:
        dist = LogNormal(mu=5.0, sigma=0.5)
        with pytest.raises(ValueError, match="sorted"):
            Provider(
                name="x",
                cost_per_token=0.0,
                ttft_dist=dist,
                tps_dist=dist,
                ttft_shift_schedule=((2.0, dist), (1.0, dist)),
            )


class TestSquareWave:
    def test_alternates_and_starts_degraded(self) -> None:
        base = LogNormal(mu=5.0, sigma=0.5)
        schedule = build_square_wave_schedule(
            base,
            anchor_sec=1000.0,
            end_sec=1000.0 + 4 * 600.0,
            period_sec=600.0,
            offset_sec=0.0,
            magnitude=3.0,
        )
        times = [entry[0] for entry in schedule]
        assert times == [1000.0, 1600.0, 2200.0, 2800.0, 3400.0]
        assert isinstance(schedule[0][1], ScaledDistribution)
        assert schedule[1][1] is base
        assert isinstance(schedule[2][1], ScaledDistribution)

    def test_apply_square_wave_staggers_offsets(self) -> None:
        providers = [_provider(f"p{i}") for i in range(4)]
        apply_square_wave(
            providers,
            anchor_sec=0.0,
            end_sec=4000.0,
            period_sec=600.0,
            magnitude=3.0,
        )
        onsets = [provider.ttft_shift_schedule[0][0] for provider in providers]
        assert onsets == [0.0, 300.0, 600.0, 900.0]
        # At any instant roughly half the providers are degraded.
        mid = 700.0
        degraded = sum(
            isinstance(provider._active_ttft_dist(mid), ScaledDistribution)
            for provider in providers
        )
        assert 1 <= degraded <= 3

    def test_mean_toggles_by_magnitude(self) -> None:
        provider = _provider()
        base_mean = provider.true_mean_ms(0.0)
        apply_square_wave(
            [provider],
            anchor_sec=0.0,
            end_sec=2000.0,
            period_sec=600.0,
            magnitude=3.0,
        )
        assert provider.true_mean_ms(10.0) == pytest.approx(3.0 * base_mean)
        assert provider.true_mean_ms(700.0) == pytest.approx(base_mean)


class TestLabels:
    def test_artifact_label_round_trip(self) -> None:
        for period, magnitude in ((0.0, 3.0), (10.0, 3.0), (2.5, 1.5), (60.0, 5.0)):
            label = shift_artifact_label(period, magnitude)
            assert parse_shift_artifact_label(label) == (period, magnitude)

    def test_static_label(self) -> None:
        assert shift_artifact_label(0.0, 3.0) == "rw8_shift__period=static__mag=3"

    def test_negative_periods_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="period_min"):
            shift_artifact_label(-5.0, 3.0)
        with pytest.raises(ValueError, match="period sweep values"):
            harness.make_scenarios(period_minutes=(-5.0,), requests=[])

    def test_policy_name_round_trip(self) -> None:
        for family, alpha, window in (
            ("lp", 0.0, 15.0),
            ("hedge", 0.25, 1.5),
            ("lp", 0.5, None),
        ):
            name = profile_window_policy_name(family, alpha, window)
            parsed = parse_profile_window_policy_name(name)
            assert parsed == {"family": family, "alpha": alpha, "window_min": window}


class TestPresets:
    def test_observed_presets_set_window_and_mode(self) -> None:
        presets = make_profile_window_presets(
            window_minutes=(2.0, 15.0),
            alpha_values=(0.0,),
        )
        observed = presets["routewise_lp_alpha0_w2m"]["params"]
        assert observed["latency_profile_mode"] == "observed"
        assert observed["profile_window_sec"] == 120.0
        assert observed["explorer"] is True
        hedged = presets["routewise_hedge_alpha0_w15m"]["params"]
        assert hedged["hedging"] == "probability_target"
        assert hedged["profile_window_sec"] == 900.0

    def test_oracle_presets_use_configured_mode(self) -> None:
        presets = make_profile_window_presets(
            window_minutes=(15.0,),
            alpha_values=(0.0,),
        )
        oracle = presets["routewise_lp_alpha0_oracle"]["params"]
        assert oracle["latency_profile_mode"] == "configured"
        assert "profile_window_sec" not in oracle

    def test_no_oracle_option(self) -> None:
        presets = make_profile_window_presets(
            window_minutes=(15.0,),
            alpha_values=(0.0,),
            include_oracle=False,
        )
        assert not any(name.endswith("_oracle") for name in presets)


class TestHarnessWorkers:
    def test_parallel_cell_forwards_slo_ms(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        monkeypatch.setattr(
            harness.common,
            "load_workload",
            lambda **kwargs: ["request"],
        )

        def fake_make_scenario(
            name: str,
            *,
            requests: list[object],
            slo_ms: float = harness.end_to_end.DEFAULT_SLO_MS,
        ) -> object:
            captured["name"] = name
            captured["requests"] = requests
            captured["slo_ms"] = slo_ms
            return object()

        monkeypatch.setattr(harness, "make_scenario", fake_make_scenario)
        monkeypatch.setattr(
            harness,
            "run_profile_window_policy",
            lambda *args, **kwargs: "run",
        )

        result = harness.run_profile_window_cell(
            harness.common.SectionCell("scenario", "policy", 7),
            {},
            "burstgpt",
            None,
            5,
            False,
            slo_ms=123.0,
        )

        assert captured == {
            "name": "scenario",
            "requests": ["request"],
            "slo_ms": 123.0,
        }
        assert result.run == "run"


class TestRollingProfileBackwardsQueries:
    """The hedging tick loop queries ahead of trace time; subsequent route
    queries land behind the advanced clock and must stay exact."""

    def test_backwards_queries_match_brute_force(self) -> None:
        rng = np.random.default_rng(3)
        window = 50.0
        profile = RollingLatencyProfile(window_sec=window)
        history: list[tuple[float, float]] = []
        t = 0.0
        for _ in range(1500):
            t += float(rng.exponential(1.0))
            ts = t + float(rng.uniform(0.0, 2.0))
            value = float(rng.lognormal(4.0, 0.7))
            profile.add_sample(ts, value)
            history.append((ts, value))
            for query in (t + float(rng.uniform(0.0, 15.0)), t - float(rng.uniform(0.0, 15.0))):
                if query < 0.0:
                    continue
                expected = [v for s, v in history if query - window <= s <= query]
                threshold = float(rng.uniform(20.0, 150.0))
                got = profile.cdf(threshold, query)
                if not expected:
                    assert got is None
                    continue
                want = sum(1 for v in expected if v <= threshold) / len(expected)
                assert got == pytest.approx(want, abs=1e-12)
                assert profile.mean(query) == pytest.approx(float(np.mean(expected)), rel=1e-9)

    def test_deep_backwards_query_respects_retention_horizon(self) -> None:
        profile = RollingLatencyProfile(window_sec=10.0)
        profile.add_sample(5.0, 100.0)
        profile.add_sample(100.0, 900.0)
        assert profile.mean(200.0) is None
        # 190s behind the clock, far beyond the retained horizon: the old
        # sample has been pruned, so the defensive path cannot answer.
        assert profile.mean(10.0) is None
        assert profile.cdf(150.0, 10.0) is None
