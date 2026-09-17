"""Stale-telemetry robustness experiment: hedging under sudden latency spikes.

Question: when one provider suffers a sudden load spike that the latency
profiler has not yet caught up with, can probability-target hedging still
contain the SLO damage by sending a backup request to a provider that is not
experiencing the same load spike?

Environment: the §2.2 same-cost real-world hedging scenario (RW3 by default)
plus a recurring spike train on the provider the router prefers at baseline
(lowest baseline mean TTFT). Every ``period`` minutes that provider's TTFT
distribution becomes ``ScaledDistribution(base, magnitude)`` for ``duration``
minutes and is then restored, via ``Provider.ttft_shift_schedule``. The other
providers stay stationary, so a backup sent to them is never spiked.

Policies: LP-only and LP+hedging RouteWise under three telemetry regimes.
``stale`` freezes the beliefs at the pre-spike baseline (the profiler never
updates), ``observed`` is the production rolling W-minute profile (it lags the
spike by up to W), and ``fresh`` is the configured-mode oracle that tracks the
spike instantly. ``stale`` and ``fresh`` bracket profiler staleness; hedging's
mitigation before the profiler updates is read off the ``stale`` pair.

Outputs (default ``outputs/ablations/stale_telemetry/``):
- ``summary.json`` / ``summary.csv``: whole-run rows plus per-phase metrics
  (``baseline_*``, ``spike_*``, ``post_spike_*``) and spike-phase deltas
  against the no-mitigation ``routewise_lp_stale`` row.
- ``spike_timeseries.csv``: 1-minute bins aligned on spike onset, pooled over
  every spike episode of the replay.
- ``metadata.json``, ``ttft_histograms*.json``: as in other sections.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from experiments.ablations.stale_telemetry.policy import StaleBeliefRouteWisePolicy
from experiments.ablations.stale_telemetry.presets import (
    DEFAULT_PROFILE_WINDOW_MIN,
    ROUTEWISE_ALPHA,
    format_number,
    make_presets,
    parse_number,
    parse_policy_name,
    policy_name,
)
from experiments.simulation import common, hedging
from llm_routewise.sim.engine.simulator import Simulator
from llm_routewise.sim.policies.routewise import RouteWisePolicy
from llm_routewise.sim.world.distributions import ScaledDistribution

if TYPE_CHECKING:
    from llm_routewise.metrics import Run
    from llm_routewise.schemas import HedgeDispatch, Request, RoutingDecision, RoutingOutcome
    from llm_routewise.sim.engine.state import SimulationState
    from llm_routewise.sim.world.providers import Provider
    from llm_routewise.sim.world.scenarios import ScenarioConfig

SECTION_NAME = "stale-telemetry"
PUBLIC_SCENARIO_TAG = "stale_telemetry"
DEFAULT_WORKLOAD = common.DEFAULT_WORKLOAD
DEFAULT_OUTPUT_DIR = common.OUTPUT_DIR.parent / "ablations" / "stale_telemetry"

POOL_SCENARIOS: dict[str, str] = {
    "rw3": hedging.REAL_WORLD_RW3_SCENARIO_NAME,
    "rw8": hedging.REAL_WORLD_RW8_SCENARIO_NAME,
}
DEFAULT_POOL = "rw3"
DEFAULT_MAGNITUDES: tuple[float, ...] = (2.0, 3.0, 5.0)
DEFAULT_SPIKE_DURATION_MINUTES: tuple[float, ...] = (10.0,)
DEFAULT_PERIOD_MIN = 60.0
DEFAULT_WARMUP_MIN = 30.0
DEFAULT_PRE_ONSET_MIN = 10.0
TIMESERIES_BIN_SEC = 60.0
PHASES: tuple[str, ...] = ("baseline", "spike", "post_spike")
NO_MITIGATION_POLICY = policy_name("lp", "stale")

_LABEL_PREFIX = "spike__mag="
_DURATION_MARKER = "__dur="
_TIMESERIES_KEY_PREFIX = "ts|"
_POLICY_CLASSES: dict[str, type[RouteWisePolicy]] = {
    "RouteWisePolicy": RouteWisePolicy,
    "StaleBeliefRouteWisePolicy": StaleBeliefRouteWisePolicy,
}


@dataclass(frozen=True)
class SpikeTrainConfig:
    """Run-wide spike-train settings shared by every scenario of one run."""

    pool: str = DEFAULT_POOL
    period_min: float = DEFAULT_PERIOD_MIN
    warmup_min: float = DEFAULT_WARMUP_MIN
    pre_onset_min: float = DEFAULT_PRE_ONSET_MIN
    spiked_provider: str | None = None

    def __post_init__(self) -> None:
        if self.pool not in POOL_SCENARIOS:
            known = ", ".join(POOL_SCENARIOS)
            raise ValueError(f"unknown provider pool {self.pool!r}; known: {known}")
        if self.period_min <= 0.0:
            raise ValueError(f"period_min must be > 0, got {self.period_min}")
        if self.warmup_min < 0.0:
            raise ValueError(f"warmup_min must be >= 0, got {self.warmup_min}")
        if not 0.0 <= self.pre_onset_min < self.period_min:
            raise ValueError(f"pre_onset_min must be in [0, period_min), got {self.pre_onset_min}")

    def validate_spike(self, magnitude: float, duration_min: float) -> None:
        """Check one scenario's spike coordinates against the run settings."""
        if magnitude <= 1.0:
            raise ValueError(f"spike magnitude must be > 1 to degrade latency, got {magnitude}")
        if not 0.0 < duration_min < self.period_min:
            raise ValueError(
                f"spike duration must be in (0, period_min={self.period_min}), got {duration_min}"
            )
        if self.pre_onset_min > self.period_min - duration_min:
            raise ValueError(
                "pre_onset_min must leave the spike inside one cycle: "
                f"pre_onset_min={self.pre_onset_min}, duration_min={duration_min}, "
                f"period_min={self.period_min}"
            )


def spike_artifact_label(magnitude: float, duration_min: float) -> str:
    """Return the stable scenario label for one (magnitude, duration) cell."""
    return (
        f"{_LABEL_PREFIX}{format_number(magnitude)}{_DURATION_MARKER}{format_number(duration_min)}m"
    )


def parse_spike_artifact_label(name: str) -> tuple[float, float]:
    """Parse ``(magnitude, duration_min)`` back from a scenario label."""
    if not name.startswith(_LABEL_PREFIX) or _DURATION_MARKER not in name:
        raise ValueError(f"unknown stale-telemetry scenario label {name!r}")
    magnitude_token, duration_token = name.removeprefix(_LABEL_PREFIX).split(_DURATION_MARKER, 1)
    if not duration_token.endswith("m"):
        raise ValueError(f"unknown stale-telemetry scenario label {name!r}")
    return parse_number(magnitude_token), parse_number(duration_token.removesuffix("m"))


def select_spiked_provider(providers: list[Provider], name: str | None = None) -> Provider:
    """Return the provider to spike: ``name`` or the lowest baseline mean TTFT."""
    if name is not None:
        for provider in providers:
            if provider.name == name:
                return provider
        known = ", ".join(provider.name for provider in providers)
        raise ValueError(f"spiked provider {name!r} not in scenario; known: {known}")
    return min(providers, key=lambda provider: (provider.ttft_dist.mean(), provider.name))


def spike_onsets(
    *,
    anchor_sec: float,
    end_sec: float,
    warmup_sec: float,
    period_sec: float,
) -> tuple[float, ...]:
    """Return spike onset times: ``anchor + warmup + k * period`` up to ``end_sec``."""
    first = float(anchor_sec) + float(warmup_sec)
    if end_sec < first:
        return ()
    count = math.floor((float(end_sec) - first) / float(period_sec)) + 1
    return tuple(first + index * float(period_sec) for index in range(count))


def build_spike_schedule(
    base_dist: Any,
    *,
    onsets: tuple[float, ...],
    duration_sec: float,
    magnitude: float,
) -> tuple[tuple[float, Any], ...]:
    """Return degraded/restore shift entries for one provider."""
    degraded = ScaledDistribution(base=base_dist, scale=magnitude)
    entries: list[tuple[float, Any]] = []
    for onset in onsets:
        entries.append((onset, degraded))
        entries.append((onset + float(duration_sec), base_dist))
    return tuple(entries)


def make_scenario(
    name: str,
    *,
    requests: list[Request],
    config: SpikeTrainConfig,
) -> ScenarioConfig:
    """Build one spike scenario from its label, anchored on the replayed trace."""
    magnitude, duration_min = parse_spike_artifact_label(name)
    config.validate_spike(magnitude, duration_min)
    if not requests:
        raise ValueError("spike scenarios need a non-empty workload to anchor the spike train")
    scenario = hedging.make_scenario(POOL_SCENARIOS[config.pool])
    spiked = select_spiked_provider(scenario.providers, config.spiked_provider)
    onsets = spike_onsets(
        anchor_sec=float(requests[0].timestamp),
        end_sec=float(requests[-1].timestamp),
        warmup_sec=config.warmup_min * 60.0,
        period_sec=config.period_min * 60.0,
    )
    spiked.ttft_shift_schedule = build_spike_schedule(
        spiked.ttft_dist,
        onsets=onsets,
        duration_sec=duration_min * 60.0,
        magnitude=magnitude,
    )
    scenario.name = name
    scenario.description = (
        f"Stale telemetry: {config.pool.upper()} same-cost pool, "
        f"{spiked.name} TTFT x{magnitude:g} for {duration_min:g} min every "
        f"{config.period_min:g} min ({len(onsets)} episodes), "
        f"SLO={scenario.primary_slo_ms:.0f} ms."
    )
    scenario.metadata = {
        **scenario.metadata,
        "public_scenario": PUBLIC_SCENARIO_TAG,
        "artifact_label": name,
        "source_hedging_scenario": POOL_SCENARIOS[config.pool],
        "real_world_pool": config.pool,
        "spike_magnitude": magnitude,
        "spike_duration_min": duration_min,
        "spike_period_min": config.period_min,
        "spike_warmup_min": config.warmup_min,
        "spike_pre_onset_min": config.pre_onset_min,
        "spike_first_onset_sec": onsets[0] if onsets else None,
        "spike_episode_count": len(onsets),
        "spiked_provider": spiked.name,
        "spiked_provider_baseline_mean_ms": spiked.ttft_dist.mean(),
        "spiked_provider_spike_mean_ms": spiked.ttft_dist.mean() * magnitude,
        "spike_shape": "scaled_distribution",
    }
    return scenario


def make_scenarios(
    *,
    magnitudes: tuple[float, ...] = DEFAULT_MAGNITUDES,
    duration_minutes: tuple[float, ...] = DEFAULT_SPIKE_DURATION_MINUTES,
    requests: list[Request],
    config: SpikeTrainConfig,
) -> dict[str, ScenarioConfig]:
    """Build the magnitude x duration scenario grid keyed by label."""
    if len(set(magnitudes)) != len(magnitudes):
        raise ValueError(f"magnitudes must be unique, got {magnitudes}")
    if len(set(duration_minutes)) != len(duration_minutes):
        raise ValueError(f"spike durations must be unique, got {duration_minutes}")
    scenarios: dict[str, ScenarioConfig] = {}
    for duration_min in duration_minutes:
        for magnitude in magnitudes:
            label = spike_artifact_label(magnitude, duration_min)
            scenarios[label] = make_scenario(label, requests=requests, config=config)
    return scenarios


def classify_phase(
    timestamp: float,
    *,
    first_onset_sec: float,
    period_sec: float,
    duration_sec: float,
) -> str:
    """Return ``spike``, ``post_spike`` (the ``duration`` after it), or ``baseline``."""
    if timestamp < first_onset_sec:
        return "baseline"
    since_onset = (float(timestamp) - float(first_onset_sec)) % float(period_sec)
    if since_onset < duration_sec:
        return "spike"
    if since_onset < 2.0 * duration_sec:
        return "post_spike"
    return "baseline"


def timeseries_bin(
    timestamp: float,
    *,
    first_onset_sec: float,
    period_sec: float,
    pre_onset_sec: float,
) -> int | None:
    """Return the onset-aligned bin index, or ``None`` before the first cycle.

    Bin ``i`` covers ``[-pre_onset + i * BIN, -pre_onset + (i + 1) * BIN)``
    seconds relative to the nearest preceding onset, wrapping every period.
    """
    if timestamp < first_onset_sec - pre_onset_sec:
        return None
    relative = (float(timestamp) - float(first_onset_sec) + float(pre_onset_sec)) % float(
        period_sec
    )
    return int(relative // TIMESERIES_BIN_SEC)


def timeseries_bin_start_min(index: int, *, pre_onset_min: float) -> float:
    """Return the start of bin ``index`` in minutes since spike onset."""
    return -float(pre_onset_min) + index * TIMESERIES_BIN_SEC / 60.0


@dataclass
class _PhaseStats:
    ttft_ms: list[float] = field(default_factory=list)
    slo_violations: int = 0
    hedges: int = 0
    cost_usd: float = 0.0
    spiked_primary: int = 0
    spiked_final: int = 0

    def metrics(self, prefix: str) -> dict[str, float]:
        n = len(self.ttft_ms)
        values = np.asarray(self.ttft_ms, dtype=float)
        nan = float("nan")

        def rate(count: int) -> float:
            return count / n if n else nan

        return {
            f"{prefix}_n_requests": float(n),
            f"{prefix}_mean_ttft_ms": float(values.mean()) if n else nan,
            f"{prefix}_p50_ms": float(np.percentile(values, 50)) if n else nan,
            f"{prefix}_p90_ms": float(np.percentile(values, 90)) if n else nan,
            f"{prefix}_p99_ms": float(np.percentile(values, 99)) if n else nan,
            f"{prefix}_slo_violation_rate": rate(self.slo_violations),
            f"{prefix}_hedge_rate": rate(self.hedges),
            f"{prefix}_mean_cost_usd": self.cost_usd / n if n else nan,
            f"{prefix}_spiked_provider_primary_share": rate(self.spiked_primary),
            f"{prefix}_spiked_provider_final_share": rate(self.spiked_final),
        }


@dataclass
class _BinStats:
    n: int = 0
    slo_violations: int = 0
    ttft_sum_ms: float = 0.0
    hedges: int = 0
    spiked_final: int = 0

    def metrics(self) -> dict[str, float]:
        n = self.n
        nan = float("nan")
        return {
            "n": float(n),
            "slo_violation_rate": self.slo_violations / n if n else nan,
            "mean_ttft_ms": self.ttft_sum_ms / n if n else nan,
            "hedge_rate": self.hedges / n if n else nan,
            "spiked_provider_final_share": self.spiked_final / n if n else nan,
        }


class _SpikeTrackingPolicy:
    """Policy wrapper that accumulates per-phase and onset-aligned metrics.

    Streams over ``observe`` outcomes so the run never has to retain
    per-request records; ``slo_violated`` matches the simulator record
    definition (rejected or user-visible TTFT above the request SLO).
    """

    def __init__(
        self,
        inner: RouteWisePolicy,
        *,
        scenario: ScenarioConfig,
    ) -> None:
        meta = scenario.metadata
        self.inner = inner
        self.slo_ms = float(scenario.primary_slo_ms)
        self.spiked_provider = str(meta["spiked_provider"])
        self.first_onset_sec = meta["spike_first_onset_sec"]
        self.period_sec = float(meta["spike_period_min"]) * 60.0
        self.duration_sec = float(meta["spike_duration_min"]) * 60.0
        self.pre_onset_min = float(meta["spike_pre_onset_min"])
        self.phases: dict[str, _PhaseStats] = {phase: _PhaseStats() for phase in PHASES}
        self.bins: dict[int, _BinStats] = {}

    def route(self, request: Request, state: SimulationState) -> RoutingDecision:
        return self.inner.route(request, state)

    def tick(
        self,
        request: Request,
        decision: RoutingDecision,
        elapsed: float,
        state: SimulationState,
    ) -> HedgeDispatch | None:
        return self.inner.tick(request, decision, elapsed, state)

    def observe(
        self,
        request: Request,
        decision: RoutingDecision,
        outcome: RoutingOutcome,
    ) -> None:
        self.inner.observe(request, decision, outcome)
        timestamp = float(request.timestamp)
        ttft_ms = float(outcome.ttft_ms)
        slo_ms = float(request.slo_ms or self.slo_ms)
        violated = bool(outcome.rejected or ttft_ms > slo_ms)
        hedged = bool(outcome.hedge_triggered)
        spiked_final = outcome.final_provider == self.spiked_provider

        if self.first_onset_sec is None:
            phase = "baseline"
            bin_index = None
        else:
            phase = classify_phase(
                timestamp,
                first_onset_sec=self.first_onset_sec,
                period_sec=self.period_sec,
                duration_sec=self.duration_sec,
            )
            bin_index = timeseries_bin(
                timestamp,
                first_onset_sec=self.first_onset_sec,
                period_sec=self.period_sec,
                pre_onset_sec=self.pre_onset_min * 60.0,
            )
        stats = self.phases[phase]
        stats.ttft_ms.append(ttft_ms)
        stats.slo_violations += violated
        stats.hedges += hedged
        stats.cost_usd += float(outcome.cost_usd)
        stats.spiked_primary += outcome.primary_provider == self.spiked_provider
        stats.spiked_final += spiked_final
        if bin_index is not None:
            bin_stats = self.bins.setdefault(bin_index, _BinStats())
            bin_stats.n += 1
            bin_stats.slo_violations += violated
            bin_stats.ttft_sum_ms += ttft_ms
            bin_stats.hedges += hedged
            bin_stats.spiked_final += spiked_final

    def extra_metrics(self) -> dict[str, float]:
        """Flatten phase and time-series metrics into ``Run.extra_metrics`` keys."""
        metrics: dict[str, float] = {}
        for phase in PHASES:
            metrics.update(self.phases[phase].metrics(phase))
        for index in sorted(self.bins):
            minute = timeseries_bin_start_min(index, pre_onset_min=self.pre_onset_min)
            for key, value in self.bins[index].metrics().items():
                metrics[f"{_TIMESERIES_KEY_PREFIX}{key}|{format_number(minute)}"] = value
        return metrics


def build_policy(
    policy_name: str,
    *,
    presets: dict[str, dict[str, Any]],
    scenario: ScenarioConfig,
    requests: list[Request],
    seed: int,
) -> RouteWisePolicy:
    """Instantiate one preset with sentinels (SLO, envelope, predictor) resolved."""
    materialized = common.materialize_policy_presets(
        presets,
        policy_name=policy_name,
        scenario=scenario,
        requests=requests,
    )
    try:
        preset = materialized[policy_name]
    except KeyError as exc:
        known = ", ".join(sorted(materialized))
        raise ValueError(f"unknown stale-telemetry policy {policy_name!r}; known: {known}") from exc
    try:
        cls = _POLICY_CLASSES[preset["policy"]]
    except KeyError as exc:
        raise ValueError(f"unsupported stale-telemetry preset {policy_name!r}: {preset!r}") from exc
    params = dict(preset.get("params", {}))
    params.setdefault("slo_ms", float(scenario.primary_slo_ms))
    params.setdefault("seed", seed)
    return cls(**params)


def run_stale_telemetry_policy(
    scenario: ScenarioConfig,
    requests: list[Request],
    policy_name: str,
    *,
    presets: dict[str, dict[str, Any]],
    seed: int,
    retain_records: bool = True,
) -> Run:
    """Run one preset and attach phase / time-series metrics to the Run."""
    inner = build_policy(
        policy_name,
        presets=presets,
        scenario=scenario,
        requests=requests,
        seed=seed,
    )
    tracker = _SpikeTrackingPolicy(inner, scenario=scenario)
    simulator = Simulator(scenario=scenario, seed=seed, retain_records=retain_records)
    run = simulator.run(requests, tracker, policy_name=policy_name)
    extra = tracker.extra_metrics()
    if inner.router.beliefs.mode == "observed":
        extra["profile_mean_fallback_rate"] = inner.router.beliefs.mean_fallback_rate()
        extra["profile_cdf_fallback_rate"] = inner.router.beliefs.cdf_fallback_rate()
    run.extra_metrics = extra
    return run


def run_stale_telemetry_cell(
    cell: common.SectionCell,
    presets: dict[str, dict[str, Any]],
    workload_dataset: str,
    duration_sec: float | None,
    max_requests: int | None,
    retain_records: bool,
    *,
    config: SpikeTrainConfig,
) -> common.SectionCellResult:
    """Run one scenario-policy-seed cell in a worker process."""
    requests = common.load_workload(
        dataset=workload_dataset,
        duration_sec=duration_sec,
        max_requests=max_requests,
    )
    scenario = make_scenario(cell.scenario_name, requests=requests, config=config)
    run = run_stale_telemetry_policy(
        scenario,
        requests,
        cell.policy,
        presets=presets,
        seed=cell.seed,
        retain_records=retain_records,
    )
    return common.SectionCellResult(
        scenario_name=cell.scenario_name,
        policy=cell.policy,
        seed=cell.seed,
        run=run,
    )


def enrich_rows(
    rows: list[dict[str, Any]],
    scenarios: dict[str, ScenarioConfig],
    presets: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Add spike / policy coordinates and deltas; split out the time series.

    Returns ``(summary_rows, timeseries_rows)``. Time-series bins arrive as
    flat ``ts|<metric>|<minute>`` keys (averaged over seeds by
    ``run_section``; seeds share the trace, so bin counts are identical and
    averaging rates is exact) and are moved into long-format rows.
    """
    summary: list[dict[str, Any]] = []
    timeseries: list[dict[str, Any]] = []
    for row in rows:
        meta = dict(scenarios[row["scenario"]].metadata or {})
        params = dict(presets[row["policy"]].get("params", {}))
        parsed = parse_policy_name(row["policy"])
        coordinates = {
            "real_world_pool": meta.get("real_world_pool"),
            "source_hedging_scenario": meta.get("source_hedging_scenario"),
            "slo_ms": meta.get("slo_ms"),
            "spike_magnitude": meta.get("spike_magnitude"),
            "spike_duration_min": meta.get("spike_duration_min"),
            "spike_period_min": meta.get("spike_period_min"),
            "spike_warmup_min": meta.get("spike_warmup_min"),
            "spike_pre_onset_min": meta.get("spike_pre_onset_min"),
            "spike_episode_count": meta.get("spike_episode_count"),
            "spiked_provider": meta.get("spiked_provider"),
            "spiked_provider_baseline_mean_ms": meta.get("spiked_provider_baseline_mean_ms"),
            "spiked_provider_spike_mean_ms": meta.get("spiked_provider_spike_mean_ms"),
            "policy_family": parsed["family"],
            "belief_mode": parsed["belief_mode"],
            "profile_window_min": parsed["window_min"],
            "hedging_enabled": bool(params.get("hedging")),
            "explorer_enabled": bool(params.get("explorer")),
            "latency_profile_mode": params.get("latency_profile_mode"),
            "routewise_alpha": float(params.get("alpha", ROUTEWISE_ALPHA)),
        }
        merged: dict[str, Any] = {}
        bins: dict[float, dict[str, float]] = {}
        for key, value in row.items():
            if not key.startswith(_TIMESERIES_KEY_PREFIX):
                merged[key] = value
                continue
            _, metric, minute_token = key.split("|", 2)
            bins.setdefault(parse_number(minute_token), {})[metric] = value
        merged.update(coordinates)
        summary.append(merged)
        for minute in sorted(bins):
            timeseries.append(
                {
                    "scenario": row["scenario"],
                    "policy": row["policy"],
                    "policy_family": parsed["family"],
                    "belief_mode": parsed["belief_mode"],
                    "profile_window_min": parsed["window_min"],
                    "hedging_enabled": coordinates["hedging_enabled"],
                    "spike_magnitude": coordinates["spike_magnitude"],
                    "spike_duration_min": coordinates["spike_duration_min"],
                    "minutes_since_onset": minute,
                    **bins[minute],
                }
            )

    baselines = {row["scenario"]: row for row in summary if row["policy"] == NO_MITIGATION_POLICY}
    for row in summary:
        _add_no_mitigation_deltas(row, baselines.get(row["scenario"]))
    return summary, timeseries


def _add_no_mitigation_deltas(row: dict[str, Any], baseline: dict[str, Any] | None) -> None:
    keys = (
        "spike_slo_violation_delta_vs_stale_lp_pp",
        "spike_p99_reduction_vs_stale_lp_pct",
        "spike_mean_ttft_reduction_vs_stale_lp_pct",
        "spike_cost_multiplier_vs_stale_lp",
    )
    if baseline is None:
        for key in keys:
            row[key] = None
        return
    row["spike_slo_violation_delta_vs_stale_lp_pp"] = (
        row["spike_slo_violation_rate"] - baseline["spike_slo_violation_rate"]
    ) * 100.0
    row["spike_p99_reduction_vs_stale_lp_pct"] = _relative_reduction_pct(
        before=baseline["spike_p99_ms"],
        after=row["spike_p99_ms"],
    )
    row["spike_mean_ttft_reduction_vs_stale_lp_pct"] = _relative_reduction_pct(
        before=baseline["spike_mean_ttft_ms"],
        after=row["spike_mean_ttft_ms"],
    )
    base_cost = baseline["spike_mean_cost_usd"]
    row["spike_cost_multiplier_vs_stale_lp"] = (
        row["spike_mean_cost_usd"] / base_cost if base_cost and base_cost > 0.0 else None
    )


def _relative_reduction_pct(*, before: float, after: float) -> float | None:
    if not before or not math.isfinite(before) or before <= 0.0:
        return None
    return (before - after) / before * 100.0


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows with union fieldnames; nested maps JSON-encoded, NaN blank."""
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in fieldnames})


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def main(argv: list[str] | None = None) -> int:
    """Run the stale-telemetry spike grid."""
    parser = argparse.ArgumentParser(
        prog="stale-telemetry",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--magnitude",
        type=float,
        action="append",
        dest="magnitudes",
        help=(
            "Spike TTFT scale factor on the spiked provider. Repeat to sweep. "
            f"Defaults to {DEFAULT_MAGNITUDES}."
        ),
    )
    parser.add_argument(
        "--spike-duration-min",
        type=float,
        action="append",
        dest="duration_minutes",
        help=(
            "Spike duration in minutes. Repeat to sweep. "
            f"Defaults to {DEFAULT_SPIKE_DURATION_MINUTES}."
        ),
    )
    parser.add_argument(
        "--period-min",
        type=float,
        default=DEFAULT_PERIOD_MIN,
        help=f"Spike repetition period in minutes. Defaults to {DEFAULT_PERIOD_MIN:g}.",
    )
    parser.add_argument(
        "--warmup-min",
        type=float,
        default=DEFAULT_WARMUP_MIN,
        help=(
            "Stationary minutes before the first spike so rolling profiles fill. "
            f"Defaults to {DEFAULT_WARMUP_MIN:g}."
        ),
    )
    parser.add_argument(
        "--pre-onset-min",
        type=float,
        default=DEFAULT_PRE_ONSET_MIN,
        help=(
            "Minutes before onset covered by the onset-aligned time series. "
            f"Defaults to {DEFAULT_PRE_ONSET_MIN:g}."
        ),
    )
    parser.add_argument(
        "--pool",
        default=DEFAULT_POOL,
        choices=list(POOL_SCENARIOS),
        help=f"§2.2 same-cost real-world provider pool. Defaults to {DEFAULT_POOL}.",
    )
    parser.add_argument(
        "--spiked-provider",
        help="Provider to spike. Defaults to the lowest baseline mean TTFT in the pool.",
    )
    parser.add_argument(
        "--window-min",
        type=float,
        default=DEFAULT_PROFILE_WINDOW_MIN,
        help=(
            "Rolling-profile window (minutes) for the observed regime. "
            f"Defaults to {DEFAULT_PROFILE_WINDOW_MIN:g}."
        ),
    )
    parser.add_argument(
        "--policy",
        action="append",
        help="Policy to run. Repeat to run multiple. Defaults to all six presets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        help=f"Seed to run. Repeat to run multiple. Defaults to {common.DEFAULT_SEEDS}.",
    )
    parser.add_argument(
        "--workload",
        default=DEFAULT_WORKLOAD,
        choices=common.WORKLOAD_CHOICES,
        help=f"Trace workload to replay. Defaults to {DEFAULT_WORKLOAD}.",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        help="Optional trace truncation (seconds of trace) for shorter runs.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        help="Optional request-count truncation for smoke runs.",
    )
    parser.add_argument(
        "--predictor",
        default=common.DEFAULT_OUTPUT_PREDICTOR,
        help=f"Output-length predictor. Defaults to {common.DEFAULT_OUTPUT_PREDICTOR}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for metadata.json, summary.{json,csv}, spike_timeseries.csv.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel scenario-policy-seed cells to run. Defaults to 1.",
    )

    args = parser.parse_args(argv)
    magnitudes = tuple(args.magnitudes or DEFAULT_MAGNITUDES)
    duration_minutes = tuple(args.duration_minutes or DEFAULT_SPIKE_DURATION_MINUTES)
    if args.window_min <= 0.0:
        raise SystemExit(f"--window-min must be > 0, got {args.window_min}")
    try:
        config = SpikeTrainConfig(
            pool=args.pool,
            period_min=args.period_min,
            warmup_min=args.warmup_min,
            pre_onset_min=args.pre_onset_min,
            spiked_provider=args.spiked_provider,
        )
        for duration_min in duration_minutes:
            for magnitude in magnitudes:
                config.validate_spike(magnitude, duration_min)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    requests = common.load_workload(
        dataset=args.workload,
        duration_sec=args.duration_sec,
        max_requests=args.max_requests,
    )
    scenarios = make_scenarios(
        magnitudes=magnitudes,
        duration_minutes=duration_minutes,
        requests=requests,
        config=config,
    )
    presets = make_presets(window_min=args.window_min, output_predictor=args.predictor)
    policies = tuple(args.policy) if args.policy else tuple(presets)
    unknown = [policy for policy in policies if policy not in presets]
    if unknown:
        known = ", ".join(presets)
        raise SystemExit(f"unknown stale-telemetry policy {unknown[0]!r}; known: {known}")

    rows = common.run_section(
        section_name=SECTION_NAME,
        scenarios=scenarios,
        policies=policies,
        presets=presets,
        seeds=tuple(args.seed) if args.seed else common.DEFAULT_SEEDS,
        section_runners={policy: _make_serial_runner(policy, presets) for policy in policies},
        parallel_cell_runner=partial(run_stale_telemetry_cell, config=config),
        workload_dataset=args.workload,
        duration_sec=args.duration_sec,
        max_requests=args.max_requests,
        output_dir=args.output_dir,
        retain_records=False,
        jobs=args.jobs,
    )
    summary_rows, timeseries_rows = enrich_rows(rows, scenarios, presets)
    common.write_json(args.output_dir / "summary.json", summary_rows)
    write_rows_csv(args.output_dir / "summary.csv", summary_rows)
    write_rows_csv(args.output_dir / "spike_timeseries.csv", timeseries_rows)
    print(
        json.dumps(
            {
                "section": SECTION_NAME,
                "rows": len(summary_rows),
                "timeseries_rows": len(timeseries_rows),
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


def _make_serial_runner(policy_name: str, presets: dict[str, dict[str, Any]]):
    def run(
        scenario: ScenarioConfig,
        requests: list[Request],
        seed: int,
        *,
        retain_records: bool = True,
    ) -> Run:
        return run_stale_telemetry_policy(
            scenario,
            requests,
            policy_name,
            presets=presets,
            seed=seed,
            retain_records=retain_records,
        )

    return run


__all__ = [
    "DEFAULT_MAGNITUDES",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PERIOD_MIN",
    "DEFAULT_POOL",
    "DEFAULT_PRE_ONSET_MIN",
    "DEFAULT_SPIKE_DURATION_MINUTES",
    "DEFAULT_WARMUP_MIN",
    "DEFAULT_WORKLOAD",
    "NO_MITIGATION_POLICY",
    "PHASES",
    "POOL_SCENARIOS",
    "SECTION_NAME",
    "TIMESERIES_BIN_SEC",
    "SpikeTrainConfig",
    "build_policy",
    "build_spike_schedule",
    "classify_phase",
    "enrich_rows",
    "main",
    "make_scenario",
    "make_scenarios",
    "parse_spike_artifact_label",
    "run_stale_telemetry_cell",
    "run_stale_telemetry_policy",
    "select_spiked_provider",
    "spike_artifact_label",
    "spike_onsets",
    "timeseries_bin",
    "timeseries_bin_start_min",
    "write_rows_csv",
]


if __name__ == "__main__":
    raise SystemExit(main())
