"""Run-level metric container and aggregations."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

from rwsim.metrics.histogram import TtftHistogram
from rwsim.metrics.record import PerRequestRecord, Status

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


def _rolling_fraction_series(
    labels: list[str],
    timestamps: np.ndarray,
    window_sec: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute rolling category fractions in fixed-width time windows."""
    if len(timestamps) == 0 or not labels:
        return np.array([]), {}

    t_min = float(timestamps[0])
    t_max = float(timestamps[-1])
    edges = np.arange(t_min, t_max + window_sec, window_sec)
    if len(edges) < 2:
        edges = np.array([t_min, t_min + window_sec])
    mids = (edges[:-1] + edges[1:]) / 2.0

    label_names = sorted(set(labels))
    fracs: dict[str, list[float]] = {name: [] for name in label_names}

    for lo, hi in pairwise(edges):
        mask = (timestamps >= lo) & (timestamps < hi)
        window_labels = [labels[i] for i in range(len(labels)) if mask[i]]
        n_window = len(window_labels)
        for label_name in label_names:
            if n_window == 0:
                fracs[label_name].append(0.0)
            else:
                fracs[label_name].append(window_labels.count(label_name) / n_window)

    return mids, {name: np.array(values) for name, values in fracs.items()}


def _percentile(values: Iterable[float], pct: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, pct))


def _mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _defined(values: Iterable[float | None]) -> list[float]:
    output: list[float] = []
    for value in values:
        if value is None:
            continue
        numeric = float(value)
        if not math.isnan(numeric):
            output.append(numeric)
    return output


@dataclass
class RunAggregate:
    """Streaming aggregate state for one run."""

    n: int = 0
    ttft_histogram: TtftHistogram = field(default_factory=TtftHistogram.default)
    e2e_histogram: TtftHistogram | None = None
    total_cost_usd: float = 0.0
    cost_count: int = 0
    status_counts: Counter[str] = field(default_factory=Counter)
    slo_violated_count: int = 0
    cost_by_tier: Counter[str] = field(default_factory=Counter)
    cost_by_provider: Counter[str] = field(default_factory=Counter)
    provider_counts: Counter[str] = field(default_factory=Counter)
    tier_counts: Counter[str] = field(default_factory=Counter)
    hedge_triggered_count: int = 0
    hedge_total_count: int = 0
    hedge_winner_counts: Counter[str] = field(default_factory=Counter)
    provider_window_counts: dict[int, Counter[str]] = field(default_factory=dict)
    tier_window_counts: dict[int, Counter[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for status in Status:
            self.status_counts.setdefault(status.value, 0)


class Run:
    """Per-request records for one policy on one source run."""

    def __init__(
        self,
        *,
        records: Sequence[PerRequestRecord],
        policy: str = "",
        scenario_name: str = "",
        source: str = "simulation",
        aggregate: RunAggregate | None = None,
        window_sec: float = 300.0,
    ) -> None:
        self.policy = policy
        self.scenario_name = scenario_name
        self.source = source
        self.records = list(records)
        self.aggregate = aggregate
        self.window_sec = float(window_sec)

    def _ttft_ms(self) -> np.ndarray:
        return np.asarray([record.ttft_ms for record in self.records], dtype=float)

    def _e2e_ms(self) -> np.ndarray:
        return np.asarray(
            [np.nan if record.e2e_ms is None else record.e2e_ms for record in self.records],
            dtype=float,
        )

    def _cost_usd(self) -> np.ndarray:
        return np.asarray([record.total_cost_usd for record in self.records], dtype=float)

    def _final_providers(self) -> list[str]:
        return [record.final_provider for record in self.records]

    def _final_tiers(self) -> list[str]:
        return [record.final_tier for record in self.records if record.final_tier]

    def _elapsed_sec(self) -> np.ndarray:
        return np.asarray([record.elapsed_sec for record in self.records], dtype=float)

    def _hedge_triggered(self) -> np.ndarray:
        return np.asarray([record.hedge_triggered for record in self.records], dtype=bool)

    def slo_violation_rate(self, slo_ms: float | None = None) -> float:
        """Return the fraction of requests whose user-visible TTFT violates SLO."""
        if self.aggregate is not None and not self.records:
            if self.aggregate.n == 0:
                return 0.0
            if slo_ms is None:
                return self.aggregate.slo_violated_count / self.aggregate.n
            non_success = self.aggregate.n - self.aggregate.status_counts.get(
                Status.SUCCESS.value,
                0,
            )
            approximate_latency_violations = self.aggregate.n * (
                1.0 - self.aggregate.ttft_histogram.cdf(float(slo_ms))
            )
            return min(
                max((approximate_latency_violations + non_success) / self.aggregate.n, 0.0),
                1.0,
            )
        if not self.records:
            return 0.0
        if slo_ms is None:
            return float(np.mean([record.slo_violated for record in self.records]))
        non_success = np.asarray(
            [record.status != Status.SUCCESS for record in self.records],
            dtype=bool,
        )
        return float(np.mean((self._ttft_ms() > slo_ms) | non_success))

    def status_breakdown(self) -> dict[str, int]:
        """Return per-status counts."""
        if self.aggregate is not None and not self.records:
            counts = self.aggregate.status_counts
            return {status.value: counts.get(status.value, 0) for status in Status}
        counts = Counter(record.status.value for record in self.records)
        return {status.value: counts.get(status.value, 0) for status in Status}

    def mean_ttft_ms(self) -> float:
        """Return mean user-visible TTFT."""
        if self.aggregate is not None and not self.records:
            return self.aggregate.ttft_histogram.mean()
        return _mean(self._ttft_ms())

    def p50_ms(self) -> float:
        """Return P50 user-visible TTFT."""
        if self.aggregate is not None and not self.records:
            return self.aggregate.ttft_histogram.quantile(0.50)
        return _percentile(self._ttft_ms(), 50)

    def p90_ms(self) -> float:
        """Return P90 user-visible TTFT."""
        if self.aggregate is not None and not self.records:
            return self.aggregate.ttft_histogram.quantile(0.90)
        return _percentile(self._ttft_ms(), 90)

    def p95_ms(self) -> float:
        """Return P95 user-visible TTFT."""
        if self.aggregate is not None and not self.records:
            return self.aggregate.ttft_histogram.quantile(0.95)
        return _percentile(self._ttft_ms(), 95)

    def p99_ms(self) -> float:
        """Return P99 user-visible TTFT."""
        if self.aggregate is not None and not self.records:
            return self.aggregate.ttft_histogram.quantile(0.99)
        return _percentile(self._ttft_ms(), 99)

    def mean_e2e_ms(self) -> float:
        """Return mean user-visible end-to-end latency."""
        if self.aggregate is not None and not self.records:
            if self.aggregate.e2e_histogram is None:
                return float("nan")
            return self.aggregate.e2e_histogram.mean()
        return _mean(_defined(record.e2e_ms for record in self.records))

    def p50_e2e_ms(self) -> float:
        """Return P50 user-visible end-to-end latency."""
        if self.aggregate is not None and not self.records:
            if self.aggregate.e2e_histogram is None:
                return float("nan")
            return self.aggregate.e2e_histogram.quantile(0.50)
        return _percentile(_defined(record.e2e_ms for record in self.records), 50)

    def p90_e2e_ms(self) -> float:
        """Return P90 user-visible end-to-end latency."""
        if self.aggregate is not None and not self.records:
            if self.aggregate.e2e_histogram is None:
                return float("nan")
            return self.aggregate.e2e_histogram.quantile(0.90)
        return _percentile(_defined(record.e2e_ms for record in self.records), 90)

    def p99_e2e_ms(self) -> float:
        """Return P99 user-visible end-to-end latency."""
        if self.aggregate is not None and not self.records:
            if self.aggregate.e2e_histogram is None:
                return float("nan")
            return self.aggregate.e2e_histogram.quantile(0.99)
        return _percentile(_defined(record.e2e_ms for record in self.records), 99)

    def mean_cost_usd(self) -> float:
        """Return mean total per-request cost."""
        if self.aggregate is not None and not self.records:
            if self.aggregate.cost_count == 0:
                return float("nan")
            return self.aggregate.total_cost_usd / self.aggregate.cost_count
        return _mean(self._cost_usd())

    def total_cost_usd(self) -> float:
        """Return total run cost."""
        if self.aggregate is not None and not self.records:
            return float(self.aggregate.total_cost_usd)
        return float(np.sum(self._cost_usd()))

    def cost_by_tier(self) -> dict[str, float]:
        """Return total cost attributed to primary and backup tiers."""
        if self.aggregate is not None and not self.records:
            return {key: float(self.aggregate.cost_by_tier[key]) for key in sorted(self.aggregate.cost_by_tier)}
        totals: dict[str, float] = {}
        for record in self.records:
            backup_cost = record.backup_cost_usd
            if record.primary_tier:
                totals[record.primary_tier] = (
                    totals.get(record.primary_tier, 0.0) + record.primary_cost_usd
                )
            if backup_cost is not None and record.backup_tier:
                totals[record.backup_tier] = totals.get(record.backup_tier, 0.0) + backup_cost
        return {key: totals[key] for key in sorted(totals)}

    def cost_by_provider(self) -> dict[str, float]:
        """Return total cost attributed to primary and backup providers."""
        if self.aggregate is not None and not self.records:
            return {
                key: float(self.aggregate.cost_by_provider[key])
                for key in sorted(self.aggregate.cost_by_provider)
            }
        totals: dict[str, float] = {}
        for record in self.records:
            backup_cost = record.backup_cost_usd
            totals[record.primary_provider] = (
                totals.get(record.primary_provider, 0.0) + record.primary_cost_usd
            )
            if backup_cost is not None and record.backup_provider:
                totals[record.backup_provider] = (
                    totals.get(record.backup_provider, 0.0) + backup_cost
                )
        return {key: totals[key] for key in sorted(totals)}

    def hedge_rate(self) -> float:
        """Return fraction of requests that triggered a hedge."""
        if self.aggregate is not None and not self.records:
            if self.aggregate.hedge_total_count == 0:
                return 0.0
            return self.aggregate.hedge_triggered_count / self.aggregate.hedge_total_count
        if not self.records:
            return 0.0
        return float(np.mean(self._hedge_triggered()))

    def hedge_winner_rate(self) -> dict[str, float]:
        """Return primary/backup win fractions among triggered hedges."""
        if self.aggregate is not None and not self.records:
            total = sum(self.aggregate.hedge_winner_counts.values())
            if total == 0:
                return {}
            return {
                name: self.aggregate.hedge_winner_counts[name] / total
                for name in sorted(self.aggregate.hedge_winner_counts)
            }
        winners = [
            record.hedge_winner
            for record in self.records
            if record.hedge_triggered and record.hedge_winner
        ]
        total = len(winners)
        if total == 0:
            return {}
        counts = Counter(winners)
        return {name: counts[name] / total for name in sorted(counts)}

    def provider_fractions(self) -> dict[str, float]:
        """Return overall final-provider selection fractions."""
        if self.aggregate is not None and not self.records:
            return _counter_fraction_map(self.aggregate.provider_counts)
        return _fraction_map(self._final_providers())

    def provider_fractions_over_time(
        self,
        window_sec: float = 300.0,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Compute rolling final-provider selection fractions in time windows."""
        if self.aggregate is not None and not self.records:
            return _window_counter_fraction_series(
                self.aggregate.provider_window_counts,
                window_sec=window_sec,
            )
        return _rolling_fraction_series(self._final_providers(), self._elapsed_sec(), window_sec)

    def tier_fractions(self) -> dict[str, float]:
        """Return overall final-tier selection fractions."""
        if self.aggregate is not None and not self.records:
            return _counter_fraction_map(self.aggregate.tier_counts)
        return _fraction_map(self._final_tiers())

    def tier_fractions_over_time(
        self,
        window_sec: float = 300.0,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Compute rolling final-tier selection fractions in time windows."""
        if self.aggregate is not None and not self.records:
            return _window_counter_fraction_series(
                self.aggregate.tier_window_counts,
                window_sec=window_sec,
            )
        return _rolling_fraction_series(self._final_tiers(), self._elapsed_sec(), window_sec)

    def ttft_histogram(self) -> TtftHistogram:
        """Return a TTFT histogram for this run."""
        if self.aggregate is not None:
            return self.aggregate.ttft_histogram
        histogram = TtftHistogram.default()
        histogram.add_array(self._ttft_ms())
        return histogram

    def e2e_histogram(self) -> TtftHistogram | None:
        """Return an E2E histogram for this run, if E2E values exist."""
        if self.aggregate is not None:
            return self.aggregate.e2e_histogram
        values = self._e2e_ms()
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        histogram = TtftHistogram.default()
        histogram.add_array(values)
        return histogram


def _fraction_map(labels: list[str]) -> dict[str, float]:
    total = len(labels)
    if total == 0:
        return {}
    counts = Counter(labels)
    return {name: counts[name] / total for name in sorted(counts)}


def _counter_fraction_map(counts: Counter[str]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {}
    return {name: counts[name] / total for name in sorted(counts)}


def _window_counter_fraction_series(
    windows: dict[int, Counter[str]],
    *,
    window_sec: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not windows:
        return np.array([]), {}
    window_ids = list(range(min(windows), max(windows) + 1))
    label_names = sorted({name for counts in windows.values() for name in counts})
    mids = np.asarray([(idx + 0.5) * window_sec for idx in window_ids], dtype=float)
    values: dict[str, list[float]] = {name: [] for name in label_names}
    for window_id in window_ids:
        counts = windows.get(window_id, Counter())
        total = sum(counts.values())
        for name in label_names:
            values[name].append((counts[name] / total) if total else 0.0)
    return mids, {name: np.asarray(series, dtype=float) for name, series in values.items()}


__all__ = ["Run", "RunAggregate"]
