"""Run-level metric container and aggregations."""

from __future__ import annotations

import math
from collections import Counter
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

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


class Run:
    """Per-request records for one policy on one source run."""

    def __init__(
        self,
        *,
        records: Sequence[PerRequestRecord],
        policy: str = "",
        scenario_name: str = "",
        source: str = "simulation",
    ) -> None:
        self.policy = policy
        self.scenario_name = scenario_name
        self.source = source
        self.records = list(records)

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
        counts = Counter(record.status.value for record in self.records)
        return {status.value: counts.get(status.value, 0) for status in Status}

    def mean_ttft_ms(self) -> float:
        """Return mean user-visible TTFT."""
        return _mean(self._ttft_ms())

    def p50_ms(self) -> float:
        """Return P50 user-visible TTFT."""
        return _percentile(self._ttft_ms(), 50)

    def p90_ms(self) -> float:
        """Return P90 user-visible TTFT."""
        return _percentile(self._ttft_ms(), 90)

    def p95_ms(self) -> float:
        """Return P95 user-visible TTFT."""
        return _percentile(self._ttft_ms(), 95)

    def p99_ms(self) -> float:
        """Return P99 user-visible TTFT."""
        return _percentile(self._ttft_ms(), 99)

    def mean_e2e_ms(self) -> float:
        """Return mean user-visible end-to-end latency."""
        return _mean(_defined(record.e2e_ms for record in self.records))

    def p50_e2e_ms(self) -> float:
        """Return P50 user-visible end-to-end latency."""
        return _percentile(_defined(record.e2e_ms for record in self.records), 50)

    def p90_e2e_ms(self) -> float:
        """Return P90 user-visible end-to-end latency."""
        return _percentile(_defined(record.e2e_ms for record in self.records), 90)

    def p99_e2e_ms(self) -> float:
        """Return P99 user-visible end-to-end latency."""
        return _percentile(_defined(record.e2e_ms for record in self.records), 99)

    def mean_cost_usd(self) -> float:
        """Return mean total per-request cost."""
        return _mean(self._cost_usd())

    def total_cost_usd(self) -> float:
        """Return total run cost."""
        return float(np.sum(self._cost_usd()))

    def cost_by_tier(self) -> dict[str, float]:
        """Return total cost attributed to primary and backup tiers."""
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
        if not self.records:
            return 0.0
        return float(np.mean(self._hedge_triggered()))

    def hedge_winner_rate(self) -> dict[str, float]:
        """Return primary/backup win fractions among triggered hedges."""
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
        return _fraction_map(self._final_providers())

    def provider_fractions_over_time(
        self,
        window_sec: float = 300.0,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Compute rolling final-provider selection fractions in time windows."""
        return _rolling_fraction_series(self._final_providers(), self._elapsed_sec(), window_sec)

    def tier_fractions(self) -> dict[str, float]:
        """Return overall final-tier selection fractions."""
        return _fraction_map(self._final_tiers())

    def tier_fractions_over_time(
        self,
        window_sec: float = 300.0,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Compute rolling final-tier selection fractions in time windows."""
        return _rolling_fraction_series(self._final_tiers(), self._elapsed_sec(), window_sec)


def _fraction_map(labels: list[str]) -> dict[str, float]:
    total = len(labels)
    if total == 0:
        return {}
    counts = Counter(labels)
    return {name: counts[name] / total for name in sorted(counts)}


__all__ = ["Run"]
