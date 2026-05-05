"""Run-level metric container and aggregations."""

from __future__ import annotations

import math
from collections import Counter
from itertools import pairwise
from typing import TYPE_CHECKING, Any

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


def _array_or_default(
    values: Sequence[Any] | np.ndarray | None,
    n: int,
    default: Any,
    *,
    dtype: Any | None = None,
) -> np.ndarray:
    if values is None:
        return np.full(n, default, dtype=dtype)
    return np.asarray(values, dtype=dtype)


def _list_or_default(values: Sequence[str] | None, n: int, default: str = "") -> list[str]:
    if values is None:
        return [default for _ in range(n)]
    return list(values)


def _metadata_fraction_value(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, dict):
        numeric = [float(item) for item in value.values() if item is not None]
        return float(max(numeric, default=0.0))
    return float(value)


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
    """Per-request records for one policy on one source run.

    ``Run`` is row-native via :class:`PerRequestRecord`, but it still exposes
    the legacy column properties used by existing simulation summaries.
    """

    def __init__(
        self,
        *,
        records: Sequence[PerRequestRecord] | None = None,
        policy: str = "",
        scenario_name: str = "",
        source: str = "simulation",
        ttft_ms: Sequence[float] | np.ndarray | None = None,
        cost_usd: Sequence[float] | np.ndarray | None = None,
        provider: Sequence[str] | None = None,
        timestamp: Sequence[float] | np.ndarray | None = None,
        hedge_triggered: Sequence[bool] | np.ndarray | None = None,
        tier: Sequence[str] | None = None,
        quota_fraction_used: Sequence[float] | np.ndarray | None = None,
        concurrency_utilization: Sequence[float] | np.ndarray | None = None,
        rejected: Sequence[bool] | np.ndarray | None = None,
        slo_ms: float | None = None,
    ) -> None:
        self.policy = policy
        self.scenario_name = scenario_name
        self.source = source
        if records is not None:
            self.records = list(records)
        else:
            self.records = self._records_from_legacy_columns(
                policy=policy,
                ttft_ms=ttft_ms,
                cost_usd=cost_usd,
                provider=provider,
                timestamp=timestamp,
                hedge_triggered=hedge_triggered,
                tier=tier,
                quota_fraction_used=quota_fraction_used,
                concurrency_utilization=concurrency_utilization,
                rejected=rejected,
                slo_ms=slo_ms,
            )

    @staticmethod
    def _records_from_legacy_columns(
        *,
        policy: str,
        ttft_ms: Sequence[float] | np.ndarray | None,
        cost_usd: Sequence[float] | np.ndarray | None,
        provider: Sequence[str] | None,
        timestamp: Sequence[float] | np.ndarray | None,
        hedge_triggered: Sequence[bool] | np.ndarray | None,
        tier: Sequence[str] | None,
        quota_fraction_used: Sequence[float] | np.ndarray | None,
        concurrency_utilization: Sequence[float] | np.ndarray | None,
        rejected: Sequence[bool] | np.ndarray | None,
        slo_ms: float | None,
    ) -> list[PerRequestRecord]:
        n = len(ttft_ms) if ttft_ms is not None else 0
        ttft_values = _array_or_default(ttft_ms, n, 0.0, dtype=float)
        cost_values = _array_or_default(cost_usd, n, 0.0, dtype=float)
        timestamp_values = _array_or_default(timestamp, n, 0.0, dtype=float)
        hedge_values = _array_or_default(hedge_triggered, n, False, dtype=bool)
        rejected_values = _array_or_default(rejected, n, False, dtype=bool)
        provider_values = _list_or_default(provider, n)
        tier_values = _list_or_default(tier, n)
        quota_values = _array_or_default(quota_fraction_used, n, np.nan, dtype=float)
        concurrency_values = _array_or_default(
            concurrency_utilization,
            n,
            np.nan,
            dtype=float,
        )

        records: list[PerRequestRecord] = []
        for idx in range(n):
            metadata: dict[str, Any] = {}
            if not math.isnan(float(quota_values[idx])):
                metadata["sim_quota_fraction_used"] = float(quota_values[idx])
            if not math.isnan(float(concurrency_values[idx])):
                metadata["sim_concurrency_utilization"] = float(concurrency_values[idx])
            status = Status.REJECTED if bool(rejected_values[idx]) else Status.SUCCESS
            slo_value = None if slo_ms is None else float(slo_ms)
            slo_violated = (
                False
                if slo_value is None
                else status != Status.SUCCESS or float(ttft_values[idx]) > slo_value
            )
            records.append(
                PerRequestRecord(
                    request_id=str(idx),
                    elapsed_sec=float(timestamp_values[idx]),
                    policy=policy,
                    prompt_tokens=0,
                    completion_tokens_budget=None,
                    completion_tokens_actual=None,
                    primary_provider=provider_values[idx],
                    primary_tier=tier_values[idx],
                    final_provider=provider_values[idx],
                    final_tier=tier_values[idx],
                    ttft_ms=float(ttft_values[idx]),
                    e2e_ms=None,
                    primary_local_ttft_ms=float(ttft_values[idx]),
                    slo_ms=slo_value,
                    slo_violated=slo_violated,
                    total_cost_usd=float(cost_values[idx]),
                    primary_cost_usd=float(cost_values[idx]),
                    backup_cost_usd=0.0,
                    hedge_triggered=bool(hedge_values[idx]),
                    hedge_winner=None,
                    status=status,
                    metadata=metadata,
                )
            )
        return records

    @property
    def ttft_ms(self) -> np.ndarray:
        return np.asarray([record.ttft_ms for record in self.records], dtype=float)

    @property
    def e2e_ms(self) -> np.ndarray:
        return np.asarray(
            [np.nan if record.e2e_ms is None else record.e2e_ms for record in self.records],
            dtype=float,
        )

    @property
    def cost_usd(self) -> np.ndarray:
        return np.asarray([record.total_cost_usd for record in self.records], dtype=float)

    @property
    def provider(self) -> list[str]:
        return [record.final_provider for record in self.records]

    @property
    def tier(self) -> list[str]:
        return [record.final_tier for record in self.records if record.final_tier]

    @property
    def timestamp(self) -> np.ndarray:
        return self.elapsed_sec

    @property
    def elapsed_sec(self) -> np.ndarray:
        return np.asarray([record.elapsed_sec for record in self.records], dtype=float)

    @property
    def hedge_triggered(self) -> np.ndarray:
        return np.asarray([record.hedge_triggered for record in self.records], dtype=bool)

    @property
    def quota_fraction_used(self) -> np.ndarray:
        return np.asarray(
            [
                _metadata_fraction_value(record.metadata.get("sim_quota_fraction_used"))
                for record in self.records
            ],
            dtype=float,
        )

    @property
    def concurrency_utilization(self) -> np.ndarray:
        return np.asarray(
            [
                _metadata_fraction_value(record.metadata.get("sim_concurrency_utilization"))
                for record in self.records
            ],
            dtype=float,
        )

    @property
    def rejected(self) -> np.ndarray:
        return np.asarray(
            [record.status == Status.REJECTED for record in self.records],
            dtype=bool,
        )

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
        return float(np.mean((self.ttft_ms > slo_ms) | non_success))

    def status_breakdown(self) -> dict[str, int]:
        """Return per-status counts."""
        counts = Counter(record.status.value for record in self.records)
        return {status.value: counts.get(status.value, 0) for status in Status}

    def mean_ttft_ms(self) -> float:
        """Return mean user-visible TTFT."""
        return _mean(self.ttft_ms)

    def p50_ms(self) -> float:
        """Return P50 user-visible TTFT."""
        return _percentile(self.ttft_ms, 50)

    def p90_ms(self) -> float:
        """Return P90 user-visible TTFT."""
        return _percentile(self.ttft_ms, 90)

    def p95_ms(self) -> float:
        """Return P95 user-visible TTFT."""
        return _percentile(self.ttft_ms, 95)

    def p99_ms(self) -> float:
        """Return P99 user-visible TTFT."""
        return _percentile(self.ttft_ms, 99)

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
        return _mean(self.cost_usd)

    def total_cost_usd(self) -> float:
        """Return total run cost."""
        return float(np.sum(self.cost_usd))

    def cost_by_tier(self) -> dict[str, float]:
        """Return total cost attributed to primary and backup tiers."""
        totals: dict[str, float] = {}
        for record in self.records:
            primary_cost = record.primary_cost_usd
            backup_cost = record.backup_cost_usd
            if primary_cost is None and backup_cost is None:
                totals[record.final_tier] = (
                    totals.get(record.final_tier, 0.0) + record.total_cost_usd
                )
                continue
            if primary_cost is not None and record.primary_tier:
                totals[record.primary_tier] = totals.get(record.primary_tier, 0.0) + primary_cost
            if backup_cost is not None and record.backup_tier:
                totals[record.backup_tier] = totals.get(record.backup_tier, 0.0) + backup_cost
        return {key: totals[key] for key in sorted(totals)}

    def cost_by_provider(self) -> dict[str, float]:
        """Return total cost attributed to primary and backup providers."""
        totals: dict[str, float] = {}
        for record in self.records:
            primary_cost = record.primary_cost_usd
            backup_cost = record.backup_cost_usd
            if primary_cost is None and backup_cost is None:
                totals[record.final_provider] = (
                    totals.get(record.final_provider, 0.0) + record.total_cost_usd
                )
                continue
            if primary_cost is not None:
                totals[record.primary_provider] = (
                    totals.get(record.primary_provider, 0.0) + primary_cost
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
        return float(np.mean(self.hedge_triggered))

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
        return _fraction_map(self.provider)

    def provider_fractions_over_time(
        self,
        window_sec: float = 300.0,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Compute rolling final-provider selection fractions in time windows."""
        return _rolling_fraction_series(self.provider, self.elapsed_sec, window_sec)

    def tier_fractions(self) -> dict[str, float]:
        """Return overall final-tier selection fractions."""
        return _fraction_map(self.tier)

    def tier_fractions_over_time(
        self,
        window_sec: float = 300.0,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Compute rolling final-tier selection fractions in time windows."""
        return _rolling_fraction_series(self.tier, self.elapsed_sec, window_sec)


def _fraction_map(labels: list[str]) -> dict[str, float]:
    total = len(labels)
    if total == 0:
        return {}
    counts = Counter(labels)
    return {name: counts[name] / total for name in sorted(counts)}


SimulationRun = Run

__all__ = ["Run", "SimulationRun"]
