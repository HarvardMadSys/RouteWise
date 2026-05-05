"""Per-request simulation result container and basic aggregations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


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
    mids = (edges[:-1] + edges[1:]) / 2.0

    label_names = sorted(set(labels))
    fracs: dict[str, list[float]] = {name: [] for name in label_names}

    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (timestamps >= lo) & (timestamps < hi)
        window_labels = [labels[i] for i in range(len(labels)) if mask[i]]
        n_window = len(window_labels)
        for label_name in label_names:
            if n_window == 0:
                fracs[label_name].append(0.0)
            else:
                fracs[label_name].append(window_labels.count(label_name) / n_window)

    return mids, {name: np.array(values) for name, values in fracs.items()}


@dataclass
class SimulationRun:
    """Per-request results for one policy on one scenario."""

    policy: str
    ttft_ms: np.ndarray
    cost_usd: np.ndarray
    provider: list[str]
    timestamp: np.ndarray
    hedge_triggered: np.ndarray
    tier: list[str] = field(default_factory=list)
    quota_fraction_used: np.ndarray = field(default_factory=lambda: np.array([]))
    concurrency_utilization: np.ndarray = field(default_factory=lambda: np.array([]))
    rejected: np.ndarray = field(default_factory=lambda: np.array([]))

    def slo_violation_rate(self, slo_ms: float) -> float:
        """Return the fraction of requests whose TTFT exceeds the SLO."""
        return float(np.mean(self.ttft_ms > slo_ms))

    def mean_cost_usd(self) -> float:
        """Return mean per-request cost."""
        return float(np.mean(self.cost_usd))

    def p50_ms(self) -> float:
        """Return P50 TTFT."""
        return float(np.percentile(self.ttft_ms, 50))

    def p90_ms(self) -> float:
        """Return P90 TTFT."""
        return float(np.percentile(self.ttft_ms, 90))

    def p99_ms(self) -> float:
        """Return P99 TTFT."""
        return float(np.percentile(self.ttft_ms, 99))

    def hedge_rate(self) -> float:
        """Return fraction of requests that triggered a hedge."""
        return float(np.mean(self.hedge_triggered))

    def provider_fractions(self) -> dict[str, float]:
        """Return overall provider selection fractions."""
        total = len(self.provider)
        if total == 0:
            return {}
        names = set(self.provider)
        return {name: self.provider.count(name) / total for name in sorted(names)}

    def provider_fractions_over_time(
        self,
        window_sec: float = 300.0,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Compute rolling provider selection fractions in time windows."""
        return _rolling_fraction_series(self.provider, self.timestamp, window_sec)

    def tier_fractions(self) -> dict[str, float]:
        """Return overall tier selection fractions."""
        total = len(self.tier)
        if total == 0:
            return {}
        uniq = set(self.tier)
        return {tier: self.tier.count(tier) / total for tier in sorted(uniq)}

    def tier_fractions_over_time(
        self,
        window_sec: float = 300.0,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Compute rolling tier-selection fractions in time windows."""
        return _rolling_fraction_series(self.tier, self.timestamp, window_sec)


__all__ = ["SimulationRun"]
