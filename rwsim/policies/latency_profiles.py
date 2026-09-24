"""Latency-profile strategies used by RouteWise policies."""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

if TYPE_CHECKING:
    from rwsim.world.providers import Provider

LatencyProfileMode = Literal["observed", "configured"]


class LatencyProfileStrategy(Protocol):
    """Source of latency objective and CDF estimates for RouteWise."""

    def ensure_provider(self, provider_name: str) -> None:
        """Initialize any provider-local state."""

    def mean_ms(self, provider: Provider, now: float) -> float:
        """Return the latency objective in milliseconds."""

    def cdf_ms(self, provider: Provider, value_ms: float, now: float) -> float:
        """Return ``P(provider latency <= value_ms)`` at ``now``."""

    def observe(self, provider_name: str, observed_at: float, ttft_ms: float) -> None:
        """Record an observed TTFT sample."""


@dataclass
class RollingLatencyProfile:
    """Causal moving-window empirical latency profile for one provider.

    The simulator queries profiles with nondecreasing route-observation time.
    ``mean`` and ``cdf`` keep defensive backwards-query fallbacks for tests and
    ad hoc probes, but hot-path callers should treat this as a monotonic-time
    API.
    """

    window_sec: float = 15 * 60.0
    samples: deque[tuple[float, float]] = field(default_factory=deque)
    _mean_pending: list[tuple[float, int, float]] = field(default_factory=list, init=False, repr=False)
    _mean_active: list[tuple[float, int, float]] = field(default_factory=list, init=False, repr=False)
    _mean_sum: float = field(default=0.0, init=False, repr=False)
    _mean_count: int = field(default=0, init=False, repr=False)
    _mean_clock_sec: float = field(default=float("-inf"), init=False, repr=False)
    _cdf_pending: list[tuple[float, int, float]] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _cdf_active: list[tuple[float, int, float]] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _cdf_active_values: dict[int, float] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _cdf_threshold_counts: dict[float, int] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _cdf_active_count: int = field(default=0, init=False, repr=False)
    _cdf_clock_sec: float = field(default=float("-inf"), init=False, repr=False)
    _sample_seq: int = field(default=0, init=False, repr=False)

    def add_sample(self, timestamp: float, ttft_ms: float) -> None:
        ts = float(timestamp)
        value = float(ttft_ms)
        self.samples.append((ts, value))
        seq = self._sample_seq
        item = (ts, seq, value)
        self._sample_seq += 1
        if ts <= self._mean_clock_sec:
            if ts >= self._mean_clock_sec - self.window_sec:
                heapq.heappush(self._mean_active, item)
                self._mean_sum += value
                self._mean_count += 1
        else:
            heapq.heappush(self._mean_pending, item)
        if self._cdf_clock_sec > float("-inf"):
            if ts <= self._cdf_clock_sec:
                if ts >= self._cdf_clock_sec - self.window_sec:
                    self._cdf_add_active(ts, seq, value)
            else:
                heapq.heappush(self._cdf_pending, item)

    def values(self, now: float) -> list[float]:
        cutoff = now - self.window_sec
        kept: deque[tuple[float, float]] = deque()
        visible: list[float] = []
        for ts, ttft_ms in self.samples:
            if ts < cutoff:
                continue
            kept.append((ts, ttft_ms))
            if ts <= now:
                visible.append(ttft_ms)
        self.samples = kept
        return visible

    def mean(self, now: float) -> float | None:
        if now < self._mean_clock_sec:
            values = [
                ttft_ms
                for ts, ttft_ms in self.samples
                if now - self.window_sec <= ts <= now
            ]
            if not values:
                return None
            return float(np.mean(values))

        self._advance_mean_window(float(now))
        if self._mean_count == 0:
            return None
        return self._mean_sum / self._mean_count

    def cdf(self, value_ms: float, now: float) -> float | None:
        if now < self._cdf_clock_sec:
            values = [
                ttft_ms
                for ts, ttft_ms in self.samples
                if now - self.window_sec <= ts <= now
            ]
            if not values:
                return None
            return float(np.mean(np.asarray(values) <= value_ms))

        now = float(now)
        if self._cdf_clock_sec == float("-inf"):
            self._initialize_cdf_window(now)
        else:
            self._advance_cdf_window(now)
        if self._cdf_active_count == 0:
            return None
        threshold = float(value_ms)
        if threshold not in self._cdf_threshold_counts:
            self._cdf_threshold_counts[threshold] = sum(
                1 for value in self._cdf_active_values.values() if value <= threshold
            )
        return self._cdf_threshold_counts[threshold] / self._cdf_active_count

    def _advance_mean_window(self, now: float) -> None:
        self._mean_clock_sec = now
        cutoff = now - self.window_sec
        while self._mean_pending and self._mean_pending[0][0] <= now:
            ts, seq, value = heapq.heappop(self._mean_pending)
            if ts >= cutoff:
                heapq.heappush(self._mean_active, (ts, seq, value))
                self._mean_sum += value
                self._mean_count += 1
        while self._mean_active and self._mean_active[0][0] < cutoff:
            _, _, value = heapq.heappop(self._mean_active)
            self._mean_sum -= value
            self._mean_count -= 1

    def _initialize_cdf_window(self, now: float) -> None:
        self._cdf_clock_sec = now
        cutoff = now - self.window_sec
        for seq, (ts, value) in enumerate(self.samples):
            if ts < cutoff:
                continue
            if ts <= now:
                self._cdf_add_active(ts, seq, value)
            else:
                heapq.heappush(self._cdf_pending, (ts, seq, value))

    def _advance_cdf_window(self, now: float) -> None:
        self._cdf_clock_sec = now
        cutoff = now - self.window_sec
        while self._cdf_pending and self._cdf_pending[0][0] <= now:
            ts, seq, value = heapq.heappop(self._cdf_pending)
            if ts >= cutoff:
                self._cdf_add_active(ts, seq, value)
        while self._cdf_active and self._cdf_active[0][0] < cutoff:
            _, seq, value = heapq.heappop(self._cdf_active)
            self._cdf_remove_active(seq, value)

    def _cdf_add_active(self, ts: float, seq: int, value: float) -> None:
        heapq.heappush(self._cdf_active, (ts, seq, value))
        self._cdf_active_values[seq] = value
        self._cdf_active_count += 1
        for threshold in self._cdf_threshold_counts:
            if value <= threshold:
                self._cdf_threshold_counts[threshold] += 1

    def _cdf_remove_active(self, seq: int, value: float) -> None:
        if seq not in self._cdf_active_values:
            return
        del self._cdf_active_values[seq]
        self._cdf_active_count -= 1
        for threshold in self._cdf_threshold_counts:
            if value <= threshold:
                self._cdf_threshold_counts[threshold] -= 1


@dataclass
class ObservedRollingLatencyProfileStrategy:
    """Latency profile learned from observed request outcomes."""

    window_sec: float = 15 * 60.0
    profiles: dict[str, RollingLatencyProfile] = field(default_factory=dict)

    def ensure_provider(self, provider_name: str) -> None:
        self._profile(provider_name)

    def mean_ms(self, provider: Provider, now: float) -> float:
        mean = self._profile(provider.name).mean(now)
        if mean is not None:
            return mean
        return provider.true_mean_ms(now)

    def cdf_ms(self, provider: Provider, value_ms: float, now: float) -> float:
        empirical = self._profile(provider.name).cdf(value_ms, now)
        if empirical is not None:
            return empirical
        return provider._active_ttft_dist(now).cdf(value_ms)

    def observe(self, provider_name: str, observed_at: float, ttft_ms: float) -> None:
        self._profile(provider_name).add_sample(observed_at, ttft_ms)

    def _profile(self, name: str) -> RollingLatencyProfile:
        return self.profiles.setdefault(
            name,
            RollingLatencyProfile(window_sec=self.window_sec),
        )


@dataclass
class ConfiguredLatencyProfileStrategy:
    """Latency profile that uses the configured provider distributions directly."""

    def ensure_provider(self, provider_name: str) -> None:
        del provider_name

    def mean_ms(self, provider: Provider, now: float) -> float:
        return provider.true_mean_ms(now)

    def cdf_ms(self, provider: Provider, value_ms: float, now: float) -> float:
        return provider._active_ttft_dist(now).cdf(value_ms)

    def observe(self, provider_name: str, observed_at: float, ttft_ms: float) -> None:
        del provider_name, observed_at, ttft_ms


def make_latency_profile_strategy(
    mode: LatencyProfileMode,
    *,
    window_sec: float,
    profiles: dict[str, RollingLatencyProfile] | None = None,
) -> LatencyProfileStrategy:
    """Build a RouteWise latency-profile strategy."""
    if mode == "observed":
        return ObservedRollingLatencyProfileStrategy(
            window_sec=window_sec,
            profiles={} if profiles is None else profiles,
        )
    if mode == "configured":
        return ConfiguredLatencyProfileStrategy()
    raise ValueError(
        "unknown latency_profile_mode "
        f"{mode!r}; expected 'observed' or 'configured'"
    )


__all__ = [
    "ConfiguredLatencyProfileStrategy",
    "LatencyProfileMode",
    "LatencyProfileStrategy",
    "ObservedRollingLatencyProfileStrategy",
    "RollingLatencyProfile",
    "make_latency_profile_strategy",
]
