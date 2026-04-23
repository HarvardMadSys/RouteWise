"""Simulation state tracking.

Maintains quota consumption, concurrency occupancy, and per-provider
latency history used by strategies for P50 estimation and shadow pricing.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np

from experiment.synthetic.provider import SyntheticProvider


@dataclass
class ProviderProfile:
    """Rolling latency profile for a provider.

    Stores (timestamp, ttft_ms) samples within a sliding window.
    """

    window_sec: float = 300.0  # 5 minutes
    samples: Deque[tuple[float, float]] = field(default_factory=deque)

    def add(self, t: float, ttft_ms: float) -> None:
        self.samples.append((t, ttft_ms))
        self._prune(t)

    def _prune(self, current_t: float) -> None:
        cutoff = current_t - self.window_sec
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def p50(self, current_t: float) -> float | None:
        """Return P50 TTFT in ms, or None if no samples."""
        self._prune(current_t)
        if not self.samples:
            return None
        return float(np.median([v for _, v in self.samples]))

    def p99(self, current_t: float) -> float | None:
        self._prune(current_t)
        if not self.samples:
            return None
        return float(np.percentile([v for _, v in self.samples], 99))

    def cdf_at(self, current_t: float, threshold_ms: float) -> float | None:
        """P(TTFT <= threshold) based on empirical samples."""
        self._prune(current_t)
        if not self.samples:
            return None
        vals = [v for _, v in self.samples]
        return float(np.mean([1.0 if v <= threshold_ms else 0.0 for v in vals]))

    def n_samples(self) -> int:
        return len(self.samples)


@dataclass
class SimState:
    """Simulation-wide state accessed by strategies.

    Attributes:
        current_time: Simulation time in seconds.
        quota_used: Per-provider daily quota consumption (req count).
        active_concurrency: Per-provider current in-flight request count.
        concurrency_free_times: For each S_C provider, a sorted list of when
            each active slot will free up (timestamp in seconds).
        profiles: Per-provider rolling latency profile.
        current_day: For S_Q daily quota resets.
    """

    current_time: float = 0.0
    current_day: int = 0
    quota_used: dict[str, int] = field(default_factory=dict)
    active_concurrency: dict[str, int] = field(default_factory=dict)
    concurrency_free_times: dict[str, list[float]] = field(default_factory=dict)
    profiles: dict[str, ProviderProfile] = field(default_factory=dict)

    @classmethod
    def initialize(cls, providers: list[SyntheticProvider]) -> "SimState":
        state = cls()
        for p in providers:
            state.quota_used[p.name] = 0
            state.active_concurrency[p.name] = 0
            state.concurrency_free_times[p.name] = []
            state.profiles[p.name] = ProviderProfile(window_sec=300.0)
        return state

    def advance_time(
        self, t: float, providers: list[SyntheticProvider], seconds_per_day: float
    ) -> None:
        """Update state for time passing to `t`.

        - Frees concurrency slots whose time has come.
        - Resets daily S_Q quotas at day boundaries.
        """
        self.current_time = t

        # Reset daily quotas.
        day = int(t // seconds_per_day)
        if day > self.current_day:
            for p in providers:
                if p.is_s_q:
                    self.quota_used[p.name] = 0
            self.current_day = day

        # Free concurrency slots.
        for p in providers:
            if not p.is_s_c:
                continue
            free_times = self.concurrency_free_times[p.name]
            while free_times and free_times[0] <= t:
                free_times.pop(0)
                self.active_concurrency[p.name] = max(
                    0, self.active_concurrency[p.name] - 1
                )

    def can_accept(self, provider: SyntheticProvider) -> bool:
        """Hard availability check: does this provider have capacity right now?"""
        if provider.is_s_q:
            return self.quota_used[provider.name] < provider.daily_quota
        if provider.is_s_c:
            return self.active_concurrency[provider.name] < provider.concurrency_limit
        return True  # S_A always accepts

    def register_dispatch(
        self, provider: SyntheticProvider, finish_time: float
    ) -> None:
        """Record that a request has been dispatched to this provider.

        For S_Q: increments quota counter.
        For S_C: occupies a slot until `finish_time`.
        """
        if provider.is_s_q:
            self.quota_used[provider.name] += 1
        elif provider.is_s_c:
            self.active_concurrency[provider.name] += 1
            # Keep the list sorted so advance_time can free slots in order.
            import bisect

            bisect.insort(self.concurrency_free_times[provider.name], finish_time)

    def quota_fraction(self, provider: SyntheticProvider) -> float:
        """Return z = quota_used / quota_limit for S_Q. 0 for non-S_Q."""
        if not provider.is_s_q or provider.daily_quota <= 0:
            return 0.0
        return min(1.0, self.quota_used[provider.name] / provider.daily_quota)

    def concurrency_fraction(self, provider: SyntheticProvider) -> float:
        """Return u = active / limit for S_C. 0 for non-S_C."""
        if not provider.is_s_c or provider.concurrency_limit <= 0:
            return 0.0
        return min(1.0, self.active_concurrency[provider.name] / provider.concurrency_limit)
