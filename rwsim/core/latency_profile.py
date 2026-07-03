"""Shared rolling latency-profile estimator.

This module is part of the environment-agnostic algorithm core: the profile
consumes ``(timestamp, ttft_ms)`` samples and answers mean/CDF queries over a
causal moving window. Both the simulator policies and the live real-eval
harness parameterize the same estimator; only the source of the samples
(mocked provider vs live provider) differs.
"""

from __future__ import annotations

import heapq
from bisect import bisect_left, bisect_right, insort
from collections import deque
from dataclasses import dataclass, field
from statistics import fmean

# Default causal window for online provider latency profiles. Single source of
# truth shared by the simulator policies and the real-eval harness.
DEFAULT_PROFILE_WINDOW_SEC: float = 15 * 60.0

# Batch size for pruning the ts-sorted sample index; deleting the stale front
# in chunks keeps the amortized cost O(1) per sample.
_PRUNE_BATCH = 1024


def _entry_ts(entry: tuple[float, int, float]) -> float:
    return entry[0]


@dataclass
class RollingLatencyProfile:
    """Causal moving-window empirical latency profile for one provider.

    The persistent window structures advance monotonically with the highest
    query time seen so far. Hedging callers also probe bounded distances in
    both directions around that clock (in-flight checkpoint times); those
    queries are answered exactly by delta-correcting the persistent window
    over the small time stripes that differ, using a ts-sorted sample index.
    Queries further than one window behind the clock fall back to a defensive
    full scan.
    """

    window_sec: float = DEFAULT_PROFILE_WINDOW_SEC
    samples: deque[tuple[float, float]] = field(default_factory=deque)
    # Failed attempts (429s, timeouts) recorded as (timestamp, error_type).
    # They carry no TTFT; the penalized-mean and error-counting-CDF queries
    # fold them in as synthetic worst-case observations.
    error_samples: deque[tuple[float, str]] = field(default_factory=deque)
    _error_clock_sec: float = field(default=float("-inf"), init=False, repr=False)
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
    # Active in-window values kept sorted so cdf queries are one bisect
    # instead of a per-add/remove sweep over every queried threshold.
    _cdf_sorted_values: list[float] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _cdf_active_count: int = field(default=0, init=False, repr=False)
    _cdf_clock_sec: float = field(default=float("-inf"), init=False, repr=False)
    _sample_seq: int = field(default=0, init=False, repr=False)
    # All samples ordered by (ts, seq), pruned to a two-window horizon behind
    # the oldest live clock. Serves the bounded backwards-query stripes.
    _samples_by_ts: list[tuple[float, int, float]] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    def add_sample(self, timestamp: float, ttft_ms: float) -> None:
        ts = float(timestamp)
        value = float(ttft_ms)
        self.samples.append((ts, value))
        seq = self._sample_seq
        item = (ts, seq, value)
        self._sample_seq += 1
        if not self._samples_by_ts or ts >= self._samples_by_ts[-1][0]:
            self._samples_by_ts.append(item)
        else:
            insort(self._samples_by_ts, item)
        self._prune_sample_index()
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

    def add_error(self, timestamp: float, error_type: str) -> None:
        """Record a failed attempt (no TTFT) at ``timestamp``."""
        self.error_samples.append((float(timestamp), str(error_type)))

    def error_count(self, now: float) -> int:
        """Number of failed attempts inside the causal window at ``now``."""
        now = float(now)
        if now > self._error_clock_sec:
            self._error_clock_sec = now
            cutoff = self._error_clock_sec - 2.0 * self.window_sec
            while self.error_samples and self.error_samples[0][0] < cutoff:
                self.error_samples.popleft()
        lo = now - self.window_sec
        return sum(1 for ts, _ in self.error_samples if lo <= ts <= now)

    def count(self, now: float) -> int:
        """Number of successful samples inside the causal window at ``now``."""
        now = float(now)
        if now < self._mean_clock_sec:
            return self._sum_count_backwards(now)[1]
        self._advance_mean_window(now)
        return self._mean_count

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
            return self._mean_backwards(float(now))

        self._advance_mean_window(float(now))
        if self._mean_count == 0:
            return None
        return self._mean_sum / self._mean_count

    def cdf(self, value_ms: float, now: float) -> float | None:
        if now < self._cdf_clock_sec:
            return self._cdf_backwards(float(value_ms), float(now))

        now = float(now)
        if self._cdf_clock_sec == float("-inf"):
            self._initialize_cdf_window(now)
        else:
            self._advance_cdf_window(now)
        if self._cdf_active_count == 0:
            return None
        covered = bisect_right(self._cdf_sorted_values, float(value_ms))
        return covered / self._cdf_active_count

    def mean_with_errors(
        self,
        now: float,
        *,
        error_penalty_ms: float,
    ) -> float | None:
        """Mean over successes plus a synthetic latency for failed attempts.

        With zero recorded errors this is exactly :meth:`mean`, so callers
        that never record errors (the simulator) see bit-identical values.
        """
        n_err = self.error_count(now)
        if n_err == 0:
            return self.mean(now)
        now = float(now)
        if now < self._mean_clock_sec:
            total, n = self._sum_count_backwards(now)
        else:
            self._advance_mean_window(now)
            total, n = self._mean_sum, self._mean_count
        return (total + n_err * float(error_penalty_ms)) / (n + n_err)

    def cdf_counting_errors(self, value_ms: float, now: float) -> float | None:
        """``P(TTFT <= value_ms)`` treating failed attempts as misses.

        With zero recorded errors this is exactly :meth:`cdf`.
        """
        n_err = self.error_count(now)
        if n_err == 0:
            return self.cdf(value_ms, now)
        covered, n = self._cdf_counts(float(value_ms), float(now))
        return covered / (n + n_err)

    def _mean_backwards(self, now: float) -> float | None:
        """Answer ``mean`` for a query behind the persistent mean clock."""
        stripes = self._backwards_stripes(now, self._mean_clock_sec)
        if stripes is None:
            values = [
                ttft_ms
                for ts, ttft_ms in self.samples
                if now - self.window_sec <= ts <= now
            ]
            if not values:
                return None
            return float(fmean(values))
        drop, restore = stripes
        total = self._mean_sum
        count = self._mean_count
        for _ts, _seq, value in drop:
            total -= value
            count -= 1
        for _ts, _seq, value in restore:
            total += value
            count += 1
        if count <= 0:
            return None
        return total / count

    def _cdf_backwards(self, threshold: float, now: float) -> float | None:
        """Answer ``cdf`` for a query behind the persistent cdf clock."""
        stripes = self._backwards_stripes(now, self._cdf_clock_sec)
        if stripes is None:
            values = [
                ttft_ms
                for ts, ttft_ms in self.samples
                if now - self.window_sec <= ts <= now
            ]
            if not values:
                return None
            return sum(1 for value in values if value <= threshold) / len(values)
        drop, restore = stripes
        covered = bisect_right(self._cdf_sorted_values, threshold)
        count = self._cdf_active_count
        for _ts, _seq, value in drop:
            count -= 1
            if value <= threshold:
                covered -= 1
        for _ts, _seq, value in restore:
            count += 1
            if value <= threshold:
                covered += 1
        if count <= 0:
            return None
        return covered / count

    def _sum_count_backwards(self, now: float) -> tuple[float, int]:
        """(sum, count) of window samples for a query behind the mean clock.

        Mirrors :meth:`_mean_backwards` but exposes the raw aggregates for
        the count and penalized-mean queries. Kept separate so the original
        ``mean`` float paths stay untouched.
        """
        stripes = self._backwards_stripes(now, self._mean_clock_sec)
        if stripes is None:
            values = [
                ttft_ms
                for ts, ttft_ms in self.samples
                if now - self.window_sec <= ts <= now
            ]
            return (float(sum(values)), len(values))
        drop, restore = stripes
        total = self._mean_sum
        count = self._mean_count
        for _ts, _seq, value in drop:
            total -= value
            count -= 1
        for _ts, _seq, value in restore:
            total += value
            count += 1
        return (total, count)

    def _cdf_counts(self, threshold: float, now: float) -> tuple[int, int]:
        """(covered, count) of window samples at ``now``.

        Integer twin of :meth:`cdf`; the division in callers is then exactly
        the one :meth:`cdf` would perform when the error count is zero.
        """
        if now < self._cdf_clock_sec:
            stripes = self._backwards_stripes(now, self._cdf_clock_sec)
            if stripes is None:
                values = [
                    ttft_ms
                    for ts, ttft_ms in self.samples
                    if now - self.window_sec <= ts <= now
                ]
                return (sum(1 for value in values if value <= threshold), len(values))
            drop, restore = stripes
            covered = bisect_right(self._cdf_sorted_values, threshold)
            count = self._cdf_active_count
            for _ts, _seq, value in drop:
                count -= 1
                if value <= threshold:
                    covered -= 1
            for _ts, _seq, value in restore:
                count += 1
                if value <= threshold:
                    covered += 1
            return (covered, count)
        if self._cdf_clock_sec == float("-inf"):
            self._initialize_cdf_window(now)
        else:
            self._advance_cdf_window(now)
        return (bisect_right(self._cdf_sorted_values, threshold), self._cdf_active_count)

    def _backwards_stripes(
        self,
        now: float,
        clock: float,
    ) -> tuple[list[tuple[float, int, float]], list[tuple[float, int, float]]] | None:
        """Return the (drop, restore) sample stripes for a backwards query.

        The persistent window at ``clock`` holds samples with ``ts`` in
        ``[clock - W, clock]``. A query at ``now < clock`` needs
        ``[now - W, now]``, i.e. the persistent set minus the stripe
        ``(now, clock]`` plus the expired stripe ``[now - W, clock - W)``.
        Returns ``None`` when the query is further than one window behind the
        clock; the caller must then use the defensive full scan. Queries
        within that range are always covered by the pruned index, which
        retains two windows behind the oldest live clock.
        """
        window_start = now - self.window_sec
        if now < clock - self.window_sec:
            return None
        drop_lo = bisect_right(self._samples_by_ts, now, key=_entry_ts)
        drop_hi = bisect_right(self._samples_by_ts, clock, key=_entry_ts)
        restore_lo = bisect_left(self._samples_by_ts, window_start, key=_entry_ts)
        restore_hi = bisect_left(self._samples_by_ts, clock - self.window_sec, key=_entry_ts)
        return (
            self._samples_by_ts[drop_lo:drop_hi],
            self._samples_by_ts[restore_lo:restore_hi],
        )

    def _prune_sample_index(self) -> None:
        clocks = [
            clock
            for clock in (self._mean_clock_sec, self._cdf_clock_sec)
            if clock > float("-inf")
        ]
        if not clocks:
            return
        cutoff = min(clocks) - 2.0 * self.window_sec
        if len(self._samples_by_ts) < _PRUNE_BATCH or self._samples_by_ts[0][0] >= cutoff:
            return
        keep_from = bisect_left(self._samples_by_ts, cutoff, key=_entry_ts)
        if keep_from > 0:
            del self._samples_by_ts[:keep_from]

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
        insort(self._cdf_sorted_values, value)

    def _cdf_remove_active(self, seq: int, value: float) -> None:
        if seq not in self._cdf_active_values:
            return
        del self._cdf_active_values[seq]
        self._cdf_active_count -= 1
        index = bisect_right(self._cdf_sorted_values, value) - 1
        # Equal values are interchangeable; drop the rightmost occurrence.
        del self._cdf_sorted_values[index]


__all__ = [
    "DEFAULT_PROFILE_WINDOW_SEC",
    "RollingLatencyProfile",
]
