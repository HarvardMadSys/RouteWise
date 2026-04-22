"""Tiered synthetic providers: S_A (pay-per-token), S_Q (quota), S_C (concurrency).

Extends SyntheticProvider with tier-specific capacity state:
  - S_A: unlimited, marginal cost = price * tokens
  - S_Q: free until Q requests consumed per window, then unavailable
  - S_C: free while active_requests < C, otherwise request must queue or spill

The provider's `sample_request(...)` stays unchanged; capacity tracking is done
through attached state objects updated by the strategy runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..providers import LogNormal, SyntheticProvider


class ProviderTier(str, Enum):
    """Pricing/capacity regime for a provider."""

    S_A = "api"           # pay-per-token, unlimited capacity
    S_Q = "quota"         # free up to Q requests per window
    S_C = "concurrency"   # free while active_requests <= C


# ---------------------------------------------------------------------------
# Capacity state
# ---------------------------------------------------------------------------


@dataclass
class QuotaState:
    """Mutable quota usage tracker for an S_Q provider.

    Parameters
    ----------
    size:
        Maximum number of requests per window.
    window_sec:
        Window length in simulated seconds. Typical: 86400 (1 day).
    used:
        Requests consumed in the current window.
    window_start:
        Simulated timestamp at which the current window began.
    """

    size: int
    window_sec: float = 86400.0
    used: int = 0
    window_start: float = 0.0

    def roll_window(self, now: float) -> None:
        """Reset the counter if the current window has expired."""
        if now - self.window_start >= self.window_sec:
            self.window_start = now
            self.used = 0

    def fraction_used(self, now: float) -> float:
        self.roll_window(now)
        if self.size <= 0:
            return 1.0
        return min(self.used / self.size, 1.0)

    def can_admit(self, now: float) -> bool:
        self.roll_window(now)
        return self.used < self.size

    def charge(self, now: float) -> None:
        self.roll_window(now)
        self.used += 1


@dataclass
class ConcurrencyState:
    """Mutable concurrency tracker for an S_C provider.

    Simplified model: each admitted request occupies one slot for its full
    sampled service time. No explicit queue — if capacity is saturated the
    strategy must decide to spill to another tier.

    Parameters
    ----------
    limit:
        Maximum concurrent requests C.
    active:
        List of (end_time, request_id) tuples for currently active requests.
    """

    limit: int
    active: list[tuple[float, int]] = field(default_factory=list)

    def _sweep(self, now: float) -> None:
        """Remove completed requests from the active list."""
        self.active = [(t_end, rid) for (t_end, rid) in self.active if t_end > now]

    def utilization(self, now: float) -> float:
        self._sweep(now)
        if self.limit <= 0:
            return 1.0
        return min(len(self.active) / self.limit, 1.0)

    def can_admit(self, now: float) -> bool:
        self._sweep(now)
        return len(self.active) < self.limit

    def admit(self, request_id: int, now: float, service_time_sec: float) -> None:
        self._sweep(now)
        self.active.append((now + service_time_sec, request_id))


# ---------------------------------------------------------------------------
# TieredProvider
# ---------------------------------------------------------------------------


@dataclass
class TieredProvider(SyntheticProvider):
    """SyntheticProvider with a tier label and attached capacity state.

    For S_A: `quota` and `concurrency` are None. Marginal cost is always
    `cost_per_token * tokens`.

    For S_Q: `quota` is populated. Marginal cost is 0 while `quota.can_admit()`;
    the strategy is responsible for honoring the capacity bound.

    For S_C: `concurrency` is populated. Marginal cost is 0 while
    `concurrency.can_admit()`. `service_time_dist` controls how long each
    admitted request occupies a slot (separate from TTFT).

    Optional time-shift support: if `shift_time` and `ttft_dist_after` are
    set, the provider's TTFT distribution changes at `shift_time`. This is
    used in stress tests to simulate mid-run provider degradation.
    """

    tier: ProviderTier = ProviderTier.S_A
    quota: QuotaState | None = None
    concurrency: ConcurrencyState | None = None

    # For S_C only: service time distribution in ms. If None, defaults to
    # TTFT distribution (i.e. fast release).
    service_time_dist: LogNormal | None = None

    # Optional time-shift for drift / degradation tests.
    shift_time: float | None = None
    ttft_dist_after: LogNormal | None = None

    # ------------------------------------------------------------------ #
    # Override sample_ttft / true_p50 to support shifts                    #
    # ------------------------------------------------------------------ #

    def _active_ttft_dist(self, current_time: float) -> LogNormal:
        if (
            self.shift_time is not None
            and self.ttft_dist_after is not None
            and current_time >= self.shift_time
        ):
            return self.ttft_dist_after
        return self.ttft_dist

    def sample_ttft(self, rng, current_time: float = 0.0) -> float:
        dist = self._active_ttft_dist(current_time)
        return float(dist.sample(rng)[0])

    def true_p50_ms(self, current_time: float = 0.0) -> float:
        return self._active_ttft_dist(current_time).p50()

    def true_p99_ms(self, current_time: float = 0.0) -> float:
        return self._active_ttft_dist(current_time).p99()

    # ------------------------------------------------------------------ #
    # Convenience predicates                                               #
    # ------------------------------------------------------------------ #

    def is_available(self, now: float) -> bool:
        """Whether this provider can admit a new request at `now`.

        S_A: always True.
        S_Q: quota window has budget left.
        S_C: concurrency slots available.
        """
        if self.tier == ProviderTier.S_A:
            return True
        if self.tier == ProviderTier.S_Q:
            assert self.quota is not None
            return self.quota.can_admit(now)
        if self.tier == ProviderTier.S_C:
            assert self.concurrency is not None
            return self.concurrency.can_admit(now)
        return False

    def marginal_cost(self, total_tokens: int, now: float) -> float:
        """Real USD cost incurred by routing one request to this provider.

        S_A: price-per-token billed. S_Q/S_C: free while capacity holds;
        if the strategy routes here while capacity is exhausted, we still
        return 0 (unavailable dispatch is the caller's bug). The shadow-price
        layer is what prevents over-consumption.
        """
        if self.tier == ProviderTier.S_A:
            return self.cost_per_token * total_tokens
        return 0.0

    # ------------------------------------------------------------------ #
    # State mutation helpers (called by the strategy runner)              #
    # ------------------------------------------------------------------ #

    def account_request(
        self, request_id: int, now: float, service_time_sec: float
    ) -> None:
        """Record a request as admitted to this provider's capacity."""
        if self.tier == ProviderTier.S_Q:
            assert self.quota is not None
            self.quota.charge(now)
        elif self.tier == ProviderTier.S_C:
            assert self.concurrency is not None
            self.concurrency.admit(request_id, now, service_time_sec)

    def reset_state(self) -> None:
        """Clear capacity state so a scenario can be re-run."""
        if self.quota is not None:
            self.quota.used = 0
            self.quota.window_start = 0.0
        if self.concurrency is not None:
            self.concurrency.active = []
