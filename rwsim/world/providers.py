"""Unified provider definitions for the RouteWise simulator."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

from rwsim.world.capacity import (
    ConcurrencyState,
    MultiWindowQuotaState,
    ProviderTier,
    QuotaState,
    WeightedConcurrencyState,
)
from rwsim.world.distributions import LatencyDistribution, LogNormal

if TYPE_CHECKING:
    import numpy as np


@dataclass
class Provider:
    """Unified provider covering API, quota, concurrency, and drift.

    Distribution fields use :class:`~rwsim.world.distributions.LatencyDistribution`
    (a Protocol) rather than ``LogNormal`` directly. The legacy YAML loader
    still produces ``LogNormal`` instances; the eval-grid factory may
    inject ``Uniform`` or ``Normal``. Routing code must access these
    fields only through the Protocol surface (``sample``, ``p50``,
    ``p95``, ``p99``, ``quantile``, ``mean``, ``std``, ``cdf``) — never
    via ``mu`` / ``sigma`` or other LogNormal-specific attributes.
    """

    name: str
    cost_per_token: float
    ttft_dist: LatencyDistribution
    tps_dist: LatencyDistribution
    tier: ProviderTier = ProviderTier.S_A
    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None
    cached_input_cost_per_token: float | None = None
    quota: QuotaState | MultiWindowQuotaState | None = None
    concurrency: ConcurrencyState | WeightedConcurrencyState | None = None
    service_time_dist: LatencyDistribution | None = None
    shift_time: float | None = None
    ttft_dist_after: LatencyDistribution | None = None
    ttft_shift_schedule: tuple[tuple[float, LatencyDistribution], ...] | None = None

    def __post_init__(self) -> None:
        if self.ttft_shift_schedule is not None:
            if self.shift_time is not None or self.ttft_dist_after is not None:
                raise ValueError(
                    "Provider accepts either ttft_shift_schedule or the legacy "
                    "shift_time/ttft_dist_after pair, not both."
                )
            times = [time_sec for time_sec, _ in self.ttft_shift_schedule]
            if any(later <= earlier for earlier, later in pairwise(times)):
                raise ValueError("ttft_shift_schedule must be strictly time-sorted.")

    def _active_ttft_dist(self, current_time: float) -> LatencyDistribution:
        """Return the TTFT distribution active at the given simulated time."""
        if self.ttft_shift_schedule:
            index = bisect_right(
                self.ttft_shift_schedule,
                current_time,
                key=lambda entry: entry[0],
            )
            if index > 0:
                return self.ttft_shift_schedule[index - 1][1]
            return self.ttft_dist
        if (
            self.shift_time is not None
            and self.ttft_dist_after is not None
            and current_time >= self.shift_time
        ):
            return self.ttft_dist_after
        return self.ttft_dist

    def sample_ttft(self, rng: np.random.Generator, current_time: float = 0.0) -> float:
        """Sample TTFT in milliseconds."""
        return float(self._active_ttft_dist(current_time).sample(rng)[0])

    def sample_request(
        self,
        output_tokens: int,
        rng: np.random.Generator,
        current_time: float = 0.0,
    ) -> tuple[float, float]:
        """Return ``(ttft_ms, e2e_ms)`` for a request."""
        ttft_ms = self.sample_ttft(rng, current_time)
        tps = max(float(self.tps_dist.sample(rng)[0]), 1.0)
        generation_ms = (output_tokens / tps) * 1000.0
        return ttft_ms, ttft_ms + generation_ms

    def true_p50_ms(self, current_time: float = 0.0) -> float:
        """Analytical P50 TTFT in ms."""
        return self._active_ttft_dist(current_time).p50()

    def true_p99_ms(self, current_time: float = 0.0) -> float:
        """Analytical P99 TTFT in ms."""
        return self._active_ttft_dist(current_time).p99()

    def true_mean_ms(self, current_time: float = 0.0) -> float:
        """Analytical mean TTFT in ms."""
        return self._active_ttft_dist(current_time).mean()

    @property
    def effective_input_cost_per_token(self) -> float:
        """Input-token price, falling back to the legacy blended token price."""
        if self.input_cost_per_token is not None:
            return self.input_cost_per_token
        return self.cost_per_token

    @property
    def effective_output_cost_per_token(self) -> float:
        """Output-token price, falling back to the legacy blended token price."""
        if self.output_cost_per_token is not None:
            return self.output_cost_per_token
        return self.cost_per_token

    def token_cost(
        self,
        *,
        request_tokens: int | float,
        response_tokens: int | float,
        total_tokens: int | float | None = None,
        cached_input_tokens: int | float = 0,
    ) -> float:
        """Cost in USD for token counts, preserving legacy blended pricing when needed."""
        cached_input_tokens = max(float(cached_input_tokens), 0.0)
        if (
            cached_input_tokens > 0.0
            and self.input_cost_per_token is not None
            and self.output_cost_per_token is not None
            and self.cached_input_cost_per_token is not None
        ):
            request_tokens = float(request_tokens)
            response_tokens = float(response_tokens)
            cached_input_tokens = min(cached_input_tokens, request_tokens)
            uncached_input_tokens = max(request_tokens - cached_input_tokens, 0.0)
            return (
                self.input_cost_per_token * uncached_input_tokens
                + self.cached_input_cost_per_token * cached_input_tokens
                + self.output_cost_per_token * response_tokens
            )
        if self.input_cost_per_token is None and self.output_cost_per_token is None:
            if total_tokens is None:
                total_tokens = float(request_tokens) + float(response_tokens)
            return self.cost_per_token * float(total_tokens)
        return (
            self.effective_input_cost_per_token * float(request_tokens)
            + self.effective_output_cost_per_token * float(response_tokens)
        )

    def cost_per_request(self, total_tokens: int) -> float:
        """Cost in USD for a request with only total-token count available."""
        return self.cost_per_token * total_tokens

    def is_available(self, now: float) -> bool:
        """Whether this provider can admit a new request at ``now``."""
        if self.tier == ProviderTier.S_A:
            return True
        if self.tier == ProviderTier.S_Q:
            assert self.quota is not None
            return self.quota.can_admit(now)
        if self.tier == ProviderTier.S_C:
            assert self.concurrency is not None
            if isinstance(self.concurrency, WeightedConcurrencyState):
                return self.concurrency.can_admit_interval(now, now)
            return self.concurrency.can_admit(now)
        return False

    def marginal_cost(self, total_tokens: int, now: float) -> float:
        """Real USD cost incurred by routing one request to this provider."""
        del now
        if self.tier == ProviderTier.S_A:
            return self.cost_per_token * total_tokens
        return 0.0

    def marginal_cost_for_request(
        self,
        request,
        now: float,
        *,
        cached_input_tokens: int | float = 0,
    ) -> float:
        """Real USD cost for routing a concrete request to this provider."""
        del now
        if self.tier != ProviderTier.S_A:
            return 0.0

        request_tokens = int(getattr(request, "request_tokens", 0) or 0)
        response_tokens = getattr(request, "response_tokens", None)
        if response_tokens is None:
            response_tokens = getattr(request, "estimated_response_tokens", None)
        total_tokens = getattr(request, "total_tokens", None)
        if response_tokens is None:
            response_tokens = max(int(total_tokens or 0) - request_tokens, 0)
        return self.token_cost(
            request_tokens=request_tokens,
            response_tokens=float(response_tokens),
            total_tokens=total_tokens,
            cached_input_tokens=cached_input_tokens,
        )

    def marginal_prefill_cost_for_request(
        self,
        request,
        now: float,
        *,
        cached_input_tokens: int | float = 0,
    ) -> float:
        """Real USD cost for a canceled request before its first output token."""
        del now
        if self.tier != ProviderTier.S_A:
            return 0.0

        request_tokens = int(getattr(request, "request_tokens", 0) or 0)
        return self.token_cost(
            request_tokens=request_tokens,
            response_tokens=0.0,
            total_tokens=None,
            cached_input_tokens=cached_input_tokens,
        )

    def account_request(
        self,
        request_id: int,
        now: float,
        service_time_sec: float,
    ) -> None:
        """Record a request as admitted to this provider's capacity."""
        if self.tier == ProviderTier.S_Q:
            assert self.quota is not None
            self.quota.charge(now)
        elif self.tier == ProviderTier.S_C:
            assert self.concurrency is not None
            if isinstance(self.concurrency, WeightedConcurrencyState):
                self.concurrency.admit_interval(
                    now,
                    service_time_sec,
                    request_id=request_id,
                )
            else:
                self.concurrency.admit(request_id, now, service_time_sec)

    def cancel_request(self, request_id: int, cancel_time: float) -> bool:
        """Cancel an admitted request's capacity occupancy at ``cancel_time``."""
        if self.tier != ProviderTier.S_C or self.concurrency is None:
            return False
        return self.concurrency.truncate(request_id, cancel_time)

    def reset_state(self, *, quota_window_start: float = 0.0) -> None:
        """Clear capacity state so a scenario can be re-run."""
        if self.quota is not None:
            self.quota.reset(window_start=quota_window_start)
        if self.concurrency is not None:
            self.concurrency.reset()


@dataclass
class ShiftingProvider(Provider):
    """Compatibility subclass that requires a time-shift definition."""

    def _active_dist(self, current_time: float) -> LatencyDistribution:
        """Legacy helper used by latency-only code paths."""
        return self._active_ttft_dist(current_time)

    def __post_init__(self) -> None:
        if self.shift_time is None or self.ttft_dist_after is None:
            raise ValueError("ShiftingProvider requires shift_time and ttft_dist_after.")


@dataclass
class SyntheticProvider(Provider):
    """Thin compatibility subclass for latency-only scenarios."""


@dataclass
class TieredProvider(Provider):
    """Thin compatibility subclass for tiered scenarios."""


__all__ = [
    "ConcurrencyState",
    "LogNormal",
    "Provider",
    "ProviderTier",
    "QuotaState",
    "ShiftingProvider",
    "SyntheticProvider",
    "TieredProvider",
    "WeightedConcurrencyState",
]
