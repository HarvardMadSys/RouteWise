"""Unified provider definitions for the synthetic simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .capacity import ConcurrencyState, ProviderTier, QuotaState
from .distributions import LogNormal


@dataclass
class Provider:
    """Unified synthetic provider covering S_A, S_Q, S_C, and shifting."""

    name: str
    cost_per_token: float
    ttft_dist: LogNormal
    tps_dist: LogNormal
    tier: ProviderTier = ProviderTier.S_A
    quota: QuotaState | None = None
    concurrency: ConcurrencyState | None = None
    service_time_dist: LogNormal | None = None
    shift_time: float | None = None
    ttft_dist_after: LogNormal | None = None

    def _active_ttft_dist(self, current_time: float) -> LogNormal:
        """Return the TTFT distribution active at the given simulated time."""
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
        """Return (ttft_ms, e2e_ms) for a request."""
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

    def cost_per_request(self, total_tokens: int) -> float:
        """Cost in USD for a request with the given token count."""
        return self.cost_per_token * total_tokens

    def is_available(self, now: float) -> bool:
        """Whether this provider can admit a new request at `now`."""
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
        """Real USD cost incurred by routing one request to this provider."""
        del now
        if self.tier == ProviderTier.S_A:
            return self.cost_per_token * total_tokens
        return 0.0

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
            self.concurrency.admit(request_id, now, service_time_sec)

    def reset_state(self) -> None:
        """Clear capacity state so a scenario can be re-run."""
        if self.quota is not None:
            self.quota.used = 0
            self.quota.window_start = 0.0
        if self.concurrency is not None:
            self.concurrency.active = []


@dataclass
class ShiftingProvider(Provider):
    """Compatibility subclass that requires a time-shift definition."""

    def _active_dist(self, current_time: float) -> LogNormal:
        """Legacy helper used by latency-only code paths."""
        return self._active_ttft_dist(current_time)

    def __post_init__(self) -> None:
        if self.shift_time is None or self.ttft_dist_after is None:
            raise ValueError(
                "ShiftingProvider requires shift_time and ttft_dist_after."
            )


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
]
