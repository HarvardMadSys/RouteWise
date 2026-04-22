"""Tiered synthetic providers: S_A (pay-per-token), S_Q (quota), S_C (concurrency).

Extends SyntheticProvider with tier-specific capacity state:
  - S_A: unlimited, marginal cost = price * tokens
  - S_Q: free until Q requests consumed per window, then unavailable
  - S_C: free while active_requests < C, otherwise request must queue or spill

The provider's `sample_request(...)` stays unchanged; capacity tracking is done
through attached state objects updated by the strategy runner.
"""

from __future__ import annotations

from dataclasses import dataclass

from .._core.capacity import ConcurrencyState, ProviderTier, QuotaState
from ..providers import LogNormal, SyntheticProvider


@dataclass
class TieredProvider(SyntheticProvider):
    """SyntheticProvider with a tier label and attached capacity state."""

    tier: ProviderTier = ProviderTier.S_A
    quota: QuotaState | None = None
    concurrency: ConcurrencyState | None = None
    service_time_dist: LogNormal | None = None
    shift_time: float | None = None
    ttft_dist_after: LogNormal | None = None

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
