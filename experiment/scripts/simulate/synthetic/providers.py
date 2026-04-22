"""Synthetic provider implementations with known latency distributions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._core.distributions import LogNormal


@dataclass
class SyntheticProvider:
    """Mock provider that samples TTFT and generation speed from known distributions.

    All latency values are in milliseconds.
    """

    name: str
    cost_per_token: float    # USD per token
    ttft_dist: LogNormal     # TTFT distribution in ms
    tps_dist: LogNormal      # Tokens-per-second distribution

    def sample_ttft(self, rng: np.random.Generator, current_time: float = 0.0) -> float:
        """Sample TTFT in milliseconds. current_time is ignored for base class."""
        del current_time
        return float(self.ttft_dist.sample(rng)[0])

    def sample_request(
        self,
        output_tokens: int,
        rng: np.random.Generator,
        current_time: float = 0.0,
    ) -> tuple[float, float]:
        """Return (ttft_ms, e2e_ms) for a request with the given output token count."""
        ttft_ms = self.sample_ttft(rng, current_time)
        tps = max(float(self.tps_dist.sample(rng)[0]), 1.0)
        generation_ms = (output_tokens / tps) * 1000.0
        return ttft_ms, ttft_ms + generation_ms

    def true_p50_ms(self, current_time: float = 0.0) -> float:
        """Analytical P50 TTFT in ms."""
        del current_time
        return self.ttft_dist.p50()

    def true_p99_ms(self, current_time: float = 0.0) -> float:
        """Analytical P99 TTFT in ms."""
        del current_time
        return self.ttft_dist.p99()

    def cost_per_request(self, total_tokens: int) -> float:
        """Cost in USD for a request with the given token count."""
        return self.cost_per_token * total_tokens


@dataclass
class ShiftingProvider(SyntheticProvider):
    """Provider whose TTFT distribution shifts at a given simulated time."""

    shift_time: float
    ttft_dist_after: LogNormal

    def _active_dist(self, current_time: float) -> LogNormal:
        return self.ttft_dist_after if current_time >= self.shift_time else self.ttft_dist

    def sample_ttft(self, rng: np.random.Generator, current_time: float = 0.0) -> float:
        return float(self._active_dist(current_time).sample(rng)[0])

    def sample_request(
        self,
        output_tokens: int,
        rng: np.random.Generator,
        current_time: float = 0.0,
    ) -> tuple[float, float]:
        ttft_ms = self.sample_ttft(rng, current_time)
        tps = max(float(self.tps_dist.sample(rng)[0]), 1.0)
        generation_ms = (output_tokens / tps) * 1000.0
        return ttft_ms, ttft_ms + generation_ms

    def true_p50_ms(self, current_time: float = 0.0) -> float:
        return self._active_dist(current_time).p50()

    def true_p99_ms(self, current_time: float = 0.0) -> float:
        return self._active_dist(current_time).p99()
