"""Synthetic provider implementations with known latency distributions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class LogNormal:
    """Log-normal distribution for latency sampling.

    P50  = exp(mu)
    P99  = exp(mu + 2.326 * sigma)
    """

    mu: float    # log-mean
    sigma: float  # log-std

    def sample(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        return rng.lognormal(self.mu, self.sigma, size)

    def p50(self) -> float:
        return math.exp(self.mu)

    def p99(self) -> float:
        return math.exp(self.mu + 2.326 * self.sigma)


@dataclass
class SyntheticProvider:
    """Mock provider that samples TTFT and generation speed from known distributions.

    All latency values are in milliseconds.
    """

    name: str
    cost_per_token: float    # USD per token
    ttft_dist: LogNormal     # TTFT distribution in ms
    tps_dist: LogNormal      # Tokens-per-second distribution

    # ------------------------------------------------------------------ #
    # Sampling                                                             #
    # ------------------------------------------------------------------ #

    def sample_ttft(self, rng: np.random.Generator, current_time: float = 0.0) -> float:
        """Sample TTFT in milliseconds. current_time is ignored for base class."""
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

    # ------------------------------------------------------------------ #
    # True (analytical) statistics                                         #
    # ------------------------------------------------------------------ #

    def true_p50_ms(self, current_time: float = 0.0) -> float:
        """Analytical P50 TTFT in ms."""
        return self.ttft_dist.p50()

    def true_p99_ms(self, current_time: float = 0.0) -> float:
        """Analytical P99 TTFT in ms."""
        return self.ttft_dist.p99()

    def cost_per_request(self, total_tokens: int) -> float:
        """Cost in USD for a request with the given token count."""
        return self.cost_per_token * total_tokens


@dataclass
class ShiftingProvider(SyntheticProvider):
    """Provider whose TTFT distribution shifts at a given simulated time.

    Before shift_time: uses ttft_dist (inherited from SyntheticProvider).
    At or after shift_time: uses ttft_dist_after.
    """

    shift_time: float        # Simulated seconds when the shift occurs
    ttft_dist_after: LogNormal  # TTFT distribution after the shift

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
