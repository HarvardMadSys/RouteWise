"""Shared statistical distributions for the synthetic simulator."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class LogNormal:
    """Log-normal distribution for latency sampling.

    P50 = exp(mu). P99 = exp(mu + 2.326 * sigma).
    """

    mu: float
    sigma: float

    def sample(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        """Draw samples from the distribution."""
        return rng.lognormal(self.mu, self.sigma, size)

    def p50(self) -> float:
        """Return the analytical P50."""
        return math.exp(self.mu)

    def p99(self) -> float:
        """Return the analytical P99."""
        return math.exp(self.mu + 2.326 * self.sigma)

    def cdf(self, value: float) -> float:
        """Return the CDF evaluated at ``value``."""
        if value <= 0.0:
            return 0.0
        z = (math.log(value) - self.mu) / (self.sigma * math.sqrt(2.0))
        return 0.5 * (1.0 + math.erf(z))


__all__ = ["LogNormal"]
