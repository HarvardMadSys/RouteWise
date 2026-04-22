"""Shared statistical distributions for the synthetic simulator."""

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


__all__ = ["LogNormal"]
