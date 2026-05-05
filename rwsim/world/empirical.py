"""Empirical latency distribution backed by raw samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EmpiricalDistribution:
    """Distribution interface backed by observed samples."""

    samples: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.samples, dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("EmpiricalDistribution requires a non-empty 1D sample array.")
        if np.any(values < 0):
            raise ValueError("EmpiricalDistribution samples must be non-negative.")
        object.__setattr__(self, "samples", values)

    def sample(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        """Draw samples with replacement."""
        return rng.choice(self.samples, size=size, replace=True)

    def quantile(self, q: float) -> float:
        """Return empirical quantile for q in (0, 1)."""
        if not 0.0 < q < 1.0:
            raise ValueError(f"quantile q must be in (0, 1), got {q}")
        return float(np.percentile(self.samples, q * 100.0))

    def p50(self) -> float:
        return self.quantile(0.50)

    def p95(self) -> float:
        return self.quantile(0.95)

    def p99(self) -> float:
        return self.quantile(0.99)

    def mean(self) -> float:
        return float(np.mean(self.samples))

    def std(self) -> float:
        return float(np.std(self.samples))

    def cdf(self, value: float) -> float:
        return float(np.mean(self.samples <= value))


__all__ = ["EmpiricalDistribution"]
