"""Empirical latency distribution backed by raw samples."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EmpiricalDistribution:
    """Distribution interface backed by observed samples.

    Moments and the sorted view are computed once in ``__post_init__`` so
    ``mean``/``std``/``quantile`` are O(1) per call. The sampler uses
    ``rng.integers + fancy indexing`` because per-call overhead is lower
    than ``rng.choice(replace=True)`` on the size=1 hot path.
    """

    samples: np.ndarray
    label: str = ""
    _sorted: np.ndarray = field(init=False, repr=False)
    _n: int = field(init=False, repr=False)
    _mean: float = field(init=False, repr=False)
    _std: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        values = np.asarray(self.samples, dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("EmpiricalDistribution requires a non-empty 1D sample array.")
        if not np.all(np.isfinite(values)):
            raise ValueError("EmpiricalDistribution samples must be finite.")
        if np.any(values < 0):
            raise ValueError("EmpiricalDistribution samples must be non-negative.")
        object.__setattr__(self, "samples", values)
        object.__setattr__(self, "_sorted", np.sort(values))
        object.__setattr__(self, "_n", int(values.size))
        object.__setattr__(self, "_mean", float(np.mean(values)))
        object.__setattr__(self, "_std", float(np.std(values)))

    @classmethod
    def from_npz(
        cls,
        npz_path: str | Path,
        provider_name: str,
        *,
        label: str | None = None,
    ) -> EmpiricalDistribution:
        """Load one provider's empirical distribution from a `.npz` artifact."""
        path = Path(npz_path)
        with np.load(path) as data:
            if provider_name not in data:
                available = ", ".join(sorted(data.files))
                raise KeyError(
                    f"provider {provider_name!r} not found in {path}; "
                    f"available providers: {available}"
                )
            samples = data[provider_name].copy()
        return cls(samples=samples, label=label or f"{path.stem}/{provider_name}")

    @classmethod
    def pooled_from_npz(
        cls,
        npz_path: str | Path,
        provider_names: tuple[str, ...] | list[str],
        *,
        label: str = "pooled",
    ) -> EmpiricalDistribution:
        """Load and concatenate multiple provider distributions from a `.npz` artifact."""
        names = tuple(provider_names)
        if not names:
            raise ValueError("pooled_from_npz requires at least one provider name.")

        path = Path(npz_path)
        with np.load(path) as data:
            missing = [name for name in names if name not in data]
            if missing:
                available = ", ".join(sorted(data.files))
                raise KeyError(
                    f"providers {missing!r} not found in {path}; available providers: {available}"
                )
            arrays = [data[name].copy() for name in names]
        return cls(samples=np.concatenate(arrays), label=f"{path.stem}/{label}")

    def sample(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        """Draw samples with replacement."""
        idx = rng.integers(0, self._n, size=size)
        return self.samples[idx]

    def quantile(self, q: float) -> float:
        """Return empirical quantile for q in (0, 1).

        Linear interpolation between adjacent sorted samples — bit-exact with
        ``np.percentile(..., method='linear')`` (NumPy's default), but skips
        the dispatch overhead that dominates per-call cost on the hot path.
        """
        if not 0.0 < q < 1.0:
            raise ValueError(f"quantile q must be in (0, 1), got {q}")
        sorted_samples = self._sorted
        pos = q * (self._n - 1)
        lo = int(pos)
        hi = lo + 1
        if hi >= self._n:
            return float(sorted_samples[self._n - 1])
        frac = pos - lo
        return float(sorted_samples[lo] * (1.0 - frac) + sorted_samples[hi] * frac)

    def p50(self) -> float:
        return self.quantile(0.50)

    def p95(self) -> float:
        return self.quantile(0.95)

    def p99(self) -> float:
        return self.quantile(0.99)

    def mean(self) -> float:
        return self._mean

    def std(self) -> float:
        return self._std

    def cdf(self, value: float) -> float:
        return float(np.searchsorted(self._sorted, value, side="right") / self._n)


__all__ = ["EmpiricalDistribution"]
