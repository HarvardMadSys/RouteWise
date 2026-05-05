"""Empirical latency distribution backed by raw samples."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EmpiricalDistribution:
    """Distribution interface backed by observed samples."""

    samples: np.ndarray
    label: str = ""
    _sorted: np.ndarray = field(init=False, repr=False)

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
        return rng.choice(self.samples, size=size, replace=True)

    def quantile(self, q: float) -> float:
        """Return empirical quantile for q in (0, 1)."""
        if not 0.0 < q < 1.0:
            raise ValueError(f"quantile q must be in (0, 1), got {q}")
        return float(np.percentile(self._sorted, q * 100.0))

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
        return float(np.searchsorted(self._sorted, value, side="right") / self._sorted.size)


__all__ = ["EmpiricalDistribution"]
