"""Shared statistical distributions for the synthetic simulator.

Provides three latency distribution families used across the simulator grid:

- ``Uniform``: bounded latency window, no tail. Easiest to reason about.
- ``Normal``: symmetric around the mean, light tail. Samples are clipped at
  ``MIN_LATENCY_MS`` so latency never becomes non-positive.
- ``LogNormal``: heavy-tailed; intended to approximate real provider tails.
  ``HeavyTail`` is exposed as an alias to make scenario configs read naturally.

Each class implements the same minimal interface so providers and routers
can stay distribution-agnostic::

    sample(rng, size) -> np.ndarray
    p50() -> float
    p95() -> float
    p99() -> float
    quantile(q) -> float
    mean() -> float
    std() -> float
    cdf(value) -> float

The percentile API (``p95``, ``quantile``) is required by tiered SLO
filters that previously read ``LogNormal.mu/sigma`` directly. Routing code
should call ``provider.ttft_dist.p95()`` rather than reaching into
distribution-specific attributes.

The simulator generally starts with ``Uniform`` for sanity, escalates to
``Normal``, then uses ``LogNormal`` for heavy-tail behaviour. Each step adds
exactly one variable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.special import erfinv as _erfinv

# Latency samples below this floor are clipped. Required for ``Normal`` since
# unclipped Gaussians can produce non-positive values, which break downstream
# code that assumes positive TTFT/service times. Kept consistent with the
# ``output_tokens >= 1`` convention in ``rwsim/world/workload.py``.
MIN_LATENCY_MS: float = 1.0


# Exact standard-normal z-scores, computed once. Using these in the closed-form
# ``p95`` / ``p99`` accessors keeps them numerically identical to
# ``quantile(0.95)`` / ``quantile(0.99)``. Hard-coding 1.645 / 2.326 introduced
# a ~1e-5 relative gap that broke the equivalence contract between the two APIs.
_Z_P95: float = float(math.sqrt(2.0) * _erfinv(0.90))
_Z_P99: float = float(math.sqrt(2.0) * _erfinv(0.98))


@dataclass
class Uniform:
    """Uniform distribution on ``[low, high]`` (in milliseconds).

    The simplest latency family: bounded support, no tail. Used as the
    Stage-1 sanity baseline so router behaviour can be reasoned about
    analytically.
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        if self.high <= self.low:
            raise ValueError(
                f"Uniform requires high > low (got low={self.low}, high={self.high})"
            )
        if self.low < 0.0:
            raise ValueError(f"Uniform low must be non-negative (got {self.low})")

    def sample(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        """Draw samples from the distribution."""
        return rng.uniform(self.low, self.high, size)

    def p50(self) -> float:
        """Return the analytical P50."""
        return self.quantile(0.50)

    def p95(self) -> float:
        """Return the analytical P95."""
        return self.quantile(0.95)

    def p99(self) -> float:
        """Return the analytical P99."""
        return self.quantile(0.99)

    def quantile(self, q: float) -> float:
        """Return the analytical quantile for ``q`` in ``(0, 1)``.

        Endpoints (``q=0`` / ``q=1``) are rejected so all three families
        share the same domain — for Normal/LogNormal the endpoints map to
        ``±inf`` / ``0`` respectively and propagating those through routing
        code tends to surface as silent bugs. Use ``Uniform.low`` /
        ``Uniform.high`` directly if a true endpoint is wanted.
        """
        if not 0.0 < q < 1.0:
            raise ValueError(f"quantile q must be in (0, 1), got {q}")
        return self.low + q * (self.high - self.low)

    def mean(self) -> float:
        """Return the analytical mean."""
        return 0.5 * (self.low + self.high)

    def std(self) -> float:
        """Return the analytical standard deviation."""
        return (self.high - self.low) / math.sqrt(12.0)

    def cdf(self, value: float) -> float:
        """Return the CDF evaluated at ``value``."""
        if value <= self.low:
            return 0.0
        if value >= self.high:
            return 1.0
        return (value - self.low) / (self.high - self.low)


@dataclass
class Normal:
    """Truncated-at-floor normal distribution for latency sampling.

    Samples are clipped at :data:`MIN_LATENCY_MS` so latency stays positive.
    The analytical moments (``p50``, ``mean``, ``std``) ignore the clipping
    because in regimes where mean - 3*sigma > MIN_LATENCY_MS the truncation
    is negligible. Scenario authors should pick parameters in that regime
    or accept that the analytical moments are slightly off the empirical ones.
    """

    mean_ms: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0.0:
            raise ValueError(f"Normal sigma must be positive (got {self.sigma})")

    def sample(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        """Draw samples from the distribution, clipped at MIN_LATENCY_MS."""
        raw = rng.normal(self.mean_ms, self.sigma, size)
        return np.clip(raw, MIN_LATENCY_MS, None)

    def p50(self) -> float:
        """Return the analytical P50 (median = mean for Gaussian)."""
        return self.mean_ms

    def p95(self) -> float:
        """Return the analytical P95."""
        return self.mean_ms + _Z_P95 * self.sigma

    def p99(self) -> float:
        """Return the analytical P99."""
        return self.mean_ms + _Z_P99 * self.sigma

    def quantile(self, q: float) -> float:
        """Return the analytical quantile for ``q`` in ``(0, 1)``.

        Endpoints are rejected because ``erfinv(-1)/erfinv(1)`` diverge to
        ``±inf``; routing code that consumes the result (e.g. SLO filters)
        is not robust against infinities. Use the named accessors
        (``p50``/``p95``/``p99``) for the common quantiles.
        """
        if not 0.0 < q < 1.0:
            raise ValueError(f"quantile q must be in (0, 1), got {q}")
        z = float(math.sqrt(2.0) * _erfinv(2.0 * q - 1.0))
        return self.mean_ms + z * self.sigma

    def mean(self) -> float:
        """Return the analytical mean."""
        return self.mean_ms

    def std(self) -> float:
        """Return the analytical standard deviation."""
        return self.sigma

    def cdf(self, value: float) -> float:
        """Return the CDF evaluated at ``value`` (ignoring left clipping)."""
        z = (value - self.mean_ms) / (self.sigma * math.sqrt(2.0))
        return 0.5 * (1.0 + math.erf(z))


@dataclass
class LogNormal:
    """Log-normal distribution for latency sampling.

    P50 = exp(mu). P99 = exp(mu + 2.326 * sigma). Heavy-tailed family used to
    approximate real provider TTFT distributions.
    """

    mu: float
    sigma: float

    def sample(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        """Draw samples from the distribution."""
        return rng.lognormal(self.mu, self.sigma, size)

    def p50(self) -> float:
        """Return the analytical P50."""
        return math.exp(self.mu)

    def p95(self) -> float:
        """Return the analytical P95."""
        return math.exp(self.mu + _Z_P95 * self.sigma)

    def p99(self) -> float:
        """Return the analytical P99."""
        return math.exp(self.mu + _Z_P99 * self.sigma)

    def quantile(self, q: float) -> float:
        """Return the analytical quantile for ``q`` in ``(0, 1)``.

        Endpoints are rejected. ``q=0`` would map to ``0`` and ``q=1`` to
        ``+inf``; both make routing code behave unpredictably. Use the
        named accessors for the common quantiles.
        """
        if not 0.0 < q < 1.0:
            raise ValueError(f"quantile q must be in (0, 1), got {q}")
        z = float(math.sqrt(2.0) * _erfinv(2.0 * q - 1.0))
        return math.exp(self.mu + z * self.sigma)

    def mean(self) -> float:
        """Return the analytical mean."""
        return math.exp(self.mu + 0.5 * self.sigma * self.sigma)

    def std(self) -> float:
        """Return the analytical standard deviation."""
        var = (math.exp(self.sigma * self.sigma) - 1.0) * math.exp(
            2.0 * self.mu + self.sigma * self.sigma
        )
        return math.sqrt(var)

    def cdf(self, value: float) -> float:
        """Return the CDF evaluated at ``value``."""
        if value <= 0.0:
            return 0.0
        z = (math.log(value) - self.mu) / (self.sigma * math.sqrt(2.0))
        return 0.5 * (1.0 + math.erf(z))


@dataclass(frozen=True)
class ScaledDistribution:
    """Multiplicative wrapper around any latency distribution.

    Scales every sample and quantile of ``base`` by ``scale`` while
    preserving the distribution's shape. Works uniformly across the
    parametric families and :class:`~rwsim.world.empirical.EmpiricalDistribution`,
    which makes it the building block for time-varying provider latency
    (e.g. periodic degradation schedules).
    """

    base: LatencyDistribution
    scale: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError(f"ScaledDistribution scale must be finite and > 0 (got {self.scale})")

    def sample(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        """Draw scaled samples from the base distribution."""
        return self.base.sample(rng, size) * self.scale

    def p50(self) -> float:
        """Return the analytical P50."""
        return self.base.p50() * self.scale

    def p95(self) -> float:
        """Return the analytical P95."""
        return self.base.p95() * self.scale

    def p99(self) -> float:
        """Return the analytical P99."""
        return self.base.p99() * self.scale

    def quantile(self, q: float) -> float:
        """Return the analytical quantile for ``q`` in ``(0, 1)``."""
        return self.base.quantile(q) * self.scale

    def mean(self) -> float:
        """Return the analytical mean."""
        return self.base.mean() * self.scale

    def std(self) -> float:
        """Return the analytical standard deviation."""
        return self.base.std() * self.scale

    def cdf(self, value: float) -> float:
        """Return the CDF evaluated at ``value``."""
        return self.base.cdf(value / self.scale)


# Alias so scenario configs can read ``HeavyTail(...)`` instead of ``LogNormal(...)``
# when the intent is "heavy-tailed family" rather than the specific parameterisation.
HeavyTail = LogNormal


# Latency family registry used by the eval grid factory. Maps the canonical
# family name (as it appears in scenario YAML / experiment configs) to its
# constructor. Keep the keys stable; downstream code keys on these names.
LATENCY_FAMILIES: dict[str, type] = {
    "uniform": Uniform,
    "normal": Normal,
    "heavy_tail": LogNormal,
}


@runtime_checkable
class LatencyDistribution(Protocol):
    """Structural type for latency distributions used by the simulator.

    Provider/router code that holds a distribution instance should use
    this Protocol rather than a concrete class so the eval grid's
    ``Uniform`` / ``Normal`` families flow through the same code path
    that historically only saw ``LogNormal``.

    Any object exposing the methods below counts — concrete classes like
    :class:`Uniform`, :class:`Normal`, :class:`LogNormal`, plus mocks in
    tests, all match.
    """

    def sample(self, rng: np.random.Generator, size: int = 1) -> np.ndarray: ...

    def p50(self) -> float: ...

    def p95(self) -> float: ...

    def p99(self) -> float: ...

    def quantile(self, q: float) -> float: ...

    def mean(self) -> float: ...

    def std(self) -> float: ...

    def cdf(self, value: float) -> float: ...


__all__ = [
    "LATENCY_FAMILIES",
    "MIN_LATENCY_MS",
    "HeavyTail",
    "LatencyDistribution",
    "LogNormal",
    "Normal",
    "ScaledDistribution",
    "Uniform",
]
