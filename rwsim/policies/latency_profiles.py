"""Latency-profile strategies used by RouteWise policies.

The pure rolling-window estimator lives in :mod:`rwsim.core.latency_profile`
(shared with the real-eval harness); this module keeps the simulator-side
strategies, which are the only pieces that touch the world model (the
``configured`` mode and the empty-window fallback read the provider's true
distribution).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

from rwsim.core.latency_profile import (
    DEFAULT_PROFILE_WINDOW_SEC,
    RollingLatencyProfile,
)

if TYPE_CHECKING:
    from rwsim.world.providers import Provider

LatencyProfileMode = Literal["observed", "configured"]


class LatencyProfileStrategy(Protocol):
    """Source of latency objective and CDF estimates for RouteWise."""

    def ensure_provider(self, provider_name: str) -> None:
        """Initialize any provider-local state."""

    def mean_ms(self, provider: Provider, now: float) -> float:
        """Return the latency objective in milliseconds."""

    def cdf_ms(self, provider: Provider, value_ms: float, now: float) -> float:
        """Return ``P(provider latency <= value_ms)`` at ``now``."""

    def observe(self, provider_name: str, observed_at: float, ttft_ms: float) -> None:
        """Record an observed TTFT sample."""


@dataclass
class ObservedRollingLatencyProfileStrategy:
    """Latency profile learned from observed request outcomes.

    When a provider's window holds no samples, queries fall back to the
    provider's configured (true) distribution. ``fallback_counts`` /
    ``query_counts`` track how often that oracle fallback fires per provider
    so experiment harnesses can report it.
    """

    window_sec: float = DEFAULT_PROFILE_WINDOW_SEC
    profiles: dict[str, RollingLatencyProfile] = field(default_factory=dict)
    query_counts: dict[str, int] = field(default_factory=dict)
    fallback_counts: dict[str, int] = field(default_factory=dict)

    def ensure_provider(self, provider_name: str) -> None:
        self._profile(provider_name)

    def mean_ms(self, provider: Provider, now: float) -> float:
        self.query_counts[provider.name] = self.query_counts.get(provider.name, 0) + 1
        mean = self._profile(provider.name).mean(now)
        if mean is not None:
            return mean
        self.fallback_counts[provider.name] = self.fallback_counts.get(provider.name, 0) + 1
        return provider.true_mean_ms(now)

    def cdf_ms(self, provider: Provider, value_ms: float, now: float) -> float:
        empirical = self._profile(provider.name).cdf(value_ms, now)
        if empirical is not None:
            return empirical
        return provider._active_ttft_dist(now).cdf(value_ms)

    def observe(self, provider_name: str, observed_at: float, ttft_ms: float) -> None:
        self._profile(provider_name).add_sample(observed_at, ttft_ms)

    def fallback_rate(self) -> float:
        """Return the fraction of mean queries that hit the true-mean fallback."""
        total_queries = sum(self.query_counts.values())
        if total_queries == 0:
            return 0.0
        return sum(self.fallback_counts.values()) / total_queries

    def _profile(self, name: str) -> RollingLatencyProfile:
        return self.profiles.setdefault(
            name,
            RollingLatencyProfile(window_sec=self.window_sec),
        )


@dataclass
class ConfiguredLatencyProfileStrategy:
    """Latency profile that uses the configured provider distributions directly."""

    def ensure_provider(self, provider_name: str) -> None:
        del provider_name

    def mean_ms(self, provider: Provider, now: float) -> float:
        return provider.true_mean_ms(now)

    def cdf_ms(self, provider: Provider, value_ms: float, now: float) -> float:
        return provider._active_ttft_dist(now).cdf(value_ms)

    def observe(self, provider_name: str, observed_at: float, ttft_ms: float) -> None:
        del provider_name, observed_at, ttft_ms


def make_latency_profile_strategy(
    mode: LatencyProfileMode,
    *,
    window_sec: float,
    profiles: dict[str, RollingLatencyProfile] | None = None,
) -> LatencyProfileStrategy:
    """Build a RouteWise latency-profile strategy."""
    if mode == "observed":
        return ObservedRollingLatencyProfileStrategy(
            window_sec=window_sec,
            profiles={} if profiles is None else profiles,
        )
    if mode == "configured":
        return ConfiguredLatencyProfileStrategy()
    raise ValueError(
        "unknown latency_profile_mode "
        f"{mode!r}; expected 'observed' or 'configured'"
    )


__all__ = [
    "DEFAULT_PROFILE_WINDOW_SEC",
    "ConfiguredLatencyProfileStrategy",
    "LatencyProfileMode",
    "LatencyProfileStrategy",
    "ObservedRollingLatencyProfileStrategy",
    "RollingLatencyProfile",
    "make_latency_profile_strategy",
]
