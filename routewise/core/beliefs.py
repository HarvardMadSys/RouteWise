"""Latency beliefs: rolling empirical profiles plus prior/penalty fallbacks.

This unifies what used to be two implementations of "what do we think this
provider's TTFT looks like right now":

* simulator ``ObservedRollingLatencyProfileStrategy`` — rolling profile with
  an oracle fallback to the provider's true distribution when the window is
  empty (and a ``configured`` mode that always uses the oracle), and
* real-eval ``_body_latency_proxy_ms`` / ``LatencyProfile.cdf_at`` — rolling
  profile with a 60s synthetic penalty per failed attempt, an unphysical
  finite penalty for unprofiled providers, and errors counted as CDF misses.

The environment difference is carried entirely by the
:class:`~routewise.core.provider_view.ProviderView` prior hooks (simulator: true
distribution; real: ``None``) and by the penalty knobs below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from routewise.core.latency_profile import DEFAULT_PROFILE_WINDOW_SEC, RollingLatencyProfile

if TYPE_CHECKING:
    from routewise.core.provider_view import ProviderView

BeliefMode = Literal["observed", "prior_only"]

# Synthetic latency for failed attempts (429s, timeouts) so burst-sensitive
# providers are avoided after live rate-limit feedback. Inert in environments
# that never record errors (the simulator).
DEFAULT_ERROR_PENALTY_MS: float = 60_000.0

# Penalty for providers with no window samples AND no prior, so the LP still
# yields a feasible solution but never picks them when any profiled candidate
# is available. ~11.5 days in milliseconds — clearly unphysical, but finite.
DEFAULT_UNPROFILED_PENALTY_MS: float = 1e9


@dataclass
class LatencyBeliefs:
    """Per-provider TTFT beliefs shared by all router queries.

    ``profiles`` is injectable so a harness can share the exact profile
    objects with its own bookkeeping (the real-eval runner snapshots and
    bootstraps ``ProviderState.profile`` directly).

    Not thread-safe: queries mutate window bookkeeping. Real-eval callers
    already serialize every entry point behind the policy lock.
    """

    window_sec: float = DEFAULT_PROFILE_WINDOW_SEC
    mode: BeliefMode = "observed"
    error_penalty_ms: float = DEFAULT_ERROR_PENALTY_MS
    unprofiled_penalty_ms: float = DEFAULT_UNPROFILED_PENALTY_MS
    profiles: dict[str, RollingLatencyProfile] = field(default_factory=dict)
    # Oracle-fallback accounting. Mean queries drive the LP body objective;
    # CDF queries drive hedging success probabilities.
    query_counts: dict[str, int] = field(default_factory=dict)
    fallback_counts: dict[str, int] = field(default_factory=dict)
    cdf_query_counts: dict[str, int] = field(default_factory=dict)
    cdf_fallback_counts: dict[str, int] = field(default_factory=dict)

    def ensure(self, provider_name: str) -> None:
        """Initialize the profile for ``provider_name`` if missing."""
        if self.mode == "prior_only":
            return
        self.profile(provider_name)

    def profile(self, provider_name: str) -> RollingLatencyProfile:
        return self.profiles.setdefault(
            provider_name,
            RollingLatencyProfile(window_sec=self.window_sec),
        )

    def observe(self, provider_name: str, observed_at: float, ttft_ms: float) -> None:
        """Record an observed TTFT sample."""
        if self.mode == "prior_only":
            return
        self.profile(provider_name).add_sample(observed_at, ttft_ms)

    def observe_error(self, provider_name: str, observed_at: float, error_type: str) -> None:
        """Record a failed attempt (no TTFT)."""
        if self.mode == "prior_only":
            return
        self.profile(provider_name).add_error(observed_at, error_type)

    def mean_ms(self, view: ProviderView, now: float) -> float:
        """Latency objective ``T̄_j(now)`` for the LP body selector."""
        if self.mode == "prior_only":
            prior = view.prior_ttft_mean_ms(now)
            if prior is not None:
                return prior
            return self.unprofiled_penalty_ms
        self.query_counts[view.name] = self.query_counts.get(view.name, 0) + 1
        mean = self.profile(view.name).mean_with_errors(
            now,
            error_penalty_ms=self.error_penalty_ms,
        )
        if mean is not None:
            return mean
        self.fallback_counts[view.name] = self.fallback_counts.get(view.name, 0) + 1
        prior = view.prior_ttft_mean_ms(now)
        if prior is not None:
            return prior
        return self.unprofiled_penalty_ms

    def cdf(self, view: ProviderView, value_ms: float, now: float) -> float:
        """``P(TTFT <= value_ms)`` for the hedging success math."""
        if self.mode == "prior_only":
            prior = view.prior_ttft_cdf(value_ms, now)
            if prior is not None:
                return prior
            return 0.0
        self.cdf_query_counts[view.name] = self.cdf_query_counts.get(view.name, 0) + 1
        empirical = self.profile(view.name).cdf_counting_errors(value_ms, now)
        if empirical is not None:
            return empirical
        self.cdf_fallback_counts[view.name] = self.cdf_fallback_counts.get(view.name, 0) + 1
        prior = view.prior_ttft_cdf(value_ms, now)
        if prior is not None:
            return prior
        return 0.0

    def prior_mean_ms(self, view: ProviderView, now: float) -> float:
        """Prior mean with the unprofiled penalty as last resort.

        Used for the simulator's hedge tie-break, which reads the oracle mean
        directly and never counts as a profile query.
        """
        prior = view.prior_ttft_mean_ms(now)
        if prior is not None:
            return prior
        return self.unprofiled_penalty_ms

    def fallback_rate(self) -> float:
        """Deprecated alias for :meth:`mean_fallback_rate`."""
        return self.mean_fallback_rate()

    def mean_fallback_rate(self) -> float:
        """Fraction of mean queries that fell back past the rolling window."""
        return _fallback_rate(self.query_counts, self.fallback_counts)

    def cdf_fallback_rate(self) -> float:
        """Fraction of CDF queries that fell back past the rolling window."""
        return _fallback_rate(self.cdf_query_counts, self.cdf_fallback_counts)


def _fallback_rate(
    query_counts: dict[str, int],
    fallback_counts: dict[str, int],
) -> float:
    total_queries = sum(query_counts.values())
    if total_queries == 0:
        return 0.0
    return sum(fallback_counts.values()) / total_queries


__all__ = [
    "DEFAULT_ERROR_PENALTY_MS",
    "DEFAULT_UNPROFILED_PENALTY_MS",
    "BeliefMode",
    "LatencyBeliefs",
]
