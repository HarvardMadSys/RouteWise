"""Per-provider profile with Bernoulli SLO-miss estimation.

Key design (see JOINT_DESIGN_v2 section 2.2):

The statistical object we estimate is F_j(SLO) = P(TTFT_j <= SLO), not the
P95 latency. For a fixed threshold, each observation is a Bernoulli trial
(hit if TTFT <= SLO, miss otherwise), so concentration is binomial and
the safety filter becomes an exact Clopper-Pearson upper bound on the
miss probability.

This replaces the oracle-based `_provider_p95_at(...)` mechanism used in
the first iteration of the joint strategy. It relies only on observed
samples, matching what a production router could realistically collect.

Profile contents
----------------
- Rolling 15-minute window of (timestamp, ttft_ms) samples
- Per-sample Bernoulli indicator cached (hit = TTFT <= SLO)
- Aggregated counts: hits, misses, total

Exposed estimates
-----------------
- miss_rate_hat      empirical miss probability
- miss_rate_ucb      Clopper-Pearson one-sided upper bound
- uncertainty_radius UCB minus hat (used for exploration bonus)
- p50_estimate       rolling median TTFT (used by hedge primary/backup
                     reasoning only; NOT the safety filter)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import beta


@dataclass
class ProviderProfile:
    """Rolling-window profile with Bernoulli SLO-miss counts.

    The profile maintains two concentric data sources:

      - A prior (persistent hit/miss counts from pre-run benchmarking).
      - A rolling window of recent samples (evicted after window_sec).

    Estimates use prior + rolling combined. The prior corresponds to
    historical benchmarking data that production systems have available
    before a provider is put into rotation (e.g., OpenRouter's published
    latency percentiles, or an internal offline benchmark). Without a
    prior, short runs in the simulator make the Clopper-Pearson UCB
    pathologically wide for rarely-probed providers.

    Parameters
    ----------
    slo_ms:
        Threshold used to label samples as hit (TTFT <= SLO) or miss.
    window_sec:
        Rolling window. Older samples are evicted on update.
    """

    slo_ms: float
    window_sec: float = 900.0

    samples: deque = field(default_factory=lambda: deque())
    # Counts within the rolling window (kept in sync with `samples`).
    hits: int = 0
    misses: int = 0
    # Persistent prior counts (e.g., from pre-run benchmarking). These are
    # never evicted. Used to provide a stable Bayesian-like prior on
    # miss_rate when the rolling window is empty or sparse.
    prior_hits: int = 0
    prior_misses: int = 0

    # ------------------------------------------------------------------ #
    # Update                                                               #
    # ------------------------------------------------------------------ #

    def add_sample(self, t: float, ttft_ms: float) -> None:
        """Append one observation and evict any samples older than the window."""
        hit = ttft_ms <= self.slo_ms
        self.samples.append((t, ttft_ms, hit))
        if hit:
            self.hits += 1
        else:
            self.misses += 1
        self._evict_old(t)

    def _evict_old(self, now: float) -> None:
        cutoff = now - self.window_sec
        while self.samples and self.samples[0][0] < cutoff:
            _, _, hit = self.samples.popleft()
            if hit:
                self.hits -= 1
            else:
                self.misses -= 1

    def add_prior(self, hit_count: int, miss_count: int) -> None:
        """Inject persistent prior counts that survive window eviction."""
        self.prior_hits += hit_count
        self.prior_misses += miss_count

    # ------------------------------------------------------------------ #
    # Basic getters                                                        #
    # ------------------------------------------------------------------ #

    @property
    def n_samples(self) -> int:
        """Total count = rolling-window samples + prior samples."""
        return self.hits + self.misses + self.prior_hits + self.prior_misses

    @property
    def n_window(self) -> int:
        return self.hits + self.misses

    def _total_hits(self) -> int:
        return self.hits + self.prior_hits

    def _total_misses(self) -> int:
        return self.misses + self.prior_misses

    def miss_rate_hat(self) -> float:
        n = self.n_samples
        if n == 0:
            return 1.0  # no data -> worst-case assumption
        return self._total_misses() / n

    def last_update_time(self) -> float | None:
        if not self.samples:
            return None
        return self.samples[-1][0]

    # ------------------------------------------------------------------ #
    # Concentration bounds                                                  #
    # ------------------------------------------------------------------ #

    def miss_rate_ucb(self, delta: float) -> float:
        """Clopper-Pearson one-sided upper bound on miss probability.

        For x=misses, n=samples, the upper bound at confidence (1 - delta) is
            U = BetaInv(1 - delta; x + 1, n - x).

        Boundary conventions:
          - n == 0   -> return 1.0 (no data, assume worst-case miss rate)
          - x == n   -> return 1.0 (all misses observed)
        """
        n = self.n_samples
        if n == 0:
            return 1.0
        x = self._total_misses()
        if x == n:
            return 1.0
        return float(beta.ppf(1.0 - delta, x + 1, n - x))

    def uncertainty_radius(self, delta: float) -> float:
        """UCB minus point estimate. Used as the exploration-bonus width."""
        return max(0.0, self.miss_rate_ucb(delta) - self.miss_rate_hat())

    # ------------------------------------------------------------------ #
    # Latency summary (for hedge primary / backup reasoning only)          #
    # ------------------------------------------------------------------ #

    def p50_estimate(self) -> float | None:
        """Rolling-window median TTFT. None if profile is empty.

        The safety filter does NOT use this; it is kept purely so the
        hedge trigger can estimate "how long to wait before firing backup".
        """
        if not self.samples:
            return None
        ttfts = [ttft for _, ttft, _ in self.samples]
        return float(np.median(ttfts))

    def p95_estimate(self) -> float | None:
        if not self.samples:
            return None
        ttfts = [ttft for _, ttft, _ in self.samples]
        return float(np.percentile(ttfts, 95))


# ---------------------------------------------------------------------------
# Delta allocation across providers and time
# ---------------------------------------------------------------------------


def delta_allocation(
    base_delta: float,
    n_providers: int,
    t: float,
    t_min: float = 1.0,
) -> float:
    """Union-bound-friendly per-decision delta.

    Uses the sum_{t>=1} 1/t^2 = pi^2/6 trick to give a valid anytime bound.
    """
    import math
    t_eff = max(t, t_min)
    return base_delta * 6.0 / (math.pi ** 2 * n_providers * t_eff ** 2)
