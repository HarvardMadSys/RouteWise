"""Online latency-aware provider router for OpenRouter.

This module implements Phase 3 of the latency routing system:
- Single moving window profiler (refreshed by periodic probing)
- LP-based optimal provider mixing
- Smooth Weighted Round-Robin (SWRR) sampler

Reference: docs/algorithm/phase3_online_latency_routing.md
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.optimize import linprog


class FailureMode(Enum):
    """How to handle failures in CDF computation.

    INFINITY: Failures count as latency = infinity (missed deadline).
              This is the recommended mode as it naturally incorporates
              reliability into the CDF constraint.

    SEPARATE: Ignore failures in the tail constraint and use the conditional
              latency CDF, P(TTFT <= L | success). This is useful as an
              ablation when errors are treated as a separate reliability
              dimension (e.g., penalized via kappa or hard filters).
    """

    INFINITY = "infinity"
    SEPARATE = "separate"


@dataclass
class ProviderProfile:
    """Real-time latency profile for a provider.

    Maintains a single time-based moving window for latency samples and error tracking.
    Uses periodic probing to keep profiles fresh.
    """

    provider: str

    # Latency samples: list of (timestamp, ttft_ms)
    # Only successful requests are stored here
    samples: list[tuple[float, float]] = field(default_factory=list)

    # Error tracking: list of (timestamp, error_type)
    # error_type: "timeout" | "rate_limit" | "server_error" | None (success)
    error_samples: list[tuple[float, str | None]] = field(default_factory=list)

    # Configuration
    failure_mode: FailureMode = FailureMode.INFINITY
    window_sec: float = 15 * 60  # 15 minutes

    def add_sample(
        self,
        timestamp: float,
        ttft_ms: float,
        error_type: str | None = None,
    ) -> None:
        """Add a new sample to the profile.

        Args:
            timestamp: Unix timestamp of the sample.
            ttft_ms: Time to first token in milliseconds (-1 if error).
            error_type: None for success, or one of "timeout", "rate_limit", "server_error".
        """
        # Record error/success status
        self.error_samples.append((timestamp, error_type))

        # Only add to latency window if successful
        if error_type is None and ttft_ms > 0:
            self.samples.append((timestamp, ttft_ms))

    def _prune_window(self, current_time: float) -> None:
        """Remove samples outside the time window."""
        cutoff = current_time - self.window_sec
        self.samples = [(t, v) for t, v in self.samples if t >= cutoff]
        self.error_samples = [(t, e) for t, e in self.error_samples if t >= cutoff]

    def get_samples_before(self, cutoff_time: float) -> list[float]:
        """Get latency samples (in seconds) before a cutoff time.

        Used for time-causal evaluation: only use data with timestamp < t
        when making decisions at time t.
        """
        # Filter by time window and cutoff
        min_time = cutoff_time - self.window_sec
        samples_ms = [v for t, v in self.samples if min_time <= t < cutoff_time]

        # Convert to seconds
        return [v / 1000.0 for v in samples_ms]

    def get_error_rate_before(self, cutoff_time: float, window_sec: float | None = None) -> float:
        """Get error rate in the window before cutoff_time.

        Args:
            cutoff_time: Only consider samples before this time.
            window_sec: Window duration in seconds. If None, uses self.window_sec.

        Returns:
            Error rate (fraction of failed requests).
        """
        if window_sec is None:
            window_sec = self.window_sec
        min_time = cutoff_time - window_sec
        samples = [(t, e) for t, e in self.error_samples if min_time <= t < cutoff_time]

        if not samples:
            return 0.0

        error_count = sum(1 for _, e in samples if e is not None)
        return error_count / len(samples)

    def get_cdf_at(
        self,
        L_sec: float,
        current_time: float | None = None,
    ) -> float:
        """Compute empirical CDF at latency threshold L.

        Args:
            L_sec: Latency threshold in seconds.
            current_time: Reference time for window pruning. If None, uses time.time().

        Returns:
            CDF value F(L) in [0, 1].
        """
        if current_time is None:
            current_time = time.time()

        self._prune_window(current_time)

        samples = self.get_samples_before(current_time)
        if not samples:
            return 0.0

        F_success = sum(1 for s in samples if s <= L_sec) / len(samples)

        if self.failure_mode == FailureMode.SEPARATE:
            return F_success

        error_rate = self.get_error_rate_before(current_time)
        success_rate = 1.0 - error_rate
        return success_rate * F_success

    def get_error_rates(self, window_sec: float | None = None) -> dict[str, float]:
        """Return error rates by type for the recent window.

        Args:
            window_sec: Window duration in seconds. If None, uses self.window_sec.

        Returns:
            Dict with keys "timeout", "rate_limit", "server_error" and their rates.
        """
        current_time = time.time()
        if window_sec is None:
            window_sec = self.window_sec
        min_time = current_time - window_sec
        samples = [(t, e) for t, e in self.error_samples if t >= min_time]

        if not samples:
            return {"timeout": 0.0, "rate_limit": 0.0, "server_error": 0.0}

        counts = {"timeout": 0, "rate_limit": 0, "server_error": 0}
        for _, error_type in samples:
            if error_type in counts:
                counts[error_type] += 1

        total = len(samples)
        return {k: v / total for k, v in counts.items()}

    def get_p99(self, current_time: float | None = None) -> float:
        """Return P99 latency in seconds (for reporting).

        Args:
            current_time: Reference time. If None, uses time.time().

        Returns:
            P99 latency in seconds, or inf if no samples.
        """
        if current_time is None:
            current_time = time.time()

        samples = self.get_samples_before(current_time)

        if not samples:
            return float("inf")

        return float(np.percentile(samples, 99))


def pre_filter(
    profiles: dict[str, ProviderProfile],
    current_time: float,
    L_min: float = 1.0,
    max_error_rate: float = 0.05,
    min_cdf_threshold: float = 0.80,
) -> list[str]:
    """Hard filtering to remove clearly unusable providers.

    Filtering rules:
    1. total_error_rate > 5% -> offline (clearly broken)
    2. F_hat(L_min) < 0.80 -> offline (can't meet even relaxed SLO)

    Args:
        profiles: Provider name -> ProviderProfile mapping.
        current_time: Reference time for window computation.
        L_min: Minimum SLO for filtering (default 1s).
        max_error_rate: Maximum allowed error rate (default 5%).
        min_cdf_threshold: Minimum CDF at L_min (default 80%).

    Returns:
        List of eligible provider names.
    """
    eligible = []

    for name, profile in profiles.items():
        # Check error rate
        error_rate = profile.get_error_rate_before(current_time)
        if error_rate > max_error_rate:
            continue  # Hard offline: clearly broken

        # Check CDF at minimum SLO
        cdf_at_min = profile.get_cdf_at(L_min, current_time)
        if cdf_at_min < min_cdf_threshold:
            continue  # Can't meet basic SLO

        eligible.append(name)

    return eligible


def solve_lp(
    providers: list[str],
    profiles: dict[str, ProviderProfile],
    costs: dict[str, float],
    slo_sec: float,
    current_time: float,
    kappa: float = 0.0,
    target_cdf: float = 0.99,
    alpha: float = 0.0,
) -> dict[str, float] | None:
    """Solve the cost-minimization LP with tail constraint.

    minimize:   sum_j pi_j * (c_j * (1 + kappa * e_j) + alpha * p50_j)
    subject to: sum_j pi_j * F_j(L) >= target_cdf
                sum_j pi_j = 1
                pi_j >= 0

    The alpha term is the "latency-weighted cost" extension proposed by
    Juncheng on 2026-04-21: each provider's effective cost includes an
    alpha-scaled P50 penalty, so the LP can prefer faster providers when
    alpha > 0. This recovers V2-like behavior (prefer Friendli over Novita
    on Llama) as alpha grows, while alpha = 0 preserves the original
    cost-first lp_mix behavior.

    Args:
        providers: List of provider names to consider.
        profiles: Provider name -> ProviderProfile mapping.
        costs: Provider name -> cost per request.
        slo_sec: SLO latency in seconds.
        current_time: Reference time for CDF computation.
        kappa: Error penalty coefficient (default 0).
        target_cdf: Target CDF value (default 0.99).
        alpha: Latency weight coefficient applied to each provider's P50
            (milliseconds). Default 0 preserves original lp_mix behavior.

    Returns:
        Dict of provider -> weight, or None if infeasible.
    """
    n = len(providers)
    if n == 0:
        return None

    # Compute CDFs, error rates, and P50s (in ms so alpha has units USD/ms)
    cdfs = []
    error_rates = []
    p50s = []
    for p in providers:
        profile = profiles[p]
        cdfs.append(profile.get_cdf_at(slo_sec, current_time))
        error_rates.append(profile.get_error_rate_before(current_time))
        # get_samples_before returns SECONDS; convert to ms for the P50 term
        samples = profile.get_samples_before(current_time)
        p50s.append(float(np.median(samples)) * 1000.0 if samples else 0.0)

    cdfs = np.array(cdfs)
    error_rates = np.array(error_rates)
    p50s = np.array(p50s)

    # Check if any single provider can satisfy the constraint
    if np.max(cdfs) < target_cdf:
        return None  # Infeasible

    # Objective: sum_j pi_j * (c_j * (1 + kappa * e_j) + alpha * p50_j)
    c = np.array(
        [
            costs.get(p, 1.0) * (1 + kappa * error_rates[i]) + alpha * p50s[i]
            for i, p in enumerate(providers)
        ]
    )

    # Inequality constraint: -sum_j pi_j * F_j(L) <= -target_cdf
    # (scipy.linprog uses A_ub @ x <= b_ub)
    A_ub = -cdfs.reshape(1, -1)
    b_ub = np.array([-target_cdf])

    # Equality constraint: sum_j pi_j = 1
    A_eq = np.ones((1, n))
    b_eq = np.array([1.0])

    # Bounds: pi_j >= 0
    bounds = [(0, None) for _ in range(n)]

    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if result.success:
        weights = {}
        for i, p in enumerate(providers):
            if result.x[i] > 1e-6:  # Threshold for numerical noise
                # Ensure JSON-serializable Python floats (avoid numpy scalar types).
                weights[p] = float(result.x[i])

        # Normalize to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {p: float(w / total) for p, w in weights.items()}

        return weights

    return None


def solve_lp_with_fallback(
    providers: list[str],
    profiles: dict[str, ProviderProfile],
    costs: dict[str, float],
    slo_sec: float,
    current_time: float,
    relaxation_factors: list[float] | None = None,
    kappa: float = 0.0,
    alpha: float = 0.0,
) -> tuple[dict[str, float], str]:
    """Solve LP with progressive relaxation fallback.

    Args:
        providers: List of provider names.
        profiles: Provider profiles.
        costs: Cost per request.
        slo_sec: Original SLO in seconds.
        current_time: Reference time.
        relaxation_factors: Factors to relax SLO by if infeasible. Defaults to [1.2, 1.5, 2.0].
        kappa: Error penalty coefficient.

    Returns:
        (weights_dict, status) where status is one of:
        - "optimal": Original SLO achieved
        - "relaxed_1.2x" / "relaxed_1.5x" / "relaxed_2.0x": Relaxed SLO used
        - "best_effort": All relaxations failed, use max F_j(L) provider
    """
    if relaxation_factors is None:
        relaxation_factors = [1.2, 1.5, 2.0]

    # Try original SLO
    result = solve_lp(providers, profiles, costs, slo_sec, current_time, kappa, alpha=alpha)
    if result is not None:
        return result, "optimal"

    # Try relaxed SLOs
    for factor in relaxation_factors:
        relaxed_slo = slo_sec * factor
        result = solve_lp(
            providers, profiles, costs, relaxed_slo, current_time, kappa, alpha=alpha
        )
        if result is not None:
            return result, f"relaxed_{factor}x"

    # Best-effort: select provider with max F_j(L)
    if providers:
        best_provider = max(providers, key=lambda p: profiles[p].get_cdf_at(slo_sec, current_time))
        return {best_provider: 1.0}, "best_effort"

    return {}, "no_providers"


class SWRRSampler:
    """Smooth Weighted Round-Robin sampler.

    Given weights pi = {A: 0.7, B: 0.3}, produces a smooth interleaving:
    A, A, B, A, A, B, A, A, A, B, ...

    Equivalent to probabilistic mixing but with reduced short-term variance.
    """

    def __init__(self, weights: dict[str, float] | None = None):
        """Initialize SWRR sampler.

        Args:
            weights: Initial provider weights (must sum to 1).
        """
        if weights is None:
            weights = {}

        self.providers = list(weights.keys())
        self.weights = weights.copy()
        self.current_weights = dict.fromkeys(self.providers, 0.0)

    def next(self) -> str | None:
        """Select next provider using SWRR algorithm.

        Returns:
            Selected provider name, or None if no providers.
        """
        if not self.providers:
            return None

        # Add original weights
        for p in self.providers:
            self.current_weights[p] += self.weights[p]

        # Select provider with highest current weight
        selected = max(self.providers, key=lambda p: self.current_weights[p])

        # Subtract total weight from selected
        total_weight = sum(self.weights.values())
        self.current_weights[selected] -= total_weight

        return selected

    def update_weights(self, new_weights: dict[str, float], smoothing: float = 0.3):
        """Update weights with exponential smoothing.

        pi_new = smoothing * pi_LP + (1 - smoothing) * pi_old

        Args:
            new_weights: New weights from LP solver.
            smoothing: Smoothing factor (default 0.3).
        """
        # Handle provider changes
        all_providers = set(self.providers) | set(new_weights.keys())

        smoothed_weights = {}
        for p in all_providers:
            old_w = self.weights.get(p, 0.0)
            new_w = new_weights.get(p, 0.0)
            smoothed_weights[p] = smoothing * new_w + (1 - smoothing) * old_w

        # Remove providers with negligible weight
        self.weights = {p: w for p, w in smoothed_weights.items() if w > 0.001}
        self.providers = list(self.weights.keys())

        # Normalize
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {p: w / total for p, w in self.weights.items()}

        # Update current_weights for new providers, soft reset for existing
        new_current = {}
        for p in self.providers:
            if p in self.current_weights:
                new_current[p] = self.current_weights[p] * 0.5  # Soft reset
            else:
                new_current[p] = 0.0
        self.current_weights = new_current

    def get_weights(self) -> dict[str, float]:
        """Return current weights."""
        return self.weights.copy()


class OnlineLatencyRouter:
    """Online latency-aware router for multi-provider platforms.

    Integrates:
    - Moving window profiler (ProviderProfile)
    - Pre-filtering (hard constraints)
    - LP solver with fallback
    - SWRR sampler
    """

    def __init__(
        self,
        costs: dict[str, float],
        slo_sec: float = 3.0,
        window_sec: float = 15 * 60,  # 15 minutes
        failure_mode: FailureMode = FailureMode.INFINITY,
        kappa: float = 0.0,
        weight_smoothing: float = 0.3,
        lp_update_interval: float = 60.0,  # seconds
        relaxation_factors: list[float] | None = None,
        alpha: float = 0.0,
    ):
        """Initialize the router.

        Args:
            costs: Provider name -> cost per request.
            slo_sec: SLO latency in seconds.
            window_sec: Moving window duration in seconds (default 15 min).
            failure_mode: How to handle failures in CDF.
            kappa: Error penalty coefficient.
            weight_smoothing: Smoothing factor for weight updates.
            lp_update_interval: Minimum seconds between LP updates.
            relaxation_factors: SLO relaxation factors for fallback. Defaults to [1.2, 1.5, 2.0].
            alpha: Latency weight coefficient for the LP objective. Default 0
                preserves the original cost-first lp_mix behavior; larger
                values trade cost for lower P50 and recover V2-like behavior
                at the limit.
        """
        self.costs = costs
        self.slo_sec = slo_sec
        self.window_sec = window_sec
        self.failure_mode = failure_mode
        self.kappa = kappa
        self.weight_smoothing = weight_smoothing
        self.lp_update_interval = lp_update_interval
        self.relaxation_factors = (
            relaxation_factors if relaxation_factors is not None else [1.2, 1.5, 2.0]
        )
        self.alpha = alpha

        # Initialize profiles for all providers
        self.profiles: dict[str, ProviderProfile] = {}
        for provider in costs:
            self.profiles[provider] = ProviderProfile(
                provider=provider,
                window_sec=window_sec,
                failure_mode=failure_mode,
            )

        # SWRR sampler (initialized empty, will be updated after first LP solve)
        self.sampler = SWRRSampler()

        # Tracking
        self.last_lp_update: float = 0.0
        self.last_lp_status: str = "not_run"
        self.last_lp_weights: dict[str, float] = {}

    def add_sample(
        self,
        provider: str,
        timestamp: float,
        ttft_ms: float,
        error_type: str | None = None,
    ) -> None:
        """Add a latency sample to the profile.

        Args:
            provider: Provider name.
            timestamp: Unix timestamp.
            ttft_ms: Time to first token in ms (-1 if error).
            error_type: None for success, or error type string.
        """
        if provider not in self.profiles:
            self.profiles[provider] = ProviderProfile(
                provider=provider,
                window_sec=self.window_sec,
                failure_mode=self.failure_mode,
            )

        self.profiles[provider].add_sample(timestamp, ttft_ms, error_type)

    def update_lp(self, current_time: float | None = None, force: bool = False) -> bool:
        """Update LP solution if enough time has passed.

        Args:
            current_time: Reference time. If None, uses time.time().
            force: Force update regardless of time interval.

        Returns:
            True if LP was updated, False if skipped.
        """
        if current_time is None:
            current_time = time.time()

        # Check if update is needed
        if not force and (current_time - self.last_lp_update) < self.lp_update_interval:
            return False

        # Pre-filter providers
        eligible = pre_filter(self.profiles, current_time)

        if not eligible:
            # Fallback: use all providers
            eligible = list(self.profiles.keys())

        # Solve LP
        weights, status = solve_lp_with_fallback(
            providers=eligible,
            profiles=self.profiles,
            costs=self.costs,
            slo_sec=self.slo_sec,
            current_time=current_time,
            relaxation_factors=self.relaxation_factors,
            kappa=self.kappa,
            alpha=self.alpha,
        )

        # Update sampler with smoothing
        self.sampler.update_weights(weights, self.weight_smoothing)

        # Record state
        self.last_lp_update = current_time
        self.last_lp_status = status
        self.last_lp_weights = weights

        return True

    def route(self, current_time: float | None = None) -> str | None:
        """Select a provider for the next request.

        Args:
            current_time: Reference time for LP update check.

        Returns:
            Selected provider name, or None if no providers.
        """
        if current_time is None:
            current_time = time.time()

        # Update LP if needed
        self.update_lp(current_time)

        # Sample from SWRR
        return self.sampler.next()

    def get_status(self) -> dict:
        """Return current router status for debugging/logging."""
        return {
            "lp_status": self.last_lp_status,
            "lp_weights": self.last_lp_weights,
            "sampler_weights": self.sampler.get_weights(),
            "profile_summary": {
                p: {
                    "p99": profile.get_p99(),
                    "error_rate": profile.get_error_rate_before(time.time()),
                }
                for p, profile in self.profiles.items()
            },
        }


def load_pricing(pricing_file: str) -> dict[str, float]:
    """Load provider pricing from JSON and compute average cost per request.

    The pricing file format is:
    {
        "Provider": [input_price_per_M, output_price_per_M],
        ...
    }

    We compute average cost assuming typical request sizes from profiling:
    - Input: 50 tokens (probe request)
    - Output: 10 tokens (short response)

    Args:
        pricing_file: Path to JSON pricing file.

    Returns:
        Dict of provider -> cost per request (in dollars).
    """
    with open(pricing_file) as f:
        pricing = json.load(f)

    # Typical request size for profiling
    input_tokens = 50
    output_tokens = 10

    costs = {}
    for provider, prices in pricing.items():
        input_price, output_price = prices
        # Convert from $/M tokens to $/request
        cost = (input_price * input_tokens + output_price * output_tokens) / 1_000_000
        costs[provider] = cost

    return costs
