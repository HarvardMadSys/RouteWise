"""Smart hedging for latency-aware routing.

This module implements Phase 4 of the latency routing system:
- Hedge decision strategies (never, always, fixed-timeout, smart)
- Backup provider selection (fastest, lp_other, cheapest_viable)
- Cost modeling with full billing

Reference: docs/algorithm/phase4_smart_hedging.md
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rwsim.policies.latency_routers.online_lp import ProviderProfile


class HedgingStrategy(Enum):
    """Hedging strategy types."""

    NEVER = "never"
    ALWAYS = "always"
    FIXED_TIMEOUT = "fixed_timeout"
    SMART_SURVIVAL = "smart_survival"
    SMART_RESIDUAL = "smart_residual"
    PERCENTILE_BASED = "percentile_based"  # Hedge at primary's P-alpha percentile
    SMART_ECONOMIC = "smart_economic"  # Cost-benefit model: P_viol * F_backup > C_b/V


class BackupSelectionMethod(Enum):
    """Backup provider selection methods."""

    FASTEST = "fastest"
    LP_OTHER = "lp_other"
    CHEAPEST_VIABLE = "cheapest_viable"


@dataclass
class HedgingParams:
    """Configuration parameters for hedging strategies."""

    strategy: HedgingStrategy
    slo_sec: float
    alpha: float = 0.7  # For fixed-timeout: hedge at alpha * slo_sec
    alpha_percentile: float = 90.0  # For percentile-based: hedge at P-alpha of primary
    cost_ratio: float = 0.1  # For smart_economic: C_b / V (backup cost / violation penalty)
    dispatch_overhead_sec: float = 0.05  # Backup launch overhead
    backup_method: BackupSelectionMethod = BackupSelectionMethod.FASTEST
    min_viable_cdf: float = 0.90  # For cheapest_viable selection


@dataclass
class HedgingResult:
    """Result of a hedged request."""

    primary_provider: str
    backup_provider: str | None
    hedged: bool
    hedge_time_sec: float | None
    winner: str  # "primary" or "backup"
    final_ttft_sec: float
    primary_cost: float
    backup_cost: float
    primary_error: str | None = None
    backup_error: str | None = None

    @property
    def total_cost(self) -> float:
        """Return total cost (both requests billed if hedged)."""
        if not self.hedged:
            return self.primary_cost
        return self.primary_cost + self.backup_cost


# -----------------------------------------------------------------------------
# Survival Functions (SEPARATE mode for hedging)
# -----------------------------------------------------------------------------


def get_survival_for_hedging(
    profile: ProviderProfile,
    t_sec: float,
    now: float,
) -> float:
    """Compute survival function S(t) for hedging decisions.

    Uses SEPARATE mode: only successful samples.
    Failures are handled separately in hedge trigger logic.

    Args:
        profile: Provider profile.
        t_sec: Time threshold in seconds.
        now: Decision timestamp. Uses samples with timestamps < now.

    Returns:
        S(t) = P(T > t | success) in [0, 1].
    """
    samples = profile.get_samples_before(now)
    if not samples:
        return 1.0  # No data, assume high latency

    n_survived = sum(1 for s in samples if s > t_sec)
    return n_survived / len(samples)


def get_cdf_for_hedging(
    profile: ProviderProfile,
    t_sec: float,
    now: float,
) -> float:
    """Compute CDF F(t) for hedging decisions.

    Uses SEPARATE mode: only successful samples.

    Args:
        profile: Provider profile.
        t_sec: Time threshold in seconds.
        now: Decision timestamp.

    Returns:
        F(t) = P(T <= t | success) in [0, 1].
    """
    return 1.0 - get_survival_for_hedging(profile, t_sec, now)


def compute_conditional_expectation(
    profile: ProviderProfile,
    elapsed_sec: float,
    now: float,
    min_samples: int = 5,
    max_latency_sec: float = 30.0,
) -> float:
    """Compute E[T - elapsed | T > elapsed] with stability protections.

    Args:
        profile: Provider profile.
        elapsed_sec: Time already elapsed in seconds.
        now: Decision timestamp.
        min_samples: Minimum surviving samples required.
        max_latency_sec: Upper bound truncation to avoid outlier inflation.

    Returns:
        Conditional expected remaining time in seconds.
    """
    samples = profile.get_samples_before(now)
    survived = [s for s in samples if s > elapsed_sec]

    # Protection 1: Minimum sample threshold
    if len(survived) < min_samples:
        # Fallback: use P99 as conservative upper bound
        if samples:
            p99 = float(np.percentile(samples, 99))
            return max(0.0, p99 - elapsed_sec)
        return max_latency_sec - elapsed_sec

    # Protection 2: Upper bound truncation
    survived_capped = [min(s, max_latency_sec) for s in survived]

    return float(np.mean(survived_capped)) - elapsed_sec


def compute_expected_latency(
    profile: ProviderProfile,
    now: float,
    max_latency_sec: float = 30.0,
) -> float:
    """Compute E[T] for a provider.

    Args:
        profile: Provider profile.
        now: Decision timestamp.
        max_latency_sec: Upper bound truncation.

    Returns:
        Expected latency in seconds.
    """
    samples = profile.get_samples_before(now)
    if not samples:
        return max_latency_sec

    capped = [min(s, max_latency_sec) for s in samples]
    return float(np.mean(capped))


# -----------------------------------------------------------------------------
# Hedge Decision Functions
# -----------------------------------------------------------------------------


def smart_hedge_survival(
    primary: str,
    backup: str,
    elapsed_sec: float,
    slo_sec: float,
    profiles: dict[str, ProviderProfile],
    now: float,
    dispatch_overhead_sec: float = 0.0,
) -> bool:
    """Hedge decision based on conditional survival probability.

    Hedge condition: P(violation if wait) > P(violation if hedge)
    Equivalently: S_primary(L) / S_primary(elapsed) > S_backup(L - elapsed - delta)

    Args:
        primary: Primary provider name.
        backup: Backup provider name.
        elapsed_sec: Time already elapsed in seconds.
        slo_sec: SLO deadline in seconds.
        profiles: Provider profiles.
        now: Decision timestamp.
        dispatch_overhead_sec: Backup launch overhead.

    Returns:
        True if hedging is recommended.
    """
    # Primary's conditional violation probability
    S_primary_L = get_survival_for_hedging(profiles[primary], slo_sec, now)
    S_primary_elapsed = get_survival_for_hedging(profiles[primary], elapsed_sec, now)

    if S_primary_elapsed < 1e-6:
        return True  # Already exceeded all samples, must hedge

    P_violation_wait = S_primary_L / S_primary_elapsed

    # Backup's violation probability
    remaining_budget = slo_sec - elapsed_sec - dispatch_overhead_sec
    if remaining_budget <= 0:
        return True  # No time left, must hedge

    S_backup_remaining = get_survival_for_hedging(profiles[backup], remaining_budget, now)
    P_violation_hedge = S_backup_remaining

    return P_violation_wait > P_violation_hedge


def smart_hedge_residual(
    primary: str,
    backup: str,
    elapsed_sec: float,
    slo_sec: float,
    profiles: dict[str, ProviderProfile],
    now: float,
    dispatch_overhead_sec: float = 0.0,
) -> bool:
    """Hedge decision based on residual life expectation.

    From advisor: "E[latency | elapsed time] + E[latency of fastest provider] > SLO"

    Formally: elapsed + E[T_primary - elapsed | T_primary > elapsed] +
              delta + E[T_backup] > SLO

    Args:
        primary: Primary provider name.
        backup: Backup provider name.
        elapsed_sec: Time already elapsed in seconds.
        slo_sec: SLO deadline in seconds.
        profiles: Provider profiles.
        now: Decision timestamp.
        dispatch_overhead_sec: Backup launch overhead.

    Returns:
        True if hedging is recommended.
    """
    E_remaining = compute_conditional_expectation(
        profiles[primary],
        elapsed_sec,
        now=now,
    )
    E_backup = compute_expected_latency(profiles[backup], now=now)

    return elapsed_sec + E_remaining + dispatch_overhead_sec + E_backup > slo_sec


def smart_hedge_economic(
    primary: str,
    backup: str,
    elapsed_sec: float,
    slo_sec: float,
    profiles: dict[str, ProviderProfile],
    now: float,
    cost_ratio: float = 0.1,
    dispatch_overhead_sec: float = 0.05,
) -> bool:
    """Hedge when expected benefit of avoiding violation exceeds backup cost.

    Decision rule:
        P_viol(t) * F_b(remaining) > C_b / V

    Left side: probability that hedging actually prevents a violation.
      - P_viol(t) = S_p(L)/S_p(t) = P(primary violates | survived to t)
      - F_b(remaining) = P(backup finishes within remaining budget)
    Right side: cost ratio (backup cost / violation penalty).

    Args:
        primary: Primary provider name.
        backup: Backup provider name.
        elapsed_sec: Time already elapsed in seconds.
        slo_sec: SLO deadline in seconds.
        profiles: Provider profiles.
        now: Decision timestamp.
        cost_ratio: C_b / V, backup cost relative to violation penalty.
        dispatch_overhead_sec: Backup launch overhead.

    Returns:
        True if hedging is recommended.
    """
    remaining = slo_sec - elapsed_sec - dispatch_overhead_sec
    if remaining <= 0:
        return False  # No time for backup, hedge is pointless

    S_L = get_survival_for_hedging(profiles[primary], slo_sec, now)
    S_t = get_survival_for_hedging(profiles[primary], elapsed_sec, now)

    P_viol = 1.0 if S_t < 1e-6 else S_L / S_t

    F_backup = get_cdf_for_hedging(profiles[backup], remaining, now)

    return P_viol * F_backup > cost_ratio


# -----------------------------------------------------------------------------
# Find Optimal Hedge Time
# -----------------------------------------------------------------------------


def find_optimal_hedge_time_survival(
    primary: str,
    backup: str,
    profiles: dict[str, ProviderProfile],
    now: float,
    slo_sec: float,
    dispatch_overhead_sec: float = 0.0,
    resolution_sec: float = 0.1,
) -> float:
    """Find minimum h where P(violation|wait) > P(violation|hedge).

    Grid search with resolution steps.

    Args:
        primary: Primary provider name.
        backup: Backup provider name.
        profiles: Provider profiles.
        now: Decision timestamp.
        slo_sec: SLO deadline in seconds.
        dispatch_overhead_sec: Backup launch overhead.
        resolution_sec: Grid search step size.

    Returns:
        Optimal hedge time in seconds.
    """
    h_values = np.arange(0, slo_sec, resolution_sec)
    for h in h_values:
        if smart_hedge_survival(
            primary,
            backup,
            elapsed_sec=h,
            slo_sec=slo_sec,
            profiles=profiles,
            now=now,
            dispatch_overhead_sec=dispatch_overhead_sec,
        ):
            return float(h)
    return float("inf")  # Never hedge if condition never met


def find_optimal_hedge_time_residual(
    primary: str,
    backup: str,
    profiles: dict[str, ProviderProfile],
    now: float,
    slo_sec: float,
    dispatch_overhead_sec: float = 0.0,
    resolution_sec: float = 0.1,
) -> float:
    """Find minimum h where residual life condition triggers hedge.

    Grid search with resolution steps.

    Args:
        primary: Primary provider name.
        backup: Backup provider name.
        profiles: Provider profiles.
        now: Decision timestamp.
        slo_sec: SLO deadline in seconds.
        dispatch_overhead_sec: Backup launch overhead.
        resolution_sec: Grid search step size.

    Returns:
        Optimal hedge time in seconds.
    """
    h_values = np.arange(0, slo_sec, resolution_sec)
    for h in h_values:
        if smart_hedge_residual(
            primary,
            backup,
            elapsed_sec=h,
            slo_sec=slo_sec,
            profiles=profiles,
            now=now,
            dispatch_overhead_sec=dispatch_overhead_sec,
        ):
            return float(h)
    return float("inf")  # Never hedge if condition never met


def find_optimal_hedge_time_economic(
    primary: str,
    backup: str,
    profiles: dict[str, ProviderProfile],
    now: float,
    slo_sec: float,
    cost_ratio: float = 0.1,
    dispatch_overhead_sec: float = 0.05,
    resolution_sec: float = 0.1,
) -> float:
    """Find minimum h where cost-benefit condition triggers hedge.

    Grid search with resolution steps.

    Args:
        primary: Primary provider name.
        backup: Backup provider name.
        profiles: Provider profiles.
        now: Decision timestamp.
        slo_sec: SLO deadline in seconds.
        cost_ratio: C_b / V threshold.
        dispatch_overhead_sec: Backup launch overhead.
        resolution_sec: Grid search step size.

    Returns:
        Optimal hedge time in seconds. Returns inf if condition never met.
    """
    h_values = np.arange(0, slo_sec, resolution_sec)
    for h in h_values:
        if smart_hedge_economic(
            primary,
            backup,
            elapsed_sec=h,
            slo_sec=slo_sec,
            profiles=profiles,
            now=now,
            cost_ratio=cost_ratio,
            dispatch_overhead_sec=dispatch_overhead_sec,
        ):
            return float(h)
    return float("inf")  # Never hedge if condition never met


def find_percentile_hedge_time(
    primary: str,
    profiles: dict[str, ProviderProfile],
    now: float,
    percentile: float = 90.0,
) -> float:
    """Find hedge time based on primary provider's latency percentile.

    Hedge at the P-percentile of primary's latency distribution.
    This is useful for reducing tail latency (P99) by hedging at P90.

    Args:
        primary: Primary provider name.
        profiles: Provider profiles.
        now: Decision timestamp.
        percentile: Percentile to use for hedge time (default: 90).

    Returns:
        Hedge time in seconds (P-percentile of primary's latency).
    """
    samples = profiles[primary].get_samples_before(now)
    if not samples:
        return float("inf")  # No data, don't hedge

    # samples are already in seconds (get_samples_before converts from ms)
    p_value = float(np.percentile(samples, percentile))
    return p_value


# -----------------------------------------------------------------------------
# Backup Provider Selection
# -----------------------------------------------------------------------------


def get_fastest_provider(
    profiles: dict[str, ProviderProfile],
    now: float,
    exclude: str | None = None,
) -> str | None:
    """Select provider with lowest P50 TTFT.

    Args:
        profiles: Provider profiles.
        now: Decision timestamp.
        exclude: Provider to exclude (usually the primary).

    Returns:
        Fastest provider name, or None if no valid candidates.
    """
    candidates = [p for p in profiles if p != exclude]
    if not candidates:
        return None

    def get_p50(provider: str) -> float:
        samples = profiles[provider].get_samples_before(now)
        if not samples:
            return float("inf")
        return float(np.percentile(samples, 50))

    return min(candidates, key=get_p50)


def get_lp_backup(
    lp_weights: dict[str, float],
    primary: str,
    min_weight: float = 0.01,
) -> str | None:
    """Get the other provider from LP solution.

    Args:
        lp_weights: LP solution weights.
        primary: Primary provider to exclude.
        min_weight: Minimum weight threshold.

    Returns:
        Backup provider name, or None if no valid candidates.
    """
    candidates = [p for p in lp_weights if p != primary and lp_weights[p] > min_weight]
    if candidates:
        # Return the one with highest weight (most likely to be good)
        return max(candidates, key=lambda p: lp_weights[p])
    return None


def get_cheapest_viable(
    profiles: dict[str, ProviderProfile],
    costs: dict[str, float],
    slo_sec: float,
    now: float,
    exclude: str | None = None,
    min_cdf: float = 0.90,
) -> str | None:
    """Select cheapest provider that can likely meet SLO.

    Args:
        profiles: Provider profiles.
        costs: Provider costs.
        slo_sec: SLO deadline in seconds.
        now: Decision timestamp.
        exclude: Provider to exclude.
        min_cdf: Minimum CDF at SLO required.

    Returns:
        Cheapest viable provider name, or None if no valid candidates.
    """
    viable = []
    for provider in profiles:
        if provider == exclude:
            continue
        cdf = get_cdf_for_hedging(profiles[provider], slo_sec, now)
        if cdf >= min_cdf:
            viable.append(provider)

    if viable:
        return min(viable, key=lambda p: costs.get(p, float("inf")))
    return None


def select_backup(
    method: BackupSelectionMethod,
    profiles: dict[str, ProviderProfile],
    costs: dict[str, float],
    primary: str,
    slo_sec: float,
    now: float,
    lp_weights: dict[str, float] | None = None,
    min_viable_cdf: float = 0.90,
) -> str | None:
    """Select backup provider based on method.

    Args:
        method: Backup selection method.
        profiles: Provider profiles.
        costs: Provider costs.
        primary: Primary provider to exclude.
        slo_sec: SLO deadline.
        now: Decision timestamp.
        lp_weights: LP solution weights (for LP_OTHER method).
        min_viable_cdf: Minimum CDF threshold for CHEAPEST_VIABLE.

    Returns:
        Backup provider name, or None if no valid candidates.
    """
    if method == BackupSelectionMethod.FASTEST:
        return get_fastest_provider(profiles, now, exclude=primary)
    elif method == BackupSelectionMethod.LP_OTHER:
        if lp_weights is None:
            # Fallback to fastest
            return get_fastest_provider(profiles, now, exclude=primary)
        backup = get_lp_backup(lp_weights, primary)
        if backup is None:
            # Fallback to fastest
            return get_fastest_provider(profiles, now, exclude=primary)
        return backup
    elif method == BackupSelectionMethod.CHEAPEST_VIABLE:
        backup = get_cheapest_viable(
            profiles,
            costs,
            slo_sec,
            now,
            exclude=primary,
            min_cdf=min_viable_cdf,
        )
        if backup is None:
            # Fallback to fastest
            return get_fastest_provider(profiles, now, exclude=primary)
        return backup
    else:
        raise ValueError(f"Unknown backup selection method: {method}")


# -----------------------------------------------------------------------------
# SmartHedger Class
# -----------------------------------------------------------------------------


class SmartHedger:
    """Smart hedging manager that integrates with Phase 3 router.

    Decides when to hedge and which backup provider to use.
    """

    def __init__(
        self,
        params: HedgingParams,
        costs: dict[str, float],
    ):
        """Initialize the hedger.

        Args:
            params: Hedging configuration.
            costs: Provider costs.
        """
        self.params = params
        self.costs = costs

    def compute_hedge_time(
        self,
        primary: str,
        backup: str,
        profiles: dict[str, ProviderProfile],
        now: float,
    ) -> float:
        """Compute the hedge time for a given primary and backup.

        Args:
            primary: Primary provider.
            backup: Backup provider.
            profiles: Provider profiles.
            now: Decision timestamp.

        Returns:
            Hedge time in seconds (inf means never hedge).
        """
        strategy = self.params.strategy

        if strategy == HedgingStrategy.NEVER:
            return float("inf")

        elif strategy == HedgingStrategy.ALWAYS:
            return 0.0

        elif strategy == HedgingStrategy.FIXED_TIMEOUT:
            return self.params.alpha * self.params.slo_sec

        elif strategy == HedgingStrategy.SMART_SURVIVAL:
            return find_optimal_hedge_time_survival(
                primary,
                backup,
                profiles,
                now=now,
                slo_sec=self.params.slo_sec,
                dispatch_overhead_sec=self.params.dispatch_overhead_sec,
            )

        elif strategy == HedgingStrategy.SMART_RESIDUAL:
            return find_optimal_hedge_time_residual(
                primary,
                backup,
                profiles,
                now=now,
                slo_sec=self.params.slo_sec,
                dispatch_overhead_sec=self.params.dispatch_overhead_sec,
            )

        elif strategy == HedgingStrategy.PERCENTILE_BASED:
            return find_percentile_hedge_time(
                primary,
                profiles,
                now=now,
                percentile=self.params.alpha_percentile,
            )

        elif strategy == HedgingStrategy.SMART_ECONOMIC:
            return find_optimal_hedge_time_economic(
                primary,
                backup,
                profiles,
                now=now,
                slo_sec=self.params.slo_sec,
                cost_ratio=self.params.cost_ratio,
                dispatch_overhead_sec=self.params.dispatch_overhead_sec,
            )

        else:
            raise ValueError(f"Unknown hedging strategy: {strategy}")

    def simulate_request(
        self,
        primary: str,
        profiles: dict[str, ProviderProfile],
        now: float,
        T_primary_sec: float,
        err_primary: str | None,
        T_backup_sec: float,
        err_backup: str | None,
        backup: str,
    ) -> HedgingResult:
        """Simulate a request with hedging.

        Args:
            primary: Primary provider.
            profiles: Provider profiles.
            now: Decision timestamp.
            T_primary_sec: Sampled primary latency in seconds.
            err_primary: Primary error type (None if success).
            T_backup_sec: Sampled backup latency in seconds.
            err_backup: Backup error type (None if success).
            backup: Backup provider.

        Returns:
            HedgingResult with outcome details.
        """
        # Compute hedge time
        h = self.compute_hedge_time(primary, backup, profiles, now)

        # Determine if hedge is triggered
        if h == float("inf"):
            hedged = False
        elif h == 0.0:
            hedged = True  # Always hedge
        else:
            # Hedge if primary hasn't responded by h or primary failed
            hedged = (T_primary_sec > h) or (err_primary is not None)

        # Compute final TTFT
        if not hedged:
            final_ttft = T_primary_sec
            winner = "primary"
            backup_cost = 0.0
        else:
            # Both requests in flight
            delta = self.params.dispatch_overhead_sec
            if err_primary is not None:
                # Primary failed, use backup
                if err_backup is not None:
                    # Both failed
                    final_ttft = max(T_primary_sec, h + delta + T_backup_sec)
                    winner = "primary"  # Arbitrary, both failed
                else:
                    final_ttft = h + delta + T_backup_sec
                    winner = "backup"
            elif err_backup is not None:
                # Backup failed, use primary
                final_ttft = T_primary_sec
                winner = "primary"
            else:
                # Both succeeded, use faster
                backup_arrival = h + delta + T_backup_sec
                if T_primary_sec <= backup_arrival:
                    final_ttft = T_primary_sec
                    winner = "primary"
                else:
                    final_ttft = backup_arrival
                    winner = "backup"

            backup_cost = self.costs.get(backup, 0.0)

        return HedgingResult(
            primary_provider=primary,
            backup_provider=backup if hedged else None,
            hedged=hedged,
            hedge_time_sec=h if hedged else None,
            winner=winner,
            final_ttft_sec=final_ttft,
            primary_cost=self.costs.get(primary, 0.0),
            backup_cost=backup_cost,
            primary_error=err_primary,
            backup_error=err_backup if hedged else None,
        )
