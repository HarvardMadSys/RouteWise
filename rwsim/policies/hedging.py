"""Shared hedging primitives for RouteWise policies.

This module contains only stable hedging math and candidate-selection helpers.
Experiment-specific knobs such as earliest/latest dispatch or random backup
ablation should live in experiment policies, not here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

# Re-export so existing `from rwsim.policies.hedging import ...` call sites
# keep working. The values live in `rwsim.const`; edit them there.
from rwsim.const import (
    DISPATCH_OVERHEAD_MS,
    HEDGE_CHECKPOINT_END_FRACTION,
    HEDGE_CHECKPOINT_INTERVAL_FRACTION,
    HEDGE_CHECKPOINT_START_FRACTION,
    HEDGE_SUCCESS_TARGET,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from rwsim.schemas import Request
    from rwsim.world.providers import Provider

EPS = 1e-9

def _ceil_to_interval_ms(value_ms: float, interval_ms: float) -> float:
    return math.ceil((value_ms - EPS) / interval_ms) * interval_ms


def hedge_checkpoints_for_slo(slo_ms: float) -> tuple[float, ...]:
    """Return SLO-relative hedge checkpoints as elapsed seconds.

    Generates an increasing tuple of elapsed times (in seconds) at which the
    policy re-evaluates whether to dispatch a hedge backup. Checkpoints span
    ``HEDGE_CHECKPOINT_START_FRACTION`` through ``HEDGE_CHECKPOINT_END_FRACTION``
    of the request's SLO, spaced ``HEDGE_CHECKPOINT_INTERVAL_FRACTION`` of SLO
    apart.

    Args:
        slo_ms: Request SLO in milliseconds. Non-positive values yield ``()``.

    Returns:
        Tuple of checkpoint times in seconds, sorted ascending. The first
        element is the earliest time at which the policy may consider hedging;
        the last is the latest. Empty if ``slo_ms`` is non-positive.
    """
    slo_ms_f = max(0.0, float(slo_ms))
    interval_ms = slo_ms_f * HEDGE_CHECKPOINT_INTERVAL_FRACTION
    if interval_ms <= 0.0:
        return ()
    start_ms = _ceil_to_interval_ms(
        slo_ms_f * HEDGE_CHECKPOINT_START_FRACTION,
        interval_ms,
    )
    end_ms = slo_ms_f * HEDGE_CHECKPOINT_END_FRACTION
    if start_ms > end_ms + EPS:
        return ()
    checkpoints_ms: list[float] = []
    current_ms = start_ms
    while current_ms <= end_ms + EPS:
        checkpoints_ms.append(current_ms)
        current_ms += interval_ms
    return tuple(ms / 1000.0 for ms in checkpoints_ms)


@dataclass(frozen=True)
class BackupCandidate:
    """One non-primary backup provider scored at a hedging checkpoint."""

    provider: Provider
    success_probability: float
    marginal_cost: float
    true_mean_ms: float
    success_target: float = HEDGE_SUCCESS_TARGET

    @property
    def feasible(self) -> bool:
        """Return whether this candidate satisfies the success target."""
        return self.success_probability >= self.success_target - EPS


def combined_success_probability(
    cdf_ms: Callable[[Provider, float], float],
    primary: Provider,
    backup: Provider,
    *,
    elapsed_ms: float,
    slo_ms: float,
    dispatch_overhead_ms: float = DISPATCH_OVERHEAD_MS,
) -> float:
    """Return probability that hedging now keeps the request within SLO.

    The request has already waited ``elapsed_ms`` for the primary provider. The
    success probability is:

        P(primary finishes by SLO | primary has not finished by elapsed)
        + P(primary misses SLO | primary has not finished by elapsed)
          * P(backup finishes within remaining time)
    """
    primary_cdf_t = cdf_ms(primary, elapsed_ms)
    primary_cdf_slo = cdf_ms(primary, slo_ms)
    primary_survival_t = max(1.0 - primary_cdf_t, 0.0)
    if primary_survival_t <= EPS:
        return 0.0

    p_not_violate = max(primary_cdf_slo - primary_cdf_t, 0.0) / primary_survival_t
    p_violate = max(1.0 - primary_cdf_slo, 0.0) / primary_survival_t
    remaining_ms = slo_ms - elapsed_ms - dispatch_overhead_ms
    backup_success = 0.0 if remaining_ms <= 0.0 else cdf_ms(backup, remaining_ms)
    return float(min(max(p_not_violate + p_violate * backup_success, 0.0), 1.0))


def collect_backup_candidates(
    providers: Iterable[Provider],
    primary: Provider,
    request: Request,
    *,
    now: float,
    elapsed_ms: float,
    slo_ms: float,
    cdf_ms: Callable[[Provider, float], float],
    success_target: float = HEDGE_SUCCESS_TARGET,
    dispatch_overhead_ms: float = DISPATCH_OVERHEAD_MS,
    marginal_cost: Callable[[Provider], float] | None = None,
) -> list[BackupCandidate]:
    """Score available non-primary providers as backup candidates."""
    candidates: list[BackupCandidate] = []
    for provider in providers:
        if provider.name == primary.name or not provider.is_available(now):
            continue
        cost = (
            marginal_cost(provider)
            if marginal_cost is not None
            else provider.marginal_cost_for_request(request, now)
        )
        candidates.append(
            BackupCandidate(
                provider=provider,
                success_probability=combined_success_probability(
                    cdf_ms,
                    primary,
                    provider,
                    elapsed_ms=elapsed_ms,
                    slo_ms=slo_ms,
                    dispatch_overhead_ms=dispatch_overhead_ms,
                ),
                marginal_cost=cost,
                true_mean_ms=provider.true_mean_ms(now),
                success_target=success_target,
            )
        )
    return candidates


def select_probability_backup(
    candidates: Sequence[BackupCandidate],
) -> BackupCandidate | None:
    """Select the cheapest feasible backup, with probability/latency tie-breaks."""
    feasible = [candidate for candidate in candidates if candidate.feasible]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda candidate: (
            candidate.marginal_cost,
            -candidate.success_probability,
            candidate.true_mean_ms,
            candidate.provider.name,
        ),
    )


def best_backup_success_probability(candidates: Sequence[BackupCandidate]) -> float:
    """Return the best success probability among backup candidates."""
    if not candidates:
        return 0.0
    return max(candidate.success_probability for candidate in candidates)


def has_feasible_backup(candidates: Sequence[BackupCandidate]) -> bool:
    """Return whether any backup candidate satisfies the success target."""
    return any(candidate.feasible for candidate in candidates)


__all__ = [
    "DISPATCH_OVERHEAD_MS",
    "EPS",
    "HEDGE_CHECKPOINT_END_FRACTION",
    "HEDGE_CHECKPOINT_INTERVAL_FRACTION",
    "HEDGE_CHECKPOINT_START_FRACTION",
    "HEDGE_SUCCESS_TARGET",
    "BackupCandidate",
    "best_backup_success_probability",
    "collect_backup_candidates",
    "combined_success_probability",
    "has_feasible_backup",
    "hedge_checkpoints_for_slo",
    "select_probability_backup",
]
