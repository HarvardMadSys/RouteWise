"""Shared RouteWise hedging primitives.

This module is limited to pure checkpoint/probability math and generic backup
selection. Harnesses own provider inventory, capacity mutation, transport, and
profile storage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

from rwsim.const import (
    DISPATCH_OVERHEAD_MS,
    HEDGE_CHECKPOINT_END_FRACTION,
    HEDGE_CHECKPOINT_INTERVAL_FRACTION,
    HEDGE_CHECKPOINT_START_FRACTION,
    HEDGE_SUCCESS_TARGET,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

ProviderT = TypeVar("ProviderT")
EPS = 1e-9


def _ceil_to_interval_ms(value_ms: float, interval_ms: float) -> float:
    return math.ceil((value_ms - EPS) / interval_ms) * interval_ms


def hedge_checkpoints_for_slo(slo_ms: float) -> tuple[float, ...]:
    """Return SLO-relative hedge checkpoints as elapsed seconds.

    Args:
        slo_ms: Request SLO in milliseconds. Non-positive values yield ``()``.

    Returns:
        Tuple of checkpoint times in seconds, sorted ascending.
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
class BackupCandidate(Generic[ProviderT]):
    """One backup candidate scored at a hedging checkpoint."""

    provider: ProviderT
    success_probability: float
    marginal_cost: float
    true_mean_ms: float
    success_target: float = HEDGE_SUCCESS_TARGET

    @property
    def feasible(self) -> bool:
        """Return whether this candidate satisfies the success target."""
        return self.success_probability >= self.success_target - EPS


def combined_success_probability(
    primary_cdf_ms: Callable[[float], float],
    backup_cdf_ms: Callable[[float], float],
    *,
    elapsed_ms: float,
    slo_ms: float,
    dispatch_overhead_ms: float = 0.0,
) -> float:
    """Return probability that hedging now keeps the request within SLO.

    The request has already waited ``elapsed_ms`` for the primary provider.
    The returned probability is conditional on the primary not having already
    finished by that elapsed time.
    """
    primary_cdf_t = primary_cdf_ms(elapsed_ms)
    primary_cdf_slo = primary_cdf_ms(slo_ms)
    primary_survival_t = max(1.0 - primary_cdf_t, 0.0)
    if primary_survival_t <= EPS:
        return 0.0

    p_not_violate = max(primary_cdf_slo - primary_cdf_t, 0.0) / primary_survival_t
    p_violate = max(1.0 - primary_cdf_slo, 0.0) / primary_survival_t
    remaining_ms = slo_ms - elapsed_ms - dispatch_overhead_ms
    backup_success = 0.0 if remaining_ms <= 0.0 else backup_cdf_ms(remaining_ms)
    return float(min(max(p_not_violate + p_violate * backup_success, 0.0), 1.0))


def latest_safe_hedge_delay_sec(
    success_probability_at_sec: Callable[[float], float],
    *,
    slo_sec: float,
    success_target: float = HEDGE_SUCCESS_TARGET,
    grid_step_sec: float = 0.05,
) -> float:
    """Return the latest grid delay whose success probability meets target."""
    max_elapsed_sec = max(0.0, float(slo_sec))
    step = max(float(grid_step_sec), EPS)
    latest_safe: float | None = None
    count = math.floor(max_elapsed_sec / step + EPS)
    for index in range(count + 1):
        elapsed_sec = min(index * step, max_elapsed_sec)
        if success_probability_at_sec(elapsed_sec) >= success_target:
            latest_safe = elapsed_sec
    if latest_safe is not None:
        return latest_safe
    return float("inf")


def select_probability_backup(
    candidates: Sequence[BackupCandidate[ProviderT]],
) -> BackupCandidate[ProviderT] | None:
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
            _provider_name(candidate.provider),
        ),
    )


def best_backup_success_probability(candidates: Sequence[BackupCandidate[object]]) -> float:
    """Return the best success probability among backup candidates."""
    if not candidates:
        return 0.0
    return max(candidate.success_probability for candidate in candidates)


def has_feasible_backup(candidates: Sequence[BackupCandidate[object]]) -> bool:
    """Return whether any backup candidate satisfies the success target."""
    return any(candidate.feasible for candidate in candidates)


def _provider_name(provider: object) -> str:
    name = getattr(provider, "name", None)
    if name is not None:
        return str(name)
    spec = getattr(provider, "spec", None)
    spec_name = getattr(spec, "name", None)
    if spec_name is not None:
        return str(spec_name)
    return str(provider)


__all__ = [
    "DISPATCH_OVERHEAD_MS",
    "EPS",
    "HEDGE_CHECKPOINT_END_FRACTION",
    "HEDGE_CHECKPOINT_INTERVAL_FRACTION",
    "HEDGE_CHECKPOINT_START_FRACTION",
    "HEDGE_SUCCESS_TARGET",
    "BackupCandidate",
    "best_backup_success_probability",
    "combined_success_probability",
    "has_feasible_backup",
    "hedge_checkpoints_for_slo",
    "latest_safe_hedge_delay_sec",
    "select_probability_backup",
]
