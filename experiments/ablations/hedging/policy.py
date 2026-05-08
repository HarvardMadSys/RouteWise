"""Ablation-local RouteWise hedging policy variants.

The production RouteWisePolicy keeps its public surface clean. This subclass
only changes two protected hedging hooks so the ablation can isolate:

- dispatch timing: latest_safe vs earliest_safe
- backup selection: probability vs random_feasible
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from rwsim.policies.hedging import BackupCandidate, select_probability_backup
from rwsim.policies.routewise import RouteWisePolicy

DispatchTiming = Literal["latest_safe", "earliest_safe"]
BackupSelection = Literal["probability", "random_feasible"]

DISPATCH_TIMINGS: tuple[DispatchTiming, ...] = ("latest_safe", "earliest_safe")
BACKUP_SELECTIONS: tuple[BackupSelection, ...] = ("probability", "random_feasible")
_BACKUP_RNG_SEED_OFFSET = 1_000_003


@dataclass
class HedgingAblationPolicy(RouteWisePolicy):
    """RouteWise hedging ablation with experiment-scoped knobs."""

    hedging: str | bool = "probability_target"
    explorer: bool = False
    dispatch_timing: DispatchTiming = "latest_safe"
    backup_selection: BackupSelection = "probability"
    _backup_rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.dispatch_timing not in DISPATCH_TIMINGS:
            known = ", ".join(DISPATCH_TIMINGS)
            raise ValueError(
                f"unknown hedging dispatch_timing {self.dispatch_timing!r}; known: {known}"
            )
        if self.backup_selection not in BACKUP_SELECTIONS:
            known = ", ".join(BACKUP_SELECTIONS)
            raise ValueError(
                f"unknown hedging backup_selection {self.backup_selection!r}; known: {known}"
            )
        if self.hedging != "probability_target":
            raise ValueError(
                "HedgingAblationPolicy only supports hedging='probability_target'"
            )
        if self.explorer:
            raise ValueError("HedgingAblationPolicy keeps explorer disabled")
        super().__post_init__()
        self._backup_rng = np.random.default_rng(
            int(self.seed) + _BACKUP_RNG_SEED_OFFSET
        )

    def _select_backup_candidate(
        self,
        candidates: list[BackupCandidate],
    ) -> BackupCandidate | None:
        if self.backup_selection == "probability":
            return select_probability_backup(candidates)
        feasible = [candidate for candidate in candidates if candidate.feasible]
        if not feasible:
            return None
        return feasible[int(self._backup_rng.integers(0, len(feasible)))]

    def _should_dispatch_now(
        self,
        selected: BackupCandidate,
        *,
        future_feasible: bool,
    ) -> bool:
        if self.dispatch_timing == "earliest_safe":
            # Parent tick already gated on selected.feasible; earliest_safe
            # dispatches at the first feasible checkpoint.
            return True
        return super()._should_dispatch_now(
            selected,
            future_feasible=future_feasible,
        )


__all__ = [
    "BACKUP_SELECTIONS",
    "DISPATCH_TIMINGS",
    "BackupSelection",
    "DispatchTiming",
    "HedgingAblationPolicy",
]
