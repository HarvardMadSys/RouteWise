"""Preset helpers for hedging dispatch/backup-selection ablations."""

from __future__ import annotations

from typing import Any

from experiments.ablations.hedging.policy import (
    BACKUP_SELECTIONS,
    DISPATCH_TIMINGS,
    BackupSelection,
    DispatchTiming,
)
from experiments.simulation.common import WORKLOAD_COST_ENVELOPE, p_label

DEFAULT_P_VALUES = (0.75,)
PRODUCTION_DISPATCH_TIMING: DispatchTiming = "latest_safe"
PRODUCTION_BACKUP_SELECTION: BackupSelection = "probability"
DEFAULT_MODES: tuple[tuple[DispatchTiming, BackupSelection], ...] = (
    ("latest_safe", "probability"),
    ("earliest_safe", "probability"),
    ("latest_safe", "random_feasible"),
)


def ablation_policy_name(
    dispatch_timing: DispatchTiming,
    backup_selection: BackupSelection,
    *,
    p: float,
) -> str:
    """Return a stable policy name for one hedging ablation mode."""
    _validate_dispatch_timing(dispatch_timing)
    _validate_backup_selection(backup_selection)
    return (
        f"hedging__dispatch={dispatch_timing}"
        f"__backup={backup_selection}__{p_label(p)}"
    )


def production_baseline_policy_name(*, p: float = DEFAULT_P_VALUES[0]) -> str:
    """Return the production-equivalent ablation baseline policy name."""
    return ablation_policy_name(
        PRODUCTION_DISPATCH_TIMING,
        PRODUCTION_BACKUP_SELECTION,
        p=p,
    )


def parse_ablation_policy_name(
    policy: str,
) -> tuple[DispatchTiming, BackupSelection, float]:
    """Parse a policy name emitted by :func:`ablation_policy_name`."""
    prefix = "hedging__"
    if not policy.startswith(prefix):
        raise ValueError(f"not a hedging ablation policy: {policy!r}")
    parts = policy.removeprefix(prefix).split("__")
    if len(parts) != 3 or not parts[0].startswith("dispatch=") or not parts[1].startswith("backup="):
        raise ValueError(f"invalid hedging ablation policy: {policy!r}")
    dispatch_timing = _validate_dispatch_timing(parts[0].removeprefix("dispatch="))
    backup_selection = _validate_backup_selection(parts[1].removeprefix("backup="))
    return dispatch_timing, backup_selection, _parse_p_label(parts[2])


def make_ablation_presets(
    *,
    modes: tuple[tuple[DispatchTiming, BackupSelection], ...] = DEFAULT_MODES,
    p_values: tuple[float, ...] = DEFAULT_P_VALUES,
    cost_envelope: tuple[float, float] | str = WORKLOAD_COST_ENVELOPE,
) -> dict[str, dict[str, Any]]:
    """Build ablation-local presets for hedging mode sweeps."""
    presets: dict[str, dict[str, Any]] = {}
    for p in p_values:
        for dispatch_timing, backup_selection in modes:
            name = ablation_policy_name(dispatch_timing, backup_selection, p=p)
            presets[name] = {
                "policy": "HedgingAblationPolicy",
                "params": {
                    "dispatch_timing": dispatch_timing,
                    "backup_selection": backup_selection,
                    "p": float(p),
                    "cost_envelope": cost_envelope,
                    "latency_profile_mode": "configured",
                },
            }
    return presets


def _parse_p_label(label: str) -> float:
    if not label.startswith("p"):
        raise ValueError(f"invalid p label {label!r}")
    try:
        return int(label.removeprefix("p")) / 100.0
    except ValueError as exc:
        raise ValueError(f"invalid p label {label!r}") from exc


def _validate_dispatch_timing(value: str) -> DispatchTiming:
    if value not in DISPATCH_TIMINGS:
        known = ", ".join(DISPATCH_TIMINGS)
        raise ValueError(f"unknown hedging dispatch timing {value!r}; known: {known}")
    return value  # type: ignore[return-value]


def _validate_backup_selection(value: str) -> BackupSelection:
    if value not in BACKUP_SELECTIONS:
        known = ", ".join(BACKUP_SELECTIONS)
        raise ValueError(f"unknown hedging backup selection {value!r}; known: {known}")
    return value  # type: ignore[return-value]


__all__ = [
    "BACKUP_SELECTIONS",
    "DEFAULT_MODES",
    "DEFAULT_P_VALUES",
    "DISPATCH_TIMINGS",
    "PRODUCTION_BACKUP_SELECTION",
    "PRODUCTION_DISPATCH_TIMING",
    "ablation_policy_name",
    "make_ablation_presets",
    "parse_ablation_policy_name",
    "production_baseline_policy_name",
]
