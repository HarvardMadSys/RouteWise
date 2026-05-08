"""Hedging timing and backup-selection ablation package."""

from experiments.ablations.hedging.policy import HedgingAblationPolicy
from experiments.ablations.hedging.presets import (
    BACKUP_SELECTIONS,
    DEFAULT_MODES,
    DEFAULT_P_VALUES,
    DISPATCH_TIMINGS,
    PRODUCTION_BACKUP_SELECTION,
    PRODUCTION_DISPATCH_TIMING,
    ablation_policy_name,
    make_ablation_presets,
    parse_ablation_policy_name,
    production_baseline_policy_name,
)

__all__ = [
    "BACKUP_SELECTIONS",
    "DEFAULT_MODES",
    "DEFAULT_P_VALUES",
    "DISPATCH_TIMINGS",
    "HedgingAblationPolicy",
    "PRODUCTION_BACKUP_SELECTION",
    "PRODUCTION_DISPATCH_TIMING",
    "ablation_policy_name",
    "make_ablation_presets",
    "parse_ablation_policy_name",
    "production_baseline_policy_name",
]
