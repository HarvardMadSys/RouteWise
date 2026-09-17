"""Stale-telemetry robustness experiment (hedging under sudden latency spikes)."""

from experiments.ablations.stale_telemetry.harness import (
    DEFAULT_MAGNITUDES,
    DEFAULT_PERIOD_MIN,
    DEFAULT_SPIKE_DURATION_MINUTES,
    SECTION_NAME,
    SpikeTrainConfig,
    main,
    make_scenario,
    make_scenarios,
)
from experiments.ablations.stale_telemetry.policy import StaleBeliefRouteWisePolicy
from experiments.ablations.stale_telemetry.presets import make_presets, policy_name

__all__ = [
    "DEFAULT_MAGNITUDES",
    "DEFAULT_PERIOD_MIN",
    "DEFAULT_SPIKE_DURATION_MINUTES",
    "SECTION_NAME",
    "SpikeTrainConfig",
    "StaleBeliefRouteWisePolicy",
    "main",
    "make_presets",
    "make_scenario",
    "make_scenarios",
    "policy_name",
]
