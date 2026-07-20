"""Latency-profile names shared by RouteWise simulator policies.

The rolling-window estimator lives in :mod:`llm_routewise.core.latency_profile` and
the observed-vs-prior fallback logic in :mod:`llm_routewise.core.beliefs`
(``LatencyBeliefs``); the simulator's oracle prior is supplied by
``_SimProviderView`` in :mod:`llm_routewise.sim.policies.routewise`. This module keeps the
simulator-side aliases: ``latency_profile_mode="configured"`` maps to the
beliefs ``prior_only`` mode.
"""

from __future__ import annotations

from typing import Literal

from llm_routewise.core.latency_profile import (
    DEFAULT_PROFILE_WINDOW_SEC,
    RollingLatencyProfile,
)

LatencyProfileMode = Literal["observed", "configured"]

__all__ = [
    "DEFAULT_PROFILE_WINDOW_SEC",
    "LatencyProfileMode",
    "RollingLatencyProfile",
]
