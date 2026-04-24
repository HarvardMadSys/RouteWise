"""Compatibility wrapper for latency-router policy stages."""

from __future__ import annotations

from rwsim.policies.latency_routers.online_lp import (
    FailureMode,
    OnlineLatencyRouter,
    ProviderProfile,
    SWRRSampler,
    load_pricing,
    pre_filter,
    solve_lp,
    solve_lp_with_fallback,
)

__all__ = [
    "FailureMode",
    "OnlineLatencyRouter",
    "ProviderProfile",
    "SWRRSampler",
    "load_pricing",
    "pre_filter",
    "solve_lp",
    "solve_lp_with_fallback",
]
