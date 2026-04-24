"""Compatibility wrapper for the V2 latency-router policy stage."""

from __future__ import annotations

from rwsim.policies.latency_routers.v2 import V2Router
from rwsim.policies.latency_routers.online_lp import (
    FailureMode,
    ProviderProfile,
    pre_filter,
)

__all__ = ["FailureMode", "ProviderProfile", "V2Router", "pre_filter"]
