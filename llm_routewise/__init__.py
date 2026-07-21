"""RouteWise's dependency-free API-provider routing facade."""

from __future__ import annotations

from llm_routewise.errors import NoProviderError, OutcomeError, RouteWiseError, ValidationError
from llm_routewise.facade import Attempt, Decision, Provider, Router, StatsSnapshot, Tuning
from llm_routewise.stateless import Candidate, RouteOnceResult, route_once

__version__ = "0.2.0"

__all__ = [
    "Attempt",
    "Candidate",
    "Decision",
    "NoProviderError",
    "OutcomeError",
    "Provider",
    "RouteOnceResult",
    "RouteWiseError",
    "Router",
    "StatsSnapshot",
    "Tuning",
    "ValidationError",
    "route_once",
]
