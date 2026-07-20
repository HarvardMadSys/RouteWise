"""Public exception hierarchy for the RouteWise library facade."""

from __future__ import annotations


class RouteWiseError(Exception):
    """Base class for errors raised by the public RouteWise API."""


class ValidationError(RouteWiseError, ValueError):
    """Raised when a caller supplies an invalid public-API argument."""


class NoProviderError(RouteWiseError):
    """Raised when no provider is eligible for a routing decision."""


class OutcomeError(RouteWiseError):
    """Raised when an attempt outcome conflicts with its existing state."""


__all__ = [
    "NoProviderError",
    "OutcomeError",
    "RouteWiseError",
    "ValidationError",
]
