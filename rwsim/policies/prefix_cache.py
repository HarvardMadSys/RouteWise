"""Trace-driven prefix-cache cost helpers.

The cached-input-token count for a request is read directly from the trace
(e.g. ``cache_read_tokens`` in the freeinference dataset). When the trace
does not supply that field we conservatively treat the request as a cold
miss; we no longer extrapolate cache locality from a provider-local prefix
model, which was too aggressive about predicting hits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rwsim.world.capacity import ProviderTier

if TYPE_CHECKING:
    from rwsim.engine.state import SimulationState
    from rwsim.schemas import Request
    from rwsim.world.providers import Provider


def response_tokens_for_request(request: Request) -> float:
    """Return the output token count used for billing-like calculations."""
    response_tokens = request.response_tokens
    if response_tokens is None:
        response_tokens = request.estimated_response_tokens
    if response_tokens is None:
        response_tokens = max(int(request.total_tokens or 0) - int(request.request_tokens or 0), 0)
    return float(response_tokens or 0.0)


def cached_input_tokens(
    provider: Provider,
    request: Request,
    state: SimulationState,
    *,
    cache_enabled: bool = True,
) -> int:
    """Return length-based cached input tokens for one provider/request pair.

    The value comes from the trace; if the trace does not report it for this
    request we return 0 (cold miss). We never synthesize cache hits from a
    provider-local prefix history.
    """
    if not cache_enabled or not _state_cache_enabled(state):
        return 0
    if provider.tier != ProviderTier.S_A:
        return 0
    if provider.cached_input_cost_per_token is None:
        return 0

    observed_cached_tokens = trace_observed_cached_input_tokens(request)
    if observed_cached_tokens is None:
        return 0
    request_tokens = max(int(request.request_tokens or 0), 0)
    return min(request_tokens, observed_cached_tokens)


def trace_observed_cached_input_tokens(request: Request) -> int | None:
    """Return trace-provided cached input tokens when the workload supplies them."""
    for key in ("cached_input_tokens", "cache_read_tokens"):
        if key not in request.metadata:
            continue
        value = request.metadata.get(key)
        if value is None or value == "":
            return None
        try:
            return max(int(float(value)), 0)
        except (TypeError, ValueError):
            return None
    return None


def cache_aware_marginal_cost(
    provider: Provider,
    request: Request,
    state: SimulationState,
    *,
    now: float,
    cache_enabled: bool = True,
) -> float:
    """Return real API marginal cost after trace-reported cache discount."""
    cached_tokens = cached_input_tokens(
        provider,
        request,
        state,
        cache_enabled=cache_enabled,
    )
    return provider.marginal_cost_for_request(
        request,
        now,
        cached_input_tokens=cached_tokens,
    )


def _state_cache_enabled(state: SimulationState) -> bool:
    return bool(state.metadata.get("prefix_cache_enabled", False))


__all__ = [
    "cache_aware_marginal_cost",
    "cached_input_tokens",
    "response_tokens_for_request",
    "trace_observed_cached_input_tokens",
]
