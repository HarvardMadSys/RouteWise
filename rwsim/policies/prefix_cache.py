"""Provider-local prefix-cache cost helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rwsim.world.capacity import ProviderTier

if TYPE_CHECKING:
    from rwsim.engine.state import SimulationState
    from rwsim.schemas import Request
    from rwsim.world.providers import Provider


def request_prefix_id(request: Request) -> str | None:
    """Return the conversation/session key used for prefix-cache locality."""
    for key in ("prefix_id", "sharegpt_conversation_id", "session_id"):
        value = request.metadata.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return None


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
    """Return length-based cached input tokens for one provider/request pair."""
    if not cache_enabled or not _state_cache_enabled(state):
        return 0
    if provider.tier != ProviderTier.S_A:
        return 0
    if provider.cached_input_cost_per_token is None:
        return 0

    prefix_id = request_prefix_id(request)
    if prefix_id is None:
        return 0

    previous_context_tokens = state.provider_prefix_cache.get(provider.name, {}).get(prefix_id, 0)
    request_tokens = max(int(request.request_tokens or 0), 0)
    return min(request_tokens, max(int(previous_context_tokens), 0))


def cache_aware_marginal_cost(
    provider: Provider,
    request: Request,
    state: SimulationState,
    *,
    now: float,
    cache_enabled: bool = True,
) -> float:
    """Return real API marginal cost after provider-local cache discount."""
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


def record_prefix_cache_dispatch(
    provider: Provider,
    request: Request,
    state: SimulationState,
    response_tokens: int | float,
    *,
    cache_enabled: bool = True,
) -> None:
    """Record the context length a provider has seen after a dispatch."""
    if not cache_enabled or not _state_cache_enabled(state):
        return
    if provider.tier != ProviderTier.S_A:
        return
    if provider.cached_input_cost_per_token is None:
        return

    prefix_id = request_prefix_id(request)
    if prefix_id is None:
        return

    context_tokens = max(int(request.request_tokens or 0), 0) + max(int(response_tokens or 0), 0)
    state.provider_prefix_cache.setdefault(provider.name, {})[prefix_id] = context_tokens


def _state_cache_enabled(state: SimulationState) -> bool:
    return bool(state.metadata.get("prefix_cache_enabled", False))


__all__ = [
    "cache_aware_marginal_cost",
    "cached_input_tokens",
    "record_prefix_cache_dispatch",
    "request_prefix_id",
    "response_tokens_for_request",
]
