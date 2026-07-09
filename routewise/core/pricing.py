"""Shared token-pricing arithmetic for cache-aware request costs.

Both the simulator (per-token prices on ``Provider``) and the real-eval
harness (per-million prices on ``ProviderSpec``) bill a request as

    input_price * uncached_input + cached_input_price * cached_input
    + output_price * output

with the cached-input count capped by the prompt length. The two entry
points below preserve each side's exact float operation order: the
simulator multiplies per-token prices directly, real-eval multiplies
per-million prices and divides by 1e6 once at the end.
"""

from __future__ import annotations


def capped_cached_input_tokens(
    prompt_tokens: int | float,
    reported_cached_tokens: int | float | None,
) -> int:
    """Clamp a trace/provider-reported cached-input count to the prompt length.

    Returns 0 when no value was reported (cold miss).
    """
    if reported_cached_tokens is None:
        return 0
    capped = max(int(reported_cached_tokens), 0)
    return min(max(int(prompt_tokens), 0), capped)


def cache_discounted_token_cost_usd(
    *,
    prompt_tokens: int | float,
    completion_tokens: int | float,
    cached_input_tokens: int | float,
    input_price_per_token: float,
    cached_input_price_per_token: float,
    output_price_per_token: float,
) -> float:
    """Cache-discounted request cost with per-token prices (simulator order)."""
    cached = max(float(cached_input_tokens), 0.0)
    request_tokens = float(prompt_tokens)
    cached = min(cached, request_tokens)
    uncached = max(request_tokens - cached, 0.0)
    return (
        input_price_per_token * uncached
        + cached_input_price_per_token * cached
        + output_price_per_token * float(completion_tokens)
    )


def cache_discounted_cost_per_million_usd(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_input_tokens: int,
    input_price_per_m: float,
    cached_input_price_per_m: float,
    output_price_per_m: float,
) -> float:
    """Cache-discounted request cost with per-million prices (real-eval order)."""
    cached = min(max(int(cached_input_tokens), 0), max(int(prompt_tokens), 0))
    uncached = max(int(prompt_tokens) - cached, 0)
    return (
        input_price_per_m * uncached
        + cached_input_price_per_m * cached
        + output_price_per_m * max(int(completion_tokens), 0)
    ) / 1_000_000.0


__all__ = [
    "cache_discounted_cost_per_million_usd",
    "cache_discounted_token_cost_usd",
    "capped_cached_input_tokens",
]
