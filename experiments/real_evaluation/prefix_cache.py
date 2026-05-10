"""Prefix-cache helpers for real-eval routing decisions.

These helpers model cache hits only for route-time cost estimation. They do
not change provider-reported billed cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from experiments.real_evaluation.transports import compute_request_cost_usd

if TYPE_CHECKING:
    from experiments.real_evaluation.inventory import ProviderSpec

ProviderPrefixCache = dict[str, dict[str, int]]


def cached_input_tokens(
    *,
    provider_name: str,
    prefix_id: str | None,
    prompt_tokens: int,
    provider_prefix_cache: ProviderPrefixCache,
) -> int:
    """Return the length-based prefix-cache hit for one provider/session."""
    if prefix_id is None:
        return 0
    previous_context = provider_prefix_cache.get(provider_name, {}).get(prefix_id, 0)
    return min(max(int(prompt_tokens), 0), max(int(previous_context), 0))


def cache_aware_request_cost_usd(
    spec: ProviderSpec,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    prefix_id: str | None,
    provider_prefix_cache: ProviderPrefixCache,
    enabled: bool,
) -> float:
    """Return routing-estimated API cost, optionally discounting cached input."""
    if spec.tier != "api":
        return 0.0
    if not enabled or spec.cached_input_price_per_m is None:
        return compute_request_cost_usd(
            prompt_tokens,
            completion_tokens,
            spec.input_price_per_m,
            spec.output_price_per_m,
        )

    cached = cached_input_tokens(
        provider_name=spec.name,
        prefix_id=prefix_id,
        prompt_tokens=prompt_tokens,
        provider_prefix_cache=provider_prefix_cache,
    )
    uncached = max(int(prompt_tokens) - cached, 0)
    return (
        spec.input_price_per_m * uncached
        + spec.cached_input_price_per_m * cached
        + spec.output_price_per_m * max(int(completion_tokens), 0)
    ) / 1_000_000.0


def record_prefix_cache_dispatch(
    *,
    provider_name: str,
    prefix_id: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    provider_prefix_cache: ProviderPrefixCache,
) -> None:
    """Record the context length available after a dispatched request."""
    if prefix_id is None:
        return
    prompt_tokens = max(int(prompt_tokens), 0)
    completion_tokens = max(int(completion_tokens), 0)
    if prompt_tokens <= 0 and completion_tokens <= 0:
        return
    provider_prefix_cache.setdefault(provider_name, {})[prefix_id] = (
        prompt_tokens + completion_tokens
    )


__all__ = [
    "ProviderPrefixCache",
    "cache_aware_request_cost_usd",
    "cached_input_tokens",
    "record_prefix_cache_dispatch",
]
