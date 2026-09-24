"""Prefix-cache helpers for real-eval routing decisions.

The cached-input-token count comes directly from the replayed trace
(``cache_read_tokens`` / ``cached_input_tokens`` in the freeinference
dataset). When the trace does not report a value, the request is treated
as a cold miss. We do not synthesize cache locality from a provider-local
prefix model, which was too aggressive at predicting hits.

These helpers model cache hits only for route-time cost estimation; they
do not change provider-reported billed cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from experiments.real_evaluation.transports import compute_request_cost_usd

if TYPE_CHECKING:
    from experiments.real_evaluation.inventory import ProviderSpec


def cached_input_tokens(
    *,
    prompt_tokens: int,
    trace_cached_input_tokens: int | None,
) -> int:
    """Return the length-based cached-input-token count for one request.

    Returns 0 when the trace does not report a value; otherwise caps the
    reported value by the prompt length.
    """
    if trace_cached_input_tokens is None:
        return 0
    capped = max(int(trace_cached_input_tokens), 0)
    return min(max(int(prompt_tokens), 0), capped)


def cache_aware_request_cost_usd(
    spec: ProviderSpec,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    trace_cached_input_tokens: int | None,
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
        prompt_tokens=prompt_tokens,
        trace_cached_input_tokens=trace_cached_input_tokens,
    )
    uncached = max(int(prompt_tokens) - cached, 0)
    return (
        spec.input_price_per_m * uncached
        + spec.cached_input_price_per_m * cached
        + spec.output_price_per_m * max(int(completion_tokens), 0)
    ) / 1_000_000.0


__all__ = [
    "cache_aware_request_cost_usd",
    "cached_input_tokens",
]
