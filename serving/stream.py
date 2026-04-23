"""SSE stream formatting helpers shared across adapters.

These helpers generate OpenAI-compatible SSE chunks for streaming
chat completions. Keeping this logic in one place ensures adapters
emit consistent formats and minimizes duplication.
"""

from __future__ import annotations

import json
import time
from typing import Any

from serving.utils.tokens import estimate_prompt_tokens, estimate_text_tokens


def make_stream_chunk(
    *,
    model: str,
    content: str = "",
    finish_reason: str | None = None,
    role: str | None = None,
) -> str:
    """Create a single SSE data line for a chat.completion.chunk.

    Args:
        model: Model identifier to emit in the chunk.
        content: Delta content for this chunk; empty for terminal chunks.
        finish_reason: When provided, marks the final chunk finish reason.
        role: Role for the first chunk (e.g., "assistant"). OpenAI spec requires the first chunk to include role.
    Returns:
        A string representing one SSE line with a trailing blank line.
    """
    delta: dict[str, Any] = {}
    if role:
        delta["role"] = role
    if content:
        delta["content"] = content

    # If delta is empty and no finish_reason, at least include empty content
    if not delta and not finish_reason:
        delta = {"content": content}

    chunk: dict[str, Any] = {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def done_sentinel() -> str:
    """Return the SSE stream termination sentinel line."""
    return "data: [DONE]\n\n"


def make_final_usage_chunk(
    *,
    model: str,
    messages: list[dict[str, Any]],
    total_content: str,
    prompt_tokens_override: int | None = None,
    completion_tokens_override: int | None = None,
    finish_reason: str = "stop",
    provider: str | None = None,
    base_url: str | None = None,
    reasoning_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> str:
    """Create the final SSE chunk carrying usage metrics.

    Computes prompt/completion tokens if not provided by the caller and returns
    a properly formatted OpenAI-compatible SSE data line containing the usage
    object and a terminal choice with the finish_reason.

    Args:
        model: Model identifier to emit in the chunk.
        messages: Chat messages for token estimation.
        total_content: Generated content for token estimation.
        prompt_tokens_override: Override for prompt tokens (if available from provider).
        completion_tokens_override: Override for completion tokens (if available from provider).
        finish_reason: Finish reason for the completion.
        provider: Provider name for routing info (enables cost calculation).
        base_url: Provider base URL for routing info (enables cost calculation).
        reasoning_tokens: Reasoning tokens count (for models like GPT-5/o1/DeepSeek-R1).
        cache_read_tokens: Tokens read from cache (for cost calculation).
        cache_write_tokens: Tokens written to cache (for cost calculation).
    """
    prompt_tokens = (
        int(prompt_tokens_override)
        if prompt_tokens_override is not None and prompt_tokens_override > 0
        else int(estimate_prompt_tokens(messages))
    )
    completion_tokens = (
        int(completion_tokens_override)
        if completion_tokens_override is not None and completion_tokens_override > 0
        else int(estimate_text_tokens(total_content))
    )

    usage: dict[str, int] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

    # Add optional token fields if present
    if reasoning_tokens > 0:
        usage["reasoning_tokens"] = reasoning_tokens
    if cache_read_tokens > 0:
        usage["cache_read_tokens"] = cache_read_tokens
    if cache_write_tokens > 0:
        usage["cache_write_tokens"] = cache_write_tokens

    chunk: dict[str, Any] = {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        "usage": usage,
    }

    # Include routing info for cost calculation if provider is specified
    if provider:
        chunk["_routing"] = {
            "provider": provider,
            "base_url": base_url,
        }

    return f"data: {json.dumps(chunk)}\n\n"


def make_role_chunk(*, model: str) -> str:
    """Create the initial SSE chunk specifying assistant role.

    Many OpenAI-compatible clients expect the first streaming chunk to include
    a delta with "role": "assistant" to mark the beginning of the AI message.
    This helper emits that role-only chunk without content.
    """
    chunk: dict[str, Any] = {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"
