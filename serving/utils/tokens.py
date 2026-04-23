"""Token estimation utilities used across serving components.

This module centralizes token counting logic so that adapters, streaming
formatters, and rate limiters share the same estimation behavior.
"""

from __future__ import annotations

from typing import Any


def _count_with_tiktoken(text: str) -> int:
    """Count tokens for a string using tiktoken when available.

    Falls back to a simple character-based heuristic when tiktoken is not
    installed. The heuristic assumes approximately 4 characters per token.

    Args:
      text: Input string to count.

    Returns:
      Estimated token count (>= 1 for non-empty strings).
    """
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # 4 characters ≈ 1 token (rough heuristic)
        return max(1, len(text) // 4)


def estimate_text_tokens(text: str) -> int:
    """Estimate token count for a plain text string.

    Uses tiktoken when available; otherwise a character-based heuristic.

    Args:
      text: Input string to estimate.

    Returns:
      Estimated token count.
    """
    if not text:
        return 0
    return _count_with_tiktoken(text)


def estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate token count for an OpenAI-style messages list.

    Each message contributes role + content tokens plus a small overhead to
    approximate protocol framing costs. This mirrors common counting approaches.

    Args:
      messages: Chat messages with ``role`` and ``content`` fields.

    Returns:
      Estimated prompt token count.
    """
    if not messages:
        return 0

    overhead_per_message = 4
    overhead_end = 3

    total = 0
    for m in messages:
        role = str(m.get("role", ""))
        content = str(m.get("content", ""))
        total += estimate_text_tokens(role)
        total += estimate_text_tokens(content)
        total += overhead_per_message

    total += overhead_end
    return total


def estimate_total_tokens(messages: list[dict[str, Any]], max_tokens: int | None = None) -> int:
    """Estimate total request tokens (prompt + completion budget).

    Args:
      messages: Chat messages in OpenAI format.
      max_tokens: Optional completion budget; defaults to 500 if None.

    Returns:
      Estimated total tokens for the request.
    """
    prompt_tokens = estimate_prompt_tokens(messages)
    completion_budget = max_tokens if max_tokens is not None else 500
    return prompt_tokens + int(completion_budget)


__all__ = [
    "estimate_prompt_tokens",
    "estimate_text_tokens",
    "estimate_total_tokens",
]
