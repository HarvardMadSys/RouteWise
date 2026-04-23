from __future__ import annotations

import json


def assert_sse_format(line: str) -> None:
    """Verify a line follows OpenAI-style SSE format (data: <json or [DONE]>)."""
    assert line.startswith("data: ")
    body = line[6:].strip()
    if body != "[DONE]":
        json.loads(body)


def assert_token_estimate_reasonable(actual: int, text: str) -> None:
    """Token count within a broad reasonable bound based on whitespace tokens."""
    words = max(1, len(text.split()))
    min_expected = words // 2
    max_expected = words * 2
    assert min_expected <= actual <= max_expected
