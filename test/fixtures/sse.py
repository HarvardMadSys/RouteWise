"""SSE helpers for tests."""

from __future__ import annotations

import json
from typing import Any


def sse_line(payload: dict[str, Any] | str) -> str:
    """Create a single SSE line (data: ...\n\n) as text (no trailing newlines needed by httpx).

    Tests commonly iterate over lines already prefixed with 'data: ' from
    serving.http.stream_post. This helper mirrors that shape.
    """
    body = payload if isinstance(payload, str) else json.dumps(payload)
    return f"data: {body}"


def sse_done() -> str:
    return "data: [DONE]"
