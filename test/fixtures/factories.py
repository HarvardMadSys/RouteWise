from __future__ import annotations

from typing import Any


def create_chat_request(**overrides: Any) -> dict[str, Any]:
    """Create a valid ChatCompletionRequest-like dict with defaults."""
    defaults: dict[str, Any] = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "test"}],
        "stream": False,
    }
    result = dict(defaults)
    result.update(overrides)
    return result
