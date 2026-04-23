"""HTTP client mocks and helpers for tests.

Provides a controllable replacement for AsyncHTTPClient.shared() that
allows tests to predefine JSON responses and SSE/NDJSON streaming lines
without hitting the network.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from serving.http import AsyncHTTPClient

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class _MockHTTPClient:
    """Simple mock matching AsyncHTTPClient public surface used in tests."""

    def __init__(self) -> None:
        self.json_responses: dict[str, Any] = {}
        self.stream_responses: dict[str, list[str]] = {}
        self.calls: list[tuple[str, str]] = []

    def set_json(self, url: str, payload: Any) -> None:
        self.json_responses[url] = payload

    def set_stream(self, url: str, lines: list[str]) -> None:
        # Lines should already be SSE/NDJSON formatted as desired by the test.
        self.stream_responses[url] = lines

    async def json_post(
        self,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout=None,
    ) -> Any:
        self.calls.append(("POST", url))
        return self.json_responses[url]

    async def json_get(
        self, url: str, *, headers: dict[str, str] | None = None, timeout=None
    ) -> Any:
        self.calls.append(("GET", url))
        return self.json_responses[url]

    async def stream_post(
        self,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout=None,
        mode: str = "sse",
    ) -> AsyncIterator[str]:
        self.calls.append(("STREAM", url))
        for line in self.stream_responses[url]:
            yield line


@pytest.fixture
def mock_http_client(monkeypatch) -> _MockHTTPClient:
    """Monkeypatch AsyncHTTPClient.shared() to return a controllable mock."""

    mock = _MockHTTPClient()
    monkeypatch.setattr(AsyncHTTPClient, "shared", classmethod(lambda cls: mock))
    return mock
