"""Integration tests for /v1/chat/completions aligned with current server.

Covers non-streaming and streaming flows, model-not-found, invalid payload,
fallback behavior, and basic rate-limit rejection using injected services.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from fastapi import FastAPI, status
from httpx import ASGITransport, AsyncClient

from routing.executor import RouteExecutor
from serving.adapters.base import BaseAdapter, ModelConfig
from serving.servers.deps import AppServices
from serving.servers.middleware.error import install_error_handlers
from serving.servers.routers import compat, completions, health, models
from serving.stream import done_sentinel, make_final_usage_chunk

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


class DummyAdapter(BaseAdapter):
    async def chat_completion(self, messages: list[dict[str, Any]], **params) -> dict[str, Any]:
        content = params.get("content", "Test response")
        resp = self.format_response(content=content, model=self.config.id)
        return resp

    async def stream_chat_completion(
        self, messages: list[dict[str, Any]], **params
    ) -> AsyncGenerator[str, None]:
        # Emit role and content chunks then final usage
        yield self.format_stream_chunk(model=self.config.id, content="Test ")
        yield self.format_stream_chunk(model=self.config.id, content="response")
        yield make_final_usage_chunk(
            model=self.config.id, messages=messages, total_content="Test response"
        )
        yield done_sentinel()


class FailingAdapter(BaseAdapter):
    async def chat_completion(self, messages: list[dict[str, Any]], **params) -> dict[str, Any]:
        raise RuntimeError("Primary adapter failed")

    async def stream_chat_completion(
        self, messages: list[dict[str, Any]], **params
    ) -> AsyncGenerator[str, None]:
        raise RuntimeError("Primary adapter failed")


def _mk_cfg(model_id: str) -> ModelConfig:
    return ModelConfig(
        id=model_id,
        name=model_id,
        provider="test",
        base_url="http://test",
        context_length=8192,
        max_output_length=4096,
        supported_params=["temperature", "top_p", "max_tokens"],
    )


@pytest.fixture
async def completions_app(monkeypatch, mock_rate_limiter, mock_db_logger) -> FastAPI:
    """Create a FastAPI app with completions/compat routers and injected services.

    Note: We set app.state.services directly to avoid relying on lifespan handling
    in the test transport.
    """

    # Disable auth for routing-focused tests to avoid auth noise.
    monkeypatch.setenv("USER_AUTH_ENABLED", "0")

    router = RouteExecutor()
    router.register_route("gpt-4", [(DummyAdapter(_mk_cfg("gpt-4")), 1.0)])

    app = FastAPI(title="Test Completions App")
    # Inject services on state directly (no lifespan dependency in tests)
    app.state.services = AppServices(  # type: ignore[attr-defined]
        router=router, db_logger=mock_db_logger, rate_limiter=mock_rate_limiter
    )

    install_error_handlers(app)
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(completions.router)
    app.include_router(compat.router)
    return app


@pytest.fixture
async def completions_client(completions_app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=completions_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_non_streaming_basic(completions_client: AsyncClient):
    resp = await completions_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert resp.status_code == status.HTTP_200_OK
    body = resp.json()
    assert body["model"] == "gpt-4"
    assert body["choices"][0]["message"]["content"] == "Test response"


@pytest.mark.asyncio
async def test_streaming_sse_format(completions_client: AsyncClient):
    async with completions_client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
    ) as resp:
        assert resp.status_code == status.HTTP_200_OK
        lines: list[str] = []
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                lines.append(line)
        assert any(line == "data: [DONE]" for line in lines)
        # Concatenate content chunks (ignore usage chunk)
        content = "".join(
            json.loads(line[6:])["choices"][0]["delta"].get("content", "")
            for line in lines
            if line != "data: [DONE]" and line != "data: {}"
        )
        assert content == "Test response"


@pytest.mark.asyncio
async def test_model_not_found_returns_404(completions_client: AsyncClient):
    resp = await completions_client.post(
        "/v1/chat/completions",
        json={"model": "unknown", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert resp.status_code == status.HTTP_404_NOT_FOUND
    data = resp.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_invalid_request_returns_400(completions_client: AsyncClient):
    # Missing required fields
    resp = await completions_client.post("/v1/chat/completions", json={"model": "gpt-4"})
    assert resp.status_code == status.HTTP_400_BAD_REQUEST


@pytest.mark.asyncio
async def test_fallback_on_primary_failure(completions_app: FastAPI, mock_rate_limiter):
    # Rebuild router with failing primary and working fallback
    router = RouteExecutor()
    router.register_route(
        "gpt-4", [(FailingAdapter(_mk_cfg("gpt-4")), 0.9), (DummyAdapter(_mk_cfg("gpt-4")), 0.1)]
    )

    services = AppServices(router=router, db_logger=None, rate_limiter=mock_rate_limiter)

    app = FastAPI(title="Fallback App")
    app.state.services = services  # type: ignore[attr-defined]
    install_error_handlers(app)
    app.include_router(completions.router)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()
        # When fallback occurs, router strips _routing before returning to user in non-streaming
        # path; our server keeps _routing only internally for db logging. We validate content.
        assert data["choices"][0]["message"]["content"] == "Test response"


@pytest.mark.asyncio
async def test_rate_limit_rejection(completions_app: FastAPI, mock_rate_limiter):
    # Configure limiter to reject
    async def reject(model_id: str, messages, max_tokens=None, priority=0, timeout=30.0):
        return False, {"error": "Rate limit exceeded", "retry_after": 1, "queue_size": 0}

    mock_rate_limiter.acquire_tokens.side_effect = reject  # type: ignore[attr-defined]

    transport = ASGITransport(app=completions_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert resp.headers.get("X-RateLimit-RetryAfter") == "1"
        data = resp.json()
        assert data["error"]["type"] == "rate_limit_exceeded"
