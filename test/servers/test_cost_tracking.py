"""Integration tests for cost logging and rate limiter priority."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from routing.executor import RouteExecutor
from serving.adapters.base import BaseAdapter, ModelConfig, UsageInfo
from serving.servers.auth import verify_api_key
from serving.servers.deps import AppServices
from serving.servers.middleware.error import install_error_handlers
from serving.servers.routers import completions

SAMPLE_PRICING = {
    "prompt": "0.15",
    "completion": "1.25",
    "input_cache_reads": "0.01",
    "input_cache_writes": "0.02",
    "image": "0",
    "request": "0",
}


class TrackingAdapter(BaseAdapter):
    """Adapter that returns deterministic responses for testing."""

    def __init__(self, config: ModelConfig, stream_usage: dict[str, int]) -> None:
        super().__init__(config)
        self._stream_usage = stream_usage

    async def chat_completion(self, messages: list[dict[str, Any]], **params) -> dict[str, Any]:
        usage = UsageInfo(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cache_read_tokens=20,
        )
        return self.format_response(
            content="Test response",
            model=self.config.id,
            usage=usage,
        )

    async def stream_chat_completion(self, messages: list[dict[str, Any]], **params) -> Any:
        yield self.format_stream_chunk("Streamed", self.config.id)
        chunk = {
            "id": "stream-usage",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": self.config.id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": self._stream_usage,
            "_routing": {
                "provider": self.config.provider,
                "base_url": self.config.base_url,
            },
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"


@pytest.fixture
async def tracking_app(monkeypatch, mock_db_logger, mock_rate_limiter) -> FastAPI:
    # Disable auth for cost tracking tests; auth has independent coverage.
    monkeypatch.setenv("USER_AUTH_ENABLED", "0")

    router = RouteExecutor()
    config = ModelConfig(
        id="tracked-model",
        name="Tracked Model",
        provider="gemini",
        base_url="https://gemini.test",
        pricing=SAMPLE_PRICING,
    )
    adapter = TrackingAdapter(
        config=config,
        stream_usage={
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "total_tokens": 150,
        },
    )
    router.register_route("tracked-model", [(adapter, 1.0)])

    app = FastAPI(title="Cost Tracking App")
    app.state.services = AppServices(  # type: ignore[attr-defined]
        router=router,
        db_logger=mock_db_logger,
        rate_limiter=mock_rate_limiter,
    )

    install_error_handlers(app)
    app.include_router(completions.router)
    yield app


@pytest.fixture
async def tracking_client(tracking_app: FastAPI):
    transport = ASGITransport(app=tracking_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, tracking_app


@pytest.mark.asyncio
async def test_non_streaming_logs_pricing_and_usage(tracking_client, mock_db_logger):
    client, _app = tracking_client
    mock_db_logger.log_request.reset_mock()

    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "tracked-model",
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )

    assert response.status_code == 200
    mock_db_logger.log_request.assert_awaited_once()
    call = mock_db_logger.log_request.await_args
    kwargs = call.kwargs
    assert kwargs["provider"] == "gemini"
    assert kwargs["pricing"]["prompt"] == "0.15"
    assert kwargs["usage"]["prompt_tokens"] == 100
    assert kwargs["usage"]["completion_tokens"] == 50


@pytest.mark.asyncio
async def test_streaming_logs_usage(tracking_client, mock_db_logger):
    client, _app = tracking_client
    mock_db_logger.log_request.reset_mock()

    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "tracked-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        async for _line in resp.aiter_lines():
            pass

    mock_db_logger.log_request.assert_awaited_once()
    kwargs = mock_db_logger.log_request.await_args.kwargs
    assert kwargs["usage"]["completion_tokens"] == 30
    assert kwargs["usage"]["prompt_tokens"] == 120


@pytest.mark.asyncio
async def test_authenticated_user_priority(tracking_client, mock_rate_limiter):
    client, app = tracking_client
    mock_rate_limiter.acquire_tokens.reset_mock()

    async def override_verify_api_key():
        return {"user_id": "user-a", "authenticated": True, "tier": "pro"}

    app.dependency_overrides[verify_api_key] = override_verify_api_key
    try:
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "tracked-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
            headers={"Authorization": "Bearer hyi-test"},
        )
    finally:
        app.dependency_overrides.pop(verify_api_key, None)

    mock_rate_limiter.acquire_tokens.assert_awaited()
    call = mock_rate_limiter.acquire_tokens.await_args
    assert call.kwargs["priority"] == 1


@pytest.mark.asyncio
async def test_anonymous_user_priority_zero(tracking_client, mock_rate_limiter):
    client, app = tracking_client
    mock_rate_limiter.acquire_tokens.reset_mock()

    async def override_verify_api_key():
        return {"user_id": "anon", "authenticated": False, "tier": "free"}

    app.dependency_overrides[verify_api_key] = override_verify_api_key
    try:
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "tracked-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
    finally:
        app.dependency_overrides.pop(verify_api_key, None)

    mock_rate_limiter.acquire_tokens.assert_awaited()
    call = mock_rate_limiter.acquire_tokens.await_args
    assert call.kwargs["priority"] == 0
