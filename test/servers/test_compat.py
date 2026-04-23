from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from fastapi import FastAPI, status
from httpx import ASGITransport, AsyncClient

from routing.executor import RouteExecutor
from serving.adapters.base import BaseAdapter, ModelConfig
from serving.servers.auth import verify_api_key
from serving.servers.deps import AppServices
from serving.servers.middleware.error import install_error_handlers
from serving.servers.routers import compat, completions

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


class _Adapter(BaseAdapter):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.last_params: dict[str, Any] | None = None
        self.last_messages: list[dict[str, Any]] | None = None

    async def chat_completion(self, messages: list[dict[str, Any]], **params) -> dict[str, Any]:
        self.last_params = dict(params)
        self.last_messages = list(messages)
        return self.format_response(content="ok", model=self.config.id)

    async def stream_chat_completion(
        self, messages: list[dict[str, Any]], **params
    ):  # pragma: no cover
        yield self.format_stream_chunk(model=self.config.id, content="ok")


def _cfg(model_id: str) -> ModelConfig:
    return ModelConfig(id=model_id, name=model_id, provider="test", base_url="http://test")


@pytest.fixture
async def compat_app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.setenv("USER_AUTH_ENABLED", "0")
    router = RouteExecutor()
    t = _Adapter(_cfg("trk"))
    router.register_route("trk", [(t, 1.0)])

    app = FastAPI()
    install_error_handlers(app)
    app.state.services = AppServices(router=router, db_logger=None, rate_limiter=None)  # type: ignore[attr-defined]
    app.include_router(compat.router)
    app.include_router(completions.router)

    # Skip database-backed auth in unit tests; compat routes only need routing glue.
    async def _anon_user():
        return {
            "user_id": "anonymous",
            "user_name": None,
            "tier": "free",
            "authenticated": False,
            "quota_remaining_cost_usd": float("inf"),
        }

    app.dependency_overrides[verify_api_key] = _anon_user
    return app


@pytest.fixture
async def compat_client(compat_app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=compat_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_compat_single_completion_endpoint(compat_client: AsyncClient):
    resp = await compat_client.post(
        "/completion",
        json={"model": "trk", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == status.HTTP_200_OK
    assert resp.json()["choices"][0]["message"]["content"] == "ok"


@pytest.mark.asyncio
async def test_legacy_completions_prompt_conversion(compat_client: AsyncClient):
    resp = await compat_client.post("/v1/completions", json={"model": "trk", "prompt": "hello"})
    assert resp.status_code == status.HTTP_200_OK
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "ok"


@pytest.mark.asyncio
async def test_legacy_completions_preserves_parameters(compat_app: FastAPI):
    # Extract the tracking adapter for assertions
    router: RouteExecutor = compat_app.state.services.router  # type: ignore[attr-defined]
    adapter = router.routes["trk"].adapters[0][0]

    transport = ASGITransport(app=compat_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/completions",
            json={
                "model": "trk",
                "prompt": "hello",
                "temperature": 0.5,
                "max_tokens": 100,
                "top_p": 0.9,
            },
        )
        assert resp.status_code == status.HTTP_200_OK
        assert adapter.last_params is not None and adapter.last_messages is not None
        assert adapter.last_params["temperature"] == 0.5
        assert adapter.last_params["max_tokens"] == 100
        assert adapter.last_params["top_p"] == 0.9
        assert adapter.last_messages[0]["content"] == "hello"
