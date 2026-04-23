from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from fastapi import FastAPI, status
from httpx import ASGITransport, AsyncClient

from routing.executor import RouteExecutor
from serving.adapters.base import BaseAdapter, ModelConfig
from serving.servers.deps import AppServices
from serving.servers.routers import admin, models

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


class _Adapter(BaseAdapter):
    async def chat_completion(self, messages: list[dict[str, Any]], **params) -> dict[str, Any]:
        return self.format_response(content="ok", model=self.config.id)

    async def stream_chat_completion(
        self, messages: list[dict[str, Any]], **params
    ):  # pragma: no cover
        yield self.format_stream_chunk(model=self.config.id, content="ok")


def _cfg(model_id: str) -> ModelConfig:
    return ModelConfig(id=model_id, name=model_id, provider="test", base_url="http://test")


@pytest.fixture
async def admin_app(mock_rate_limiter, mock_db_logger) -> FastAPI:
    router = RouteExecutor()
    router.register_route("canonical-model", [(_Adapter(_cfg("canonical-model")), 1.0)])
    app = FastAPI(title="Admin App")
    app.state.services = AppServices(
        router=router, db_logger=mock_db_logger, rate_limiter=mock_rate_limiter
    )  # type: ignore[attr-defined]
    app.include_router(models.router)
    app.include_router(admin.router)
    return app


@pytest.fixture
async def admin_client(admin_app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=admin_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_admin_rate_limits_status(admin_client: AsyncClient):
    resp = await admin_client.get("/admin/rate-limits/m1")
    assert resp.status_code in (status.HTTP_200_OK, status.HTTP_404_NOT_FOUND)
    if resp.status_code == status.HTTP_200_OK:
        assert resp.json().get("configured") is True


@pytest.mark.asyncio
async def test_admin_rate_limits_metrics(admin_client: AsyncClient):
    resp = await admin_client.get("/admin/rate-limits")
    assert resp.status_code == status.HTTP_200_OK
    assert isinstance(resp.json(), dict)


@pytest.mark.asyncio
async def test_admin_stats_without_db_logger(mock_rate_limiter):
    router = RouteExecutor()
    services = AppServices(router=router, db_logger=None, rate_limiter=mock_rate_limiter)
    app = FastAPI()
    app.state.services = services  # type: ignore[attr-defined]
    app.include_router(admin.router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/admin/stats")
        assert resp.status_code == status.HTTP_200_OK
        assert resp.json().get("error") == "Database logging not configured"


@pytest.mark.asyncio
async def test_admin_reset_circuit_breaker(admin_client: AsyncClient):
    resp = await admin_client.post("/admin/rate-limits/m1/reset")
    assert resp.status_code == status.HTTP_200_OK
    assert "Circuit breaker reset" in resp.json().get("message", "")
