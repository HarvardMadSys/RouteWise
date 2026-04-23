from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient

from routing.executor import RouteExecutor
from serving.adapters.base import BaseAdapter, ModelConfig
from serving.servers.deps import AppServices
from serving.servers.middleware.error import install_error_handlers
from serving.servers.routers import admin, models


class _EchoAdapter(BaseAdapter):
    async def chat_completion(self, messages: list[dict[str, Any]], **params) -> dict[str, Any]:
        return self.format_response(content="ok", model=self.config.id)

    async def stream_chat_completion(
        self, messages: list[dict[str, Any]], **params
    ):  # pragma: no cover
        yield self.format_stream_chunk(model=self.config.id, content="ok")


def _cfg(model_id: str) -> ModelConfig:
    return ModelConfig(
        id=model_id,
        name=model_id,
        provider="test",
        base_url="http://test",
    )


@pytest.mark.asyncio
async def test_admin_routing_status_with_and_without_manager():
    # Build router with one model
    router = RouteExecutor()
    router.register_route("m1", [(_EchoAdapter(_cfg("m1")), 1.0)])

    # Case 1: with routing manager
    class _Mgr:
        def get_status(self) -> dict[str, Any]:
            return {
                "loaded": True,
                "strategy": "fixed",
                "deployments": {"local": 1, "remote": 0},
            }

    app = FastAPI()
    app.state.services = AppServices(
        router=router, db_logger=None, rate_limiter=None, routing_manager=_Mgr()
    )  # type: ignore[attr-defined]
    app.include_router(models.router)
    app.include_router(admin.router)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/admin/routing")
        assert resp.status_code == 200
        data = resp.json()
        assert "routes" in data and "manager_status" in data
        assert "m1" in data["routes"]

    # Case 2: without routing manager
    app2 = FastAPI()
    app2.state.services = AppServices(
        router=router, db_logger=None, rate_limiter=None, routing_manager=None
    )  # type: ignore[attr-defined]
    app2.include_router(models.router)
    app2.include_router(admin.router)

    transport2 = ASGITransport(app=app2)
    async with AsyncClient(transport=transport2, base_url="http://test") as client:
        resp = await client.get("/admin/routing")
        assert resp.status_code == 200
        data = resp.json()
        assert "routes" in data and "manager_status" not in data


@pytest.mark.asyncio
async def test_error_middleware_integration_http_exception_and_generic():
    app = FastAPI()

    @app.get("/raise-http")
    def raise_http():
        # Will be wrapped into error payload with matching status code
        raise HTTPException(status_code=418, detail="teapot")

    @app.get("/raise-any")
    def raise_any():
        raise RuntimeError("boom")

    # Install error handlers after routes are defined
    install_error_handlers(app)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r1 = await client.get("/raise-http")
        assert r1.status_code == 418
        body1 = r1.json()
        assert body1["error"]["message"].lower().find("teapot") >= 0
        assert body1["error"]["code"] == 418

        r2 = await client.get("/raise-any")
        assert r2.status_code == 500
        body2 = r2.json()
        assert body2["error"]["message"].lower().find("boom") >= 0
        assert body2["error"]["code"] == 500
