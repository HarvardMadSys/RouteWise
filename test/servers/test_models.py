from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import pytest
from fastapi import FastAPI, status
from httpx import ASGITransport, AsyncClient

from routing.executor import RouteExecutor
from serving.adapters.base import BaseAdapter, ModelConfig
from serving.servers.deps import AppServices
from serving.servers.routers import models

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


class _Adapter(BaseAdapter):
    def __init__(self, cfg: ModelConfig, content: str = "ok"):
        super().__init__(cfg)
        self._content = content

    async def chat_completion(self, messages: list[dict[str, Any]], **params) -> dict[str, Any]:
        return self.format_response(content=self._content, model=self.config.id)

    async def stream_chat_completion(
        self, messages: list[dict[str, Any]], **params
    ):  # pragma: no cover
        yield self.format_stream_chunk(model=self.config.id, content=self._content)


def _cfg(
    *,
    id: str,
    provider: str = "prov",
    context: int = 8192,
    max_out: int = 4096,
    supported: list[str] | None = None,
    tools: bool = False,
    structured: bool = False,
) -> ModelConfig:
    return ModelConfig(
        id=id,
        name=id,
        provider=provider,
        base_url="http://test",
        context_length=context,
        max_output_length=max_out,
        supported_params=supported or ["temperature", "top_p", "max_tokens"],
        supports_tools=tools,
        supports_structured_output=structured,
        input_modalities=["text"],
        output_modalities=["text"],
        quantization="bf16",
    )


@pytest.fixture
async def models_app() -> FastAPI:
    router = RouteExecutor()
    # Two adapters to exercise aggregation
    a1 = _Adapter(
        _cfg(
            id="canonical-model",
            provider="provA",
            context=8192,
            max_out=4096,
            supported=["temperature", "top_p", "max_tokens", "seed"],
            tools=True,
            structured=False,
        )
    )
    a2 = _Adapter(
        _cfg(
            id="canonical-model",
            provider="provB",
            context=4096,
            max_out=2048,
            supported=["temperature", "max_tokens"],
            tools=False,
            structured=True,
        )
    )
    router.register_route("alias-model", [(a1, 0.5), (a2, 0.5)])
    router.register_route("canonical-model", [(a1, 1.0)])

    app = FastAPI(title="Models App")
    app.state.services = AppServices(router=router, db_logger=None, rate_limiter=None)  # type: ignore[attr-defined]
    app.include_router(models.router)
    return app


@pytest.fixture
async def models_client(models_app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=models_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_models_aggregation_and_slug(models_client: AsyncClient):
    resp = await models_client.get("/v1/models")
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()["data"]
    items = [m for m in data if m["id"] == "canonical-model"]
    assert len(items) == 1
    item = items[0]
    assert item["context_length"] == 4096
    assert item["max_output_length"] == 2048
    assert item["supported_sampling_parameters"] == ["max_tokens", "temperature"]
    assert "tools" in item["supported_features"]
    assert "json_mode" in item["supported_features"]
    assert "structured_outputs" in item["supported_features"]
    assert item.get("openrouter", {}).get("slug") in {"alias-model", None}


@pytest.mark.asyncio
async def test_models_empty_routes_returns_empty_list():
    router = RouteExecutor()
    app = FastAPI()
    app.state.services = AppServices(router=router, db_logger=None, rate_limiter=None)  # type: ignore[attr-defined]
    app.include_router(models.router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v1/models")
        assert resp.status_code == status.HTTP_200_OK
        assert resp.json()["data"] == []


@pytest.mark.asyncio
async def test_models_single_adapter_no_aggregation():
    router = RouteExecutor()
    a = _Adapter(_cfg(id="solo", context=1234, max_out=321, supported=["temperature"], tools=True))
    router.register_route("solo", [(a, 1.0)])
    app = FastAPI()
    app.state.services = AppServices(router=router, db_logger=None, rate_limiter=None)  # type: ignore[attr-defined]
    app.include_router(models.router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v1/models")
        data = resp.json()["data"]
        item = next(m for m in data if m["id"] == "solo")
        assert item["context_length"] == 1234
        assert item["max_output_length"] == 321
        assert item["supported_sampling_parameters"] == ["temperature"]
        assert "tools" in item["supported_features"]


@pytest.mark.asyncio
async def test_models_pricing_primary_config_behavior():
    router = RouteExecutor()
    p_primary = {"prompt": "1", "completion": "2"}
    p_secondary = {"prompt": "9", "completion": "9"}
    a1 = _Adapter(_cfg(id="price", provider="p1"))
    a1.config.pricing = dict(p_primary)
    a2 = _Adapter(_cfg(id="price", provider="p2"))
    a2.config.pricing = dict(p_secondary)
    router.register_route("price", [(a1, 0.9), (a2, 0.1)])
    app = FastAPI()
    app.state.services = AppServices(router=router, db_logger=None, rate_limiter=None)  # type: ignore[attr-defined]
    app.include_router(models.router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v1/models")
        data = resp.json()["data"]
        item = next(m for m in data if m["id"] == "price")
        assert item["pricing"] == p_primary


@pytest.mark.asyncio
async def test_concurrent_model_requests_consistent(models_client: AsyncClient):
    tasks = [models_client.get("/v1/models") for _ in range(10)]
    results = await asyncio.gather(*tasks)
    first = results[0].json()
    assert all(r.json() == first for r in results)


@pytest.mark.asyncio
@pytest.mark.perf
async def test_models_endpoint_performance(models_client: AsyncClient):
    start = time.perf_counter()
    resp = await models_client.get("/v1/models")
    elapsed = time.perf_counter() - start
    assert resp.status_code == status.HTTP_200_OK
    assert elapsed < 0.2
