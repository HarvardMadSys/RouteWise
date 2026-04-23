"""Tests for admin API key management endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from serving.servers.deps import AppServices
from serving.servers.routers import admin as admin_router


class _AcquireContext:
    """Async context manager for mocked connections."""

    def __init__(self, connection: AsyncMock) -> None:
        self._connection = connection

    async def __aenter__(self) -> AsyncMock:
        return self._connection

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        return None


@pytest.fixture
def mocked_db_logger():
    logger = MagicMock()
    connection = AsyncMock()
    connection.fetch = AsyncMock()
    connection.fetchrow = AsyncMock()
    connection.execute = AsyncMock()
    pool = MagicMock()
    pool.acquire.return_value = _AcquireContext(connection)
    logger.pool = pool
    return logger, connection


@pytest.fixture
async def admin_client(monkeypatch, mocked_db_logger):
    logger, connection = mocked_db_logger
    app = FastAPI(title="Admin API Test")

    services = AppServices(
        router=MagicMock(),
        db_logger=logger,
        rate_limiter=None,
        routing_manager=None,
    )
    app.state.services = services  # type: ignore[attr-defined]
    app.include_router(admin_router.router)

    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://test")

    mock_log_action = AsyncMock()
    monkeypatch.setattr("serving.servers.routers.admin.log_admin_action", mock_log_action)
    monkeypatch.setenv("ADMIN_TOKEN", "test-admin")
    monkeypatch.setenv("API_KEY_SECRET", "unit-test-secret")

    try:
        yield client, connection, mock_log_action
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_create_api_key_success(admin_client, monkeypatch):
    client, connection, log_action = admin_client
    created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    connection.fetchrow.reset_mock()
    connection.fetchrow.side_effect = [
        None,
        {"id": 1, "created_at": created_at},
    ]

    monkeypatch.setattr("serving.servers.routers.admin.generate_api_key", lambda: "hyi-fixed-key")
    monkeypatch.setattr("serving.servers.routers.admin.hash_api_key", lambda _: "hashed-key")

    response = await client.post(
        "/admin/api-keys",
        headers={"Authorization": "Bearer test-admin"},
        json={"user_id": "alice", "tier": "pro", "quota_daily_cost_usd": 500},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["api_key"] == "hyi-fixed-key"
    assert body["key_prefix"] == "hyi-fixed-key"[:12]
    connection.fetchrow.assert_awaited()
    log_action.assert_awaited()


@pytest.mark.asyncio
async def test_create_api_key_conflict(admin_client, monkeypatch):
    client, connection, log_action = admin_client
    connection.fetchrow.reset_mock()
    connection.fetchrow.side_effect = [{"user_id": "alice"}]

    response = await client.post(
        "/admin/api-keys",
        headers={"Authorization": "Bearer test-admin"},
        json={"user_id": "alice"},
    )

    assert response.status_code == 409
    log_action.assert_awaited()
    call = log_action.await_args
    assert call.kwargs.get("success") is False


@pytest.mark.asyncio
async def test_create_api_key_requires_token(admin_client):
    client, _connection, _log_action = admin_client
    response = await client.post("/admin/api-keys", json={"user_id": "alice"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_list_api_keys_batches_usage(admin_client):
    client, connection, _log_action = admin_client
    connection.fetchrow.reset_mock()
    connection.fetchrow.return_value = {"total": 1}
    created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)

    connection.fetch.reset_mock()
    connection.fetch.side_effect = [
        [
            {
                "user_id": "alice",
                "user_name": "Alice",
                "key_prefix": "hyi-alice",
                "tier": "pro",
                "status": "active",
                "quota_daily_cost_usd": Decimal("500"),
                "quota_monthly_cost_usd": Decimal("10000"),
                "created_at": created_at,
                "last_used_at": None,
                "expires_at": None,
                "notes": None,
            }
        ],
        [{"user_id": "alice", "cost_spent": Decimal("12.34")}],
        [{"user_id": "alice", "cost_spent": Decimal("23.45")}],
    ]

    response = await client.get(
        "/admin/api-keys",
        headers={"Authorization": "Bearer test-admin"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["keys"]) == 1
    assert float(data["keys"][0]["usage_today_usd"]) == pytest.approx(12.34)
    assert float(data["keys"][0]["usage_month_usd"]) == pytest.approx(23.45)
    assert connection.fetch.await_count == 3


@pytest.mark.asyncio
async def test_get_api_key_detail_success(admin_client):
    client, connection, _log_action = admin_client
    created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    usage_today = {"cost_spent": Decimal("20.0"), "requests": 5}
    usage_month = {"cost_spent": Decimal("50.0"), "requests": 12}
    connection.fetchrow.reset_mock()
    connection.fetchrow.side_effect = [
        {
            "user_id": "alice",
            "user_name": "Alice",
            "key_prefix": "hyi-alice",
            "tier": "pro",
            "status": "active",
            "quota_daily_cost_usd": Decimal("100"),
            "quota_monthly_cost_usd": Decimal("500"),
            "created_at": created_at,
            "last_used_at": None,
            "expires_at": None,
            "notes": None,
            "metadata": None,
        },
        usage_today,
        usage_month,
        {"last_request_at": created_at},
    ]
    connection.fetch.reset_mock()
    connection.fetch.side_effect = [[{"model_id": "model-a"}]]

    response = await client.get(
        "/admin/api-keys/alice",
        headers={"Authorization": "Bearer test-admin"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == "alice"
    assert payload["usage"]["today"]["quota_remaining_usd"] == 80.0
    assert payload["usage"]["models_used"] == ["model-a"]


@pytest.mark.asyncio
async def test_update_api_key_success(admin_client):
    client, connection, log_action = admin_client
    connection.fetchrow.reset_mock()
    connection.fetchrow.side_effect = [{"user_id": "alice"}]

    response = await client.patch(
        "/admin/api-keys/alice",
        headers={"Authorization": "Bearer test-admin"},
        json={"tier": "enterprise", "quota_daily_cost_usd": 200},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["updated_fields"] == ["tier", "quota_daily_cost_usd"]
    log_action.assert_awaited()


@pytest.mark.asyncio
async def test_update_api_key_no_fields(admin_client):
    client, _connection, _log_action = admin_client
    response = await client.patch(
        "/admin/api-keys/alice",
        headers={"Authorization": "Bearer test-admin"},
        json={},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_update_api_key_not_found(admin_client):
    client, connection, _log_action = admin_client
    connection.fetchrow.reset_mock()
    connection.fetchrow.side_effect = [None]

    response = await client.patch(
        "/admin/api-keys/missing",
        headers={"Authorization": "Bearer test-admin"},
        json={"tier": "pro"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_revoke_api_key_soft_delete(admin_client):
    client, connection, log_action = admin_client
    connection.fetchrow.reset_mock()
    connection.fetchrow.side_effect = [{"user_id": "alice"}]

    response = await client.delete(
        "/admin/api-keys/alice",
        headers={"Authorization": "Bearer test-admin"},
    )

    assert response.status_code == 200
    connection.execute.assert_awaited()
    log_action.assert_awaited()


@pytest.mark.asyncio
async def test_revoke_api_key_hard_delete(admin_client):
    client, connection, _log_action = admin_client
    connection.fetchrow.reset_mock()
    connection.fetchrow.side_effect = [{"user_id": "alice"}]

    response = await client.delete(
        "/admin/api-keys/alice",
        headers={"Authorization": "Bearer test-admin"},
        params={"hard_delete": "true"},
    )

    assert response.status_code == 200
    connection.execute.assert_awaited()


@pytest.mark.asyncio
async def test_regenerate_api_key_success(admin_client, monkeypatch):
    client, connection, log_action = admin_client
    connection.fetchrow.reset_mock()
    connection.fetchrow.side_effect = [{"key_prefix": "hyi-old"}]
    monkeypatch.setattr("serving.servers.routers.admin.generate_api_key", lambda: "hyi-new-key")
    monkeypatch.setattr("serving.servers.routers.admin.hash_api_key", lambda _: "new-hash")

    response = await client.post(
        "/admin/api-keys/alice/regenerate",
        headers={"Authorization": "Bearer test-admin"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["api_key"] == "hyi-new-key"
    assert payload["old_key_prefix"] == "hyi-old"
    connection.execute.assert_awaited()
    log_action.assert_awaited()
