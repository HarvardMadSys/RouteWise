"""Unit tests for admin authentication helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Request

from serving.servers.auth import log_admin_action, verify_admin_token
from serving.servers.routers import admin as admin_router


class _AcquireContext:
    """Async context manager returning the mocked connection."""

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
def mock_request() -> Request:
    request = MagicMock(spec=Request)
    client = MagicMock()
    client.host = "127.0.0.1"
    request.client = client
    return request  # type: ignore[return-value]


@pytest.fixture
def db_logger_with_pool():
    logger = MagicMock()
    connection = AsyncMock()
    connection.execute = AsyncMock()
    connection.fetchrow = AsyncMock()
    pool = MagicMock()
    pool.acquire.return_value = _AcquireContext(connection)
    logger.pool = pool
    return logger, connection


@pytest.mark.asyncio
async def test_verify_admin_token_success(monkeypatch, mock_request):
    monkeypatch.setenv("ADMIN_TOKEN", "test-admin")
    result = await verify_admin_token(
        request=mock_request,
        authorization="Bearer test-admin",
    )
    assert result == "127.0.0.1"


@pytest.mark.asyncio
async def test_verify_admin_token_missing_header(monkeypatch, mock_request):
    monkeypatch.setenv("ADMIN_TOKEN", "another-token")
    with pytest.raises(HTTPException) as exc:
        await verify_admin_token(request=mock_request, authorization=None)
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_admin_token_invalid_token(monkeypatch, mock_request):
    monkeypatch.setenv("ADMIN_TOKEN", "correct-token")
    with pytest.raises(HTTPException) as exc:
        await verify_admin_token(request=mock_request, authorization="Bearer wrong")
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_admin_token_missing_env(monkeypatch, mock_request):
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    with pytest.raises(HTTPException) as exc:
        await verify_admin_token(request=mock_request, authorization="Bearer anything")
    assert exc.value.status_code == 500


def test_serialize_for_audit_handles_decimal_and_datetime():
    now = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    data = {
        "quota": Decimal("123.45"),
        "expires": now,
        "nested": {"values": [Decimal("1.1"), {"ts": now}]},
        "plain": "ok",
    }

    result = admin_router._serialize_for_audit(data)

    assert float(result["quota"]) == pytest.approx(123.45)
    assert result["expires"] == now.isoformat()
    assert float(result["nested"]["values"][0]) == pytest.approx(1.1)
    assert result["nested"]["values"][1]["ts"] == now.isoformat()
    assert result["plain"] == "ok"


@pytest.mark.asyncio
async def test_log_admin_action_writes_entry(monkeypatch, db_logger_with_pool):
    logger, connection = db_logger_with_pool
    await log_admin_action(
        logger,
        admin_ip="10.0.0.1",
        action="create_key",
        target_user_id="user123",
        details={"quota": 100},
    )

    connection.execute.assert_awaited()
    call = connection.execute.await_args
    sql = call.args[0]
    assert "INSERT INTO admin_audit_log" in sql
    assert call.args[1] == "10.0.0.1"
    assert call.args[2] == "create_key"
    assert call.args[3] == "user123"
    assert call.args[4] == '{"quota": 100}'
    assert call.args[5] is True


@pytest.mark.asyncio
async def test_log_admin_action_no_pool(monkeypatch):
    logger = MagicMock()
    logger.pool = None
    # Should not raise even if pool missing
    await log_admin_action(logger, admin_ip="0.0.0.0", action="noop")
