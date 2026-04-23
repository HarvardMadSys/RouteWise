"""Unit tests for verify_api_key authentication and quota logic."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Request

from serving.servers.auth import hash_api_key, verify_api_key
from serving.storage.database import DatabaseLogger


class _AcquireContext:
    """Simple async context manager returning a predefined connection."""

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
    """Return a Request-like mock; verify_api_key does not inspect the object."""

    return MagicMock(spec=Request)


@pytest.fixture
def mock_db_with_pool() -> tuple[DatabaseLogger, AsyncMock]:
    """Create a DatabaseLogger mock equipped with an asyncpg-like pool."""

    db_logger = MagicMock(spec=DatabaseLogger)
    connection = AsyncMock()
    connection.fetchrow = AsyncMock()
    connection.execute = AsyncMock()
    pool = MagicMock()
    pool.acquire.side_effect = lambda: _AcquireContext(connection)
    db_logger.pool = pool
    return db_logger, connection


@pytest.mark.asyncio
async def test_auth_disabled_returns_anonymous(monkeypatch, mock_request):
    monkeypatch.setenv("USER_AUTH_ENABLED", "0")
    result = await verify_api_key(request=mock_request)
    assert result == {
        "user_id": "anonymous",
        "tier": "free",
        "authenticated": False,
    }


@pytest.mark.asyncio
async def test_auth_missing_headers_returns_401(monkeypatch, mock_request, mock_db_with_pool):
    monkeypatch.setenv("USER_AUTH_ENABLED", "1")
    monkeypatch.setenv("API_KEY_SECRET", "test-secret")
    db_logger, _ = mock_db_with_pool

    with pytest.raises(HTTPException) as exc:
        await verify_api_key(
            request=mock_request,
            authorization=None,
            x_api_key=None,
            db_logger=db_logger,
        )

    assert exc.value.status_code == 401
    assert "Missing API key" in str(exc.value.detail)


def _setup_fetch_side_effects(
    connection: AsyncMock,
    user_row: dict[str, Any] | None,
    usage_row: dict[str, Any] | None,
) -> None:
    """Configure fetchrow side effects for user lookup and usage query."""

    side_effect: list[Any] = []
    if user_row is not None:
        side_effect.append(user_row)
    else:
        side_effect.append(None)
    if usage_row is not None:
        side_effect.append(usage_row)
    connection.fetchrow.reset_mock()
    connection.fetchrow.side_effect = side_effect


def _hashed_key(monkeypatch, plaintext: str) -> str:
    monkeypatch.setenv("API_KEY_SECRET", "test-secret")
    return hash_api_key(plaintext)


@pytest.mark.asyncio
async def test_auth_authorization_bearer_valid(monkeypatch, mock_request, mock_db_with_pool):
    monkeypatch.setenv("USER_AUTH_ENABLED", "1")
    db_logger, connection = mock_db_with_pool
    plaintext_key = "hyi-valid-key"
    hashed_key = _hashed_key(monkeypatch, plaintext_key)

    _setup_fetch_side_effects(
        connection,
        {
            "id": 1,
            "user_id": "user123",
            "user_name": "Test User",
            "quota_daily_cost_usd": 1000.0,
            "tier": "free",
        },
        {"cost_spent": 10.0},
    )

    result = await verify_api_key(
        request=mock_request,
        authorization=f"Bearer {plaintext_key}",
        db_logger=db_logger,
    )

    assert result["user_id"] == "user123"
    assert result["authenticated"] is True
    assert pytest.approx(result["quota_remaining_cost_usd"], rel=1e-6) == 990.0
    assert connection.fetchrow.await_count == 2
    first_call_hash = connection.fetchrow.await_args_list[0].args[1]
    assert first_call_hash == hashed_key
    assert connection.execute.await_count == 1


@pytest.mark.asyncio
async def test_auth_x_api_key_header_valid(monkeypatch, mock_request, mock_db_with_pool):
    monkeypatch.setenv("USER_AUTH_ENABLED", "1")
    plaintext_key = "hyi-x-header"
    _hashed_key(monkeypatch, plaintext_key)
    db_logger, connection = mock_db_with_pool

    _setup_fetch_side_effects(
        connection,
        {
            "id": 2,
            "user_id": "user456",
            "user_name": "X Header",
            "quota_daily_cost_usd": 500.0,
            "tier": "pro",
        },
        {"cost_spent": 100.0},
    )

    result = await verify_api_key(
        request=mock_request,
        authorization=None,
        x_api_key=plaintext_key,
        db_logger=db_logger,
    )

    assert result["user_id"] == "user456"
    assert result["authenticated"] is True
    assert pytest.approx(result["quota_remaining_cost_usd"], rel=1e-6) == 400.0


@pytest.mark.asyncio
async def test_auth_invalid_key_hash_returns_401(monkeypatch, mock_request, mock_db_with_pool):
    monkeypatch.setenv("USER_AUTH_ENABLED", "1")
    plaintext_key = "hyi-invalid"
    _hashed_key(monkeypatch, plaintext_key)
    db_logger, connection = mock_db_with_pool

    _setup_fetch_side_effects(connection, None, None)

    with pytest.raises(HTTPException) as exc:
        await verify_api_key(
            request=mock_request,
            authorization=f"Bearer {plaintext_key}",
            db_logger=db_logger,
        )

    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_auth_quota_exceeded_returns_429(monkeypatch, mock_request, mock_db_with_pool):
    monkeypatch.setenv("USER_AUTH_ENABLED", "1")
    plaintext_key = "hyi-over-quota"
    _hashed_key(monkeypatch, plaintext_key)
    db_logger, connection = mock_db_with_pool

    _setup_fetch_side_effects(
        connection,
        {
            "id": 3,
            "user_id": "heavy-user",
            "user_name": "Over Quota",
            "quota_daily_cost_usd": 1000.0,
            "tier": "free",
        },
        {"cost_spent": 1000.0},
    )

    with pytest.raises(HTTPException) as exc:
        await verify_api_key(
            request=mock_request,
            authorization=f"Bearer {plaintext_key}",
            db_logger=db_logger,
        )

    assert exc.value.status_code == 429
    assert exc.value.headers["Retry-After"]
    assert exc.value.headers["X-RateLimit-Limit-Cost"] == "1000.0"


@pytest.mark.asyncio
async def test_auth_quota_null_uses_default_1000(monkeypatch, mock_request, mock_db_with_pool):
    monkeypatch.setenv("USER_AUTH_ENABLED", "1")
    plaintext_key = "hyi-null-quota"
    _hashed_key(monkeypatch, plaintext_key)
    db_logger, connection = mock_db_with_pool

    _setup_fetch_side_effects(
        connection,
        {
            "id": 4,
            "user_id": "user-null-quota",
            "user_name": "Null Quota",
            "quota_daily_cost_usd": None,
            "tier": "free",
        },
        {"cost_spent": 0.5},
    )

    result = await verify_api_key(
        request=mock_request,
        authorization=f"Bearer {plaintext_key}",
        db_logger=db_logger,
    )

    assert result["user_id"] == "user-null-quota"
    assert pytest.approx(result["quota_remaining_cost_usd"], rel=1e-6) == 999.5


@pytest.mark.asyncio
async def test_auth_missing_secret_raises_error(monkeypatch, mock_request, mock_db_with_pool):
    monkeypatch.setenv("USER_AUTH_ENABLED", "1")
    monkeypatch.delenv("API_KEY_SECRET", raising=False)
    plaintext_key = "hyi-no-secret"
    db_logger, connection = mock_db_with_pool

    _setup_fetch_side_effects(
        connection,
        {
            "id": 5,
            "user_id": "no-secret",
            "user_name": "No Secret",
            "quota_daily_cost_usd": 1000.0,
            "tier": "free",
        },
        {"cost_spent": 0.0},
    )

    with pytest.raises(ValueError):
        await verify_api_key(
            request=mock_request,
            authorization=f"Bearer {plaintext_key}",
            db_logger=db_logger,
        )


@pytest.mark.asyncio
async def test_auth_updates_last_used_at(monkeypatch, mock_request, mock_db_with_pool):
    monkeypatch.setenv("USER_AUTH_ENABLED", "1")
    plaintext_key = "hyi-update-last-used"
    _hashed_key(monkeypatch, plaintext_key)
    db_logger, connection = mock_db_with_pool

    _setup_fetch_side_effects(
        connection,
        {
            "id": 6,
            "user_id": "user-updated",
            "user_name": "Updated",
            "quota_daily_cost_usd": 200.0,
            "tier": "free",
        },
        {"cost_spent": 50.0},
    )

    await verify_api_key(
        request=mock_request,
        authorization=f"Bearer {plaintext_key}",
        db_logger=db_logger,
    )

    assert connection.execute.await_count == 1
    sql, key_id = connection.execute.await_args_list[0].args
    assert "UPDATE api_keys SET last_used_at = NOW()" in sql
    assert key_id == 6
