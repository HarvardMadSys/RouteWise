"""Integration tests against a real PostgreSQL instance."""

from __future__ import annotations

import os
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from fastapi import Request

from serving.servers.auth import hash_api_key, verify_api_key
from serving.storage.database import DatabaseLogger, calculate_cost

if TYPE_CHECKING:
    import asyncpg

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def pg_dsn() -> str:
    dsn = os.getenv("TEST_PG_DSN")
    if not dsn:
        pytest.skip("TEST_PG_DSN is not set; skipping database integration tests")
    return dsn


async def _truncate_tables(pool: asyncpg.pool.Pool) -> None:
    async with pool.acquire() as conn:
        for table in ("api_logs", "api_keys"):
            exists = await conn.fetchval("SELECT to_regclass($1)", f"public.{table}")
            if exists:
                await conn.execute(f"TRUNCATE TABLE {table}")


@pytest_asyncio.fixture
async def db_logger(pg_dsn: str):
    logger = DatabaseLogger({"dsn": pg_dsn}, store_full_prompts=False)
    await logger.initialize()
    assert logger.pool is not None
    await _truncate_tables(logger.pool)
    try:
        yield logger
    finally:
        assert logger.pool is not None
        await _truncate_tables(logger.pool)
        await logger.cleanup()


@pytest.mark.asyncio
async def test_api_logs_schema_contains_cost_columns(db_logger: DatabaseLogger):
    assert db_logger.pool is not None
    async with db_logger.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'api_logs'
            """
        )
        column_names = {row["column_name"] for row in rows}

    assert {"cache_read_tokens", "cache_write_tokens", "cost_usd"}.issubset(column_names)


@pytest.mark.asyncio
async def test_log_request_persists_cost_and_usage(db_logger: DatabaseLogger):
    usage: dict[str, Any] = {
        "prompt_tokens": 1000,
        "completion_tokens": 400,
        "reasoning_tokens": 200,
        "cache_read_tokens": 500,
        "cache_write_tokens": 250,
    }
    pricing = {
        "prompt": "0.15",
        "completion": "1.25",
        "input_cache_reads": "0.05",
        "input_cache_writes": "0.10",
    }
    expected_cost = calculate_cost(usage, pricing)

    await db_logger.log_request(
        request_id="req-integration-1",
        model_id="integration-model",
        provider="gemini",
        prompt=[{"role": "user", "content": "hi"}],
        response={"message": "ok"},
        usage=usage,
        latency_ms=123,
        status_code=200,
        params={},
        metadata={"user_id": "integration-user"},
        pricing=pricing,
    )

    assert db_logger.pool is not None
    async with db_logger.pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT prompt_tokens, cache_read_tokens, cost_usd FROM api_logs WHERE request_id=$1",
            "req-integration-1",
        )

    assert row is not None
    assert row["prompt_tokens"] == 1000
    assert row["cache_read_tokens"] == 500
    assert float(row["cost_usd"]) == pytest.approx(expected_cost or 0.0)


@pytest.mark.asyncio
async def test_verify_api_key_against_real_database(db_logger: DatabaseLogger, monkeypatch):
    assert db_logger.pool is not None
    monkeypatch.setenv("USER_AUTH_ENABLED", "1")
    monkeypatch.setenv("API_KEY_SECRET", "integration-secret")

    plaintext_key = "hyi-integration-key"
    key_hash = hash_api_key(plaintext_key)
    key_prefix = plaintext_key[:12]
    user_id = "integration-user"

    async with db_logger.pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO api_keys (
                key_hash, key_prefix, user_id, user_name, quota_daily_cost_usd, tier
            ) VALUES ($1, $2, $3, $4, $5, $6)
            """,
            key_hash,
            key_prefix,
            user_id,
            "Integration User",
            Decimal("1000.00"),
            "enterprise",
        )

    pricing = {"prompt": "0.15", "completion": "1.25"}
    usage = {"prompt_tokens": 400, "completion_tokens": 200}
    await db_logger.log_request(
        request_id="req-integration-2",
        model_id="integration-model",
        provider="gemini",
        prompt=[{"role": "user", "content": "ping"}],
        response={"message": "pong"},
        usage=usage,
        latency_ms=45,
        status_code=200,
        params={},
        metadata={"user_id": user_id},
        pricing=pricing,
    )

    mock_request = MagicMock(spec=Request)
    result = await verify_api_key(
        request=mock_request,
        authorization=f"Bearer {plaintext_key}",
        db_logger=db_logger,
    )

    assert result["user_id"] == user_id
    assert result["authenticated"] is True
    assert result["quota_remaining_cost_usd"] < 1000
