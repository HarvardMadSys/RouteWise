"""PostgreSQL-backed request/metrics logger using asyncpg.

This module provides a simple database logger that writes API requests,
responses, and usage metrics into PostgreSQL tables. It is intended for
production or staging environments where PostgreSQL is available.
"""

import hashlib
import json
from typing import Any

import asyncpg


def compute_prompt_hash(prompt: list[dict[str, Any]] | str) -> str:
    """Compute SHA256 hash of prompt for deduplication and caching.

    Args:
        prompt: Prompt messages (list of dicts) or string.

    Returns:
        Hex-encoded SHA256 hash (64 characters).
    """
    # Normalize to JSON string for consistent hashing
    if isinstance(prompt, list):
        # Sort keys for deterministic JSON output
        prompt_str = json.dumps(prompt, sort_keys=True, ensure_ascii=False)
    else:
        prompt_str = str(prompt)

    return hashlib.sha256(prompt_str.encode("utf-8")).hexdigest()


def calculate_cost(
    usage: dict[str, Any] | None,
    pricing: dict[str, str] | None,
) -> float | None:
    """Compute request cost in USD based on usage and pricing tables."""
    if not usage or not pricing:
        return None

    try:
        prompt_tokens = float(usage.get("prompt_tokens", 0))
        completion_tokens = float(usage.get("completion_tokens", 0))
        reasoning_tokens = float(usage.get("reasoning_tokens", 0))
        cache_read_tokens = float(usage.get("cache_read_tokens", 0))
        cache_write_tokens = float(usage.get("cache_write_tokens", 0))

        prompt_price = float(pricing.get("prompt", "0"))
        completion_price = float(pricing.get("completion", "0"))
        cache_read_price = float(pricing.get("input_cache_reads", "0"))
        cache_write_price = float(pricing.get("input_cache_writes", "0"))

        return (
            (prompt_tokens * prompt_price / 1_000_000)
            + (completion_tokens * completion_price / 1_000_000)
            + (reasoning_tokens * completion_price / 1_000_000)
            + (cache_read_tokens * cache_read_price / 1_000_000)
            + (cache_write_tokens * cache_write_price / 1_000_000)
        )
    except (ValueError, TypeError):
        return None


class DatabaseLogger:
    """Asynchronous PostgreSQL logger using a pooled connection."""

    def __init__(self, db_config: dict[str, str], store_full_prompts: bool = True):
        """Initialize the logger with a DSN/config mapping.

        Args:
            db_config: Mapping with asyncpg pool connection arguments.
            store_full_prompts: If False, only store prompt_hash for privacy.
                When disabled, prompt and response fields will be NULL and only
                hashes are stored for analytics.
        """
        self.db_config = db_config
        self.store_full_prompts = store_full_prompts
        # Use Any to avoid mypy issues when asyncpg types are unavailable.
        self.pool: Any | None = None

    async def initialize(self) -> None:
        """Create the connection pool and ensure tables exist."""
        self.pool = await asyncpg.create_pool(
            **self.db_config, min_size=2, max_size=10, command_timeout=60
        )
        await self._create_tables()

    async def _create_tables(self) -> None:
        """Create tables and indexes if they do not exist."""
        if self.pool is None:
            raise RuntimeError("DatabaseLogger not initialized")
        async with self.pool.acquire() as conn:
            # Main logs table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS api_logs (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    request_id TEXT NOT NULL UNIQUE,
                    model_id TEXT NOT NULL,
                    provider TEXT NOT NULL,

                    -- Request parameters
                    temperature FLOAT,
                    top_p FLOAT,
                    max_tokens INTEGER,
                    seed INTEGER,
                    stream BOOLEAN,

                    -- Performance metrics
                    ttft_ms INTEGER,
                    latency_ms INTEGER,

                    -- Token usage (optional)
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    reasoning_tokens INTEGER,
                    total_tokens INTEGER,

                    -- Request/response content
                    prompt TEXT,
                    response TEXT,
                    prompt_hash TEXT,

                    -- Additional metadata (kept for compatibility)
                    status_code INTEGER,
                    error TEXT,
                    user_id TEXT,
                    session_id TEXT,
                    metadata JSONB,
                    tools JSONB
                )
            """)

            # Indexes for performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_logs_timestamp
                ON api_logs(timestamp DESC)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_logs_model
                ON api_logs(model_id, timestamp DESC)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_logs_provider
                ON api_logs(provider, timestamp DESC)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_logs_request_id
                ON api_logs(request_id)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_logs_user
                ON api_logs(user_id, timestamp DESC)
                WHERE user_id IS NOT NULL
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_logs_session
                ON api_logs(session_id, timestamp DESC)
                WHERE session_id IS NOT NULL
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_logs_error
                ON api_logs(timestamp DESC)
                WHERE error IS NOT NULL
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_logs_prompt_hash
                ON api_logs(prompt_hash)
                WHERE prompt_hash IS NOT NULL
            """)

            # Migrations for existing databases
            await conn.execute("""
                ALTER TABLE api_logs
                ADD COLUMN IF NOT EXISTS reasoning_tokens INTEGER
            """)

            await conn.execute("""
                ALTER TABLE api_logs
                ADD COLUMN IF NOT EXISTS stream BOOLEAN
            """)

            await conn.execute("""
                ALTER TABLE api_logs
                ADD COLUMN IF NOT EXISTS ttft_ms INTEGER
            """)

            await conn.execute("""
                ALTER TABLE api_logs
                ADD COLUMN IF NOT EXISTS prompt_hash TEXT
            """)

            await conn.execute("""
                ALTER TABLE api_logs
                ADD COLUMN IF NOT EXISTS cache_read_tokens INTEGER
            """)

            await conn.execute("""
                ALTER TABLE api_logs
                ADD COLUMN IF NOT EXISTS cache_write_tokens INTEGER
            """)

            await conn.execute("""
                ALTER TABLE api_logs
                ADD COLUMN IF NOT EXISTS cost_usd DECIMAL(12, 8)
            """)

            # Aggregated stats table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS api_stats_hourly (
                    hour TIMESTAMPTZ NOT NULL,
                    model_id TEXT NOT NULL,
                    provider TEXT NOT NULL,

                    request_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    error_count INTEGER DEFAULT 0,

                    total_prompt_tokens BIGINT DEFAULT 0,
                    total_completion_tokens BIGINT DEFAULT 0,
                    total_tokens BIGINT DEFAULT 0,

                    avg_latency_ms FLOAT,
                    p50_latency_ms INTEGER,
                    p95_latency_ms INTEGER,
                    p99_latency_ms INTEGER,
                    PRIMARY KEY (hour, model_id, provider)
                )
            """)

            # API Keys table for user authentication
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id BIGSERIAL PRIMARY KEY,
                    key_hash TEXT NOT NULL UNIQUE,
                    key_prefix TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    user_name TEXT,
                    status TEXT NOT NULL DEFAULT 'active',

                    quota_daily_cost_usd DECIMAL(10, 4) DEFAULT 1000.00,
                    quota_monthly_cost_usd DECIMAL(10, 4),

                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    expires_at TIMESTAMPTZ,
                    last_used_at TIMESTAMPTZ,

                    tier TEXT DEFAULT 'free',
                    notes TEXT,
                    metadata JSONB
                )
            """)

            # Indexes for api_keys
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_keys_user
                ON api_keys(user_id)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_keys_status
                ON api_keys(status, expires_at)
            """)

            await conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_api_keys_prefix_unique
                ON api_keys(key_prefix)
            """)

            # Enforce 1:1 user-to-key relationship (one active key per user)
            await conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_api_keys_user_unique
                ON api_keys(user_id) WHERE status = 'active'
            """)

            # Migrations for api_keys table (from token-based to cost-based quotas)
            await conn.execute("""
                ALTER TABLE api_keys
                ADD COLUMN IF NOT EXISTS quota_daily_cost_usd DECIMAL(10, 4) DEFAULT 1000.00
            """)

            await conn.execute("""
                ALTER TABLE api_keys
                ADD COLUMN IF NOT EXISTS quota_monthly_cost_usd DECIMAL(10, 4)
            """)

            # Admin audit log table for tracking all admin operations
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS admin_audit_log (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    admin_ip TEXT NOT NULL,
                    action TEXT NOT NULL,
                    target_user_id TEXT,
                    details JSONB,
                    success BOOLEAN DEFAULT TRUE
                )
            """)

            # Indexes for admin audit log
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_admin_audit_timestamp
                ON admin_audit_log(timestamp DESC)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_admin_audit_user
                ON admin_audit_log(target_user_id, timestamp DESC)
            """)

            # Critical index for usage analytics (prevents full table scan on cost queries)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_logs_user_cost
                ON api_logs(user_id, timestamp, cost_usd)
            """)

    async def log_request(
        self,
        request_id: str,
        model_id: str,
        provider: str,
        prompt: list[dict[str, Any]] | str,
        response: dict[str, Any] | str | None,
        usage: dict[str, int] | None,
        latency_ms: int,
        status_code: int,
        error: str | None = None,
        params: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        ttft_ms: int | None = None,
        prompt_hash: str | None = None,
        store_full_content: bool | None = None,
        pricing: dict[str, str] | None = None,
    ) -> None:
        """Insert a single request log row.

        Args:
            request_id: Unique request identifier.
            model_id: Logical model identifier.
            provider: Provider name for the request.
            prompt: Request messages payload (list of dicts or string).
            response: Provider response payload (dict, string, or None).
            usage: Token usage breakdown.
            latency_ms: End-to-end latency in milliseconds.
            status_code: HTTP status code returned to client.
            error: Optional error message.
            params: Request parameters (temperature, max_tokens, stream, etc.).
            metadata: Additional metadata (user_id, session_id, etc.).
            ttft_ms: Time to first token in milliseconds.
            prompt_hash: Hash of prompt for deduplication/caching (auto-computed if None).
            store_full_content: Override instance default for storing full prompt/response.
                If None, uses self.store_full_prompts. Set False for privacy mode (hash only).
            pricing: Model pricing config for cost calculation (per 1M tokens).
        """
        if not self.pool:
            raise RuntimeError("DatabaseLogger not initialized")

        # Auto-compute prompt_hash if not provided
        if prompt_hash is None:
            prompt_hash = compute_prompt_hash(prompt)

        # Privacy control: use per-request override or instance default
        should_store_full = (
            store_full_content if store_full_content is not None else self.store_full_prompts
        )

        if should_store_full:
            # Store full prompt and response text
            prompt_str = json.dumps(prompt) if isinstance(prompt, list) else str(prompt)
            response_str = (
                json.dumps(response)
                if isinstance(response, dict)
                else str(response)
                if response is not None
                else None
            )
        else:
            # Privacy mode: only store hash, not content
            prompt_str = None
            response_str = None

        # Calculate cost based on usage and pricing
        cost_usd = calculate_cost(usage, pricing)

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO api_logs (
                    request_id, model_id, provider,
                    temperature, top_p, max_tokens, seed, stream,
                    ttft_ms, latency_ms,
                    prompt_tokens, completion_tokens, reasoning_tokens, total_tokens,
                    cache_read_tokens, cache_write_tokens, cost_usd,
                    prompt, response, prompt_hash,
                    status_code, error, user_id, session_id, metadata,
                    tools
                )
                VALUES (
                    $1, $2, $3,
                    $4, $5, $6, $7, $8,
                    $9, $10,
                    $11, $12, $13, $14,
                    $15, $16, $17,
                    $18, $19, $20,
                    $21, $22, $23, $24, $25::jsonb,
                    $26::jsonb
                )
                ON CONFLICT (request_id) DO NOTHING
                """,
                request_id,
                model_id,
                provider,
                # Request parameters
                (params or {}).get("temperature"),
                (params or {}).get("top_p"),
                (params or {}).get("max_tokens"),
                (params or {}).get("seed"),
                (params or {}).get("stream"),
                # Performance metrics
                ttft_ms,
                latency_ms,
                # Token usage
                (usage or {}).get("prompt_tokens"),
                (usage or {}).get("completion_tokens"),
                (usage or {}).get("reasoning_tokens"),
                (usage or {}).get("total_tokens"),
                # Cache and cost
                (usage or {}).get("cache_read_tokens"),
                (usage or {}).get("cache_write_tokens"),
                cost_usd,
                # Content
                prompt_str,
                response_str,
                prompt_hash,
                # Metadata
                status_code,
                error,
                (metadata or {}).get("user_id"),
                (metadata or {}).get("session_id"),
                json.dumps(metadata) if metadata else None,
                json.dumps((params or {}).get("tools")) if (params or {}).get("tools") else None,
            )

    async def get_stats(
        self, model_id: str | None = None, provider: str | None = None, hours: int = 24
    ) -> list[dict[str, Any]]:
        """Fetch aggregated hourly stats for the given time window."""
        if not self.pool:
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM api_stats_hourly
                WHERE hour >= NOW() - ($1 || ' hours')::interval
                AND ($2::text IS NULL OR model_id = $2)
                AND ($3::text IS NULL OR provider = $3)
                ORDER BY hour DESC
                LIMIT 1000
                """,
                hours,
                model_id,
                provider,
            )
        return [dict(r) for r in rows]

    async def cleanup(self) -> None:
        """Close the connection pool if initialized."""
        if self.pool:
            try:
                await self.pool.close()  # type: ignore[attr-defined]
            finally:
                self.pool = None
