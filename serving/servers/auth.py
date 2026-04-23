"""API key authentication and quota enforcement."""

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Depends, Header, HTTPException, Request

from serving.servers.deps import get_db_logger


def generate_api_key() -> str:
    """Generate a new API key with format: hyi-{32 random bytes}."""
    random_part = secrets.token_urlsafe(32)
    return f"hyi-{random_part}"


def hash_api_key(plaintext_key: str) -> str:
    """Hash API key using HMAC-SHA256 with server secret."""
    secret = os.getenv("API_KEY_SECRET", "").encode()
    if not secret:
        raise ValueError("API_KEY_SECRET must be set in environment")
    return hmac.new(secret, plaintext_key.encode(), hashlib.sha256).hexdigest()


def constant_time_compare(a: str, b: str) -> bool:
    """Constant-time string comparison to prevent timing attacks."""
    return hmac.compare_digest(a, b)


async def verify_api_key(
    request: Request,
    authorization: str | None = Header(None),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    db_logger=Depends(get_db_logger),
) -> dict[str, Any]:
    """Verify API key and enforce quotas.

    Returns user context dict with user_id, tier, etc.
    Raises HTTPException(401/429) on auth/quota failures.
    """
    # Check if auth is enabled
    if os.getenv("USER_AUTH_ENABLED", "0") != "1":
        # Auth disabled - allow all, mark as anonymous
        return {"user_id": "anonymous", "tier": "free", "authenticated": False}

    # Extract API key from headers
    api_key = None
    if authorization and authorization.startswith("Bearer "):
        api_key = authorization[7:]
    elif x_api_key:
        api_key = x_api_key

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Use 'Authorization: Bearer hyi-xxx' or 'X-API-Key: hyi-xxx'",
        )

    # Validate key against database
    if not db_logger or not db_logger.pool:
        # Update metric to reflect database unavailability
        from serving.observability.metrics import DATABASE_CONNECTED

        DATABASE_CONNECTED.set(0)
        raise HTTPException(status_code=500, detail="Database not available for authentication")

    key_hash = hash_api_key(api_key)

    try:
        import asyncpg

        async with db_logger.pool.acquire() as conn:
            user_row = await conn.fetchrow(
                """
                SELECT id, user_id, user_name, quota_daily_cost_usd, tier
                FROM api_keys
                WHERE key_hash = $1
                  AND status = 'active'
                  AND (expires_at IS NULL OR expires_at > NOW())
                """,
                key_hash,
            )

        # Database query succeeded - mark as healthy for faster recovery detection
        from serving.observability.metrics import DATABASE_CONNECTED

        DATABASE_CONNECTED.set(1)

    except asyncpg.PostgresError:
        # Database-specific error (connection failure, timeout, query error, etc.)
        # Update metric and re-raise
        from serving.observability.metrics import DATABASE_CONNECTED

        DATABASE_CONNECTED.set(0)
        raise

    if not user_row:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired API key",
        )

    user = dict(user_row)

    # Pre-check daily cost quota
    # Get cost usage since UTC midnight today
    async with db_logger.pool.acquire() as conn:
        usage_row = await conn.fetchrow(
            """
            SELECT COALESCE(SUM(cost_usd), 0) AS cost_spent
            FROM api_logs
            WHERE user_id = $1
              AND timestamp >= date_trunc('day', NOW() AT TIME ZONE 'UTC')
            """,
            user["user_id"],
        )

    cost_spent = float(usage_row["cost_spent"]) if usage_row else 0.0

    # Estimate cost for this request
    # Strategy: Conservative estimate ~$0.01 per request
    # Actual cost will be calculated precisely during logging
    estimated_cost = 0.01

    # Get quota with fallback for NULL (old rows from migration)
    quota_daily_cost_usd = user.get("quota_daily_cost_usd")
    quota_daily_cost_usd = 1000.0 if quota_daily_cost_usd is None else float(quota_daily_cost_usd)

    # Check cost quota
    if cost_spent + estimated_cost > quota_daily_cost_usd:
        seconds_until_midnight_utc = _seconds_until_utc_midnight()
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Daily cost quota exceeded",
                "quota_usd": quota_daily_cost_usd,
                "spent_usd": cost_spent,
                "retry_after": seconds_until_midnight_utc,
            },
            headers={
                "Retry-After": str(seconds_until_midnight_utc),
                "X-RateLimit-Limit-Cost": str(quota_daily_cost_usd),
                "X-RateLimit-Remaining-Cost": str(max(0, quota_daily_cost_usd - cost_spent)),
                "X-RateLimit-Reset": str(
                    int(
                        (
                            datetime.now(timezone.utc)
                            + timedelta(seconds=seconds_until_midnight_utc)
                        ).timestamp()
                    )
                ),
            },
        )

    # Update last_used timestamp (fire and forget)
    async with db_logger.pool.acquire() as conn:
        await conn.execute(
            "UPDATE api_keys SET last_used_at = NOW() WHERE id = $1",
            user["id"],
        )

    # Return user context
    return {
        "user_id": user["user_id"],
        "user_name": user["user_name"],
        "tier": user["tier"],
        "authenticated": True,
        "quota_remaining_cost_usd": quota_daily_cost_usd - cost_spent,
    }


def _seconds_until_utc_midnight() -> int:
    """Calculate seconds until next UTC midnight."""
    now = datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return int((tomorrow - now).total_seconds())


async def verify_admin_token(
    request: Request,
    authorization: str | None = Header(None),
    db_logger=Depends(get_db_logger),
) -> str:
    """Verify admin token from Authorization header.

    Returns admin IP address for audit logging.
    Raises HTTPException(401) if invalid or missing token.

    All /admin/* routes should use this dependency to protect access.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing admin token. Use 'Authorization: Bearer {ADMIN_TOKEN}' header.",
        )

    token = authorization[7:]  # Strip "Bearer " prefix
    admin_token = os.getenv("ADMIN_TOKEN", "")

    if not admin_token:
        raise HTTPException(
            status_code=500,
            detail="Server misconfiguration: ADMIN_TOKEN environment variable not set",
        )

    # Constant-time comparison to prevent timing attacks
    if not constant_time_compare(token, admin_token):
        raise HTTPException(
            status_code=401,
            detail="Invalid admin token",
        )

    # Extract admin IP for audit logging
    admin_ip = request.client.host if request.client else "unknown"
    return admin_ip


async def log_admin_action(
    db_logger,
    admin_ip: str,
    action: str,
    target_user_id: str | None = None,
    details: dict[str, Any] | None = None,
    success: bool = True,
) -> None:
    """Log admin action to audit trail.

    Args:
        db_logger: DatabaseLogger instance
        admin_ip: IP address of admin performing the action
        action: Action type (e.g., 'create_key', 'revoke_key')
        target_user_id: User ID affected by the action (if applicable)
        details: Additional context (will be stored as JSONB)
        success: Whether the action succeeded
    """
    if not db_logger or not db_logger.pool:
        return  # Silently skip if logging not configured

    import json

    async with db_logger.pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO admin_audit_log (admin_ip, action, target_user_id, details, success)
            VALUES ($1, $2, $3, $4::jsonb, $5)
            """,
            admin_ip,
            action,
            target_user_id,
            json.dumps(details) if details else None,
            success,
        )
