"""Admin API endpoints for user and system management."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from serving.schemas_admin import (
    APIKeyDetailResponse,
    APIKeyDetailUsage,
    APIKeyListItem,
    CreateAPIKeyRequest,
    CreateAPIKeyResponse,
    ListAPIKeysResponse,
    RegenerateAPIKeyResponse,
    RevokeAPIKeyResponse,
    UpdateAPIKeyRequest,
    UpdateAPIKeyResponse,
)
from serving.servers.auth import (
    generate_api_key,
    hash_api_key,
    log_admin_action,
    verify_admin_token,
)
from serving.servers.deps import get_db_logger, get_rate_limiter, get_router, get_services

router = APIRouter()


def _to_json_safe(value: Any) -> Any:
    """Convert values to JSON-serializable primitives."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(item) for item in value]
    return value


def _serialize_for_audit(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize data dict to JSON-safe primitives for audit logging."""
    return {key: _to_json_safe(value) for key, value in data.items()}


@router.get("/stats")
@router.get("/admin/stats")
async def get_stats(
    model: str | None = None,
    provider: str | None = None,
    hours: int = 24,
    db_logger=Depends(get_db_logger),
) -> dict[str, Any]:
    """Return usage statistics from the database logger."""
    if not db_logger:
        return {"error": "Database logging not configured"}

    stats = await db_logger.get_stats(model_id=model, provider=provider, hours=hours)
    return {
        "period_hours": hours,
        "filters": {"model": model, "provider": provider},
        "stats": stats,
    }


@router.get("/rate-limits/{model_id}")
@router.get("/admin/rate-limits/{model_id}")
async def get_rate_limit_status(model_id: str, rate_limiter=Depends(get_rate_limiter)):
    """Get rate limit status and metrics for a specific model."""
    if not rate_limiter:
        return {"error": "Rate limiting not configured"}
    status = rate_limiter.get_status(model_id)
    if not status.get("configured"):
        raise HTTPException(404, f"No rate limit configured for model '{model_id}'")
    return status


@router.get("/rate-limits")
@router.get("/admin/rate-limits")
async def get_all_rate_limits(rate_limiter=Depends(get_rate_limiter)):
    """Get rate limit metrics for all models."""
    if not rate_limiter:
        return {"error": "Rate limiting not configured"}
    return rate_limiter.get_metrics()


@router.post("/rate-limits/{model_id}/reset")
@router.post("/admin/rate-limits/{model_id}/reset")
async def reset_circuit_breaker(model_id: str, rate_limiter=Depends(get_rate_limiter)):
    """Reset circuit breaker for a model (admin endpoint)."""
    if not rate_limiter:
        return {"error": "Rate limiting not configured"}
    rate_limiter.reset_circuit_breaker(model_id)
    return {"message": f"Circuit breaker reset for {model_id}"}


@router.get("/admin/routing")
async def admin_get_routing(
    router_exec=Depends(get_router), services=Depends(get_services)
) -> dict[str, Any]:
    """Admin alias for routing information."""
    routing_info = {}
    for model_id, route in router_exec.routes.items():
        routing_info[model_id] = [
            {
                "provider": adapter.config.provider,
                "base_url": adapter.config.base_url,
                "weight": f"{weight * 100:.0f}%",
            }
            for adapter, weight in route.adapters
        ]

    response: dict[str, Any] = {
        "routes": routing_info,
        "description": "Weight distribution for each model.",
    }
    if services.routing_manager:
        response["manager_status"] = services.routing_manager.get_status()
    return response


# ========================================
# API Key Management Endpoints
# ========================================


@router.post("/admin/api-keys", response_model=CreateAPIKeyResponse, status_code=201)
async def create_api_key(
    request: Request,
    payload: CreateAPIKeyRequest,
    admin_ip: str = Depends(verify_admin_token),
    db_logger=Depends(get_db_logger),
) -> CreateAPIKeyResponse:
    """Create a new API key for a user.

    Returns the plaintext API key ONLY ONCE. Save it immediately.

    Requires: Authorization: Bearer {ADMIN_TOKEN}
    """
    if not db_logger or not db_logger.pool:
        raise HTTPException(500, "Database not configured")

    # Check if user_id already has an active key
    async with db_logger.pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT user_id FROM api_keys WHERE user_id = $1 AND status = 'active'",
            payload.user_id,
        )

    if existing:
        await log_admin_action(
            db_logger,
            admin_ip,
            "create_key",
            payload.user_id,
            {"error": "user_id already exists"},
            success=False,
        )
        raise HTTPException(
            status_code=409,
            detail=f"User '{payload.user_id}' already has an active API key. "
            "Revoke it first or use /regenerate endpoint.",
        )

    # Generate new API key
    plaintext_key = generate_api_key()
    key_hash = hash_api_key(plaintext_key)
    key_prefix = plaintext_key[:12]

    # Insert into database
    async with db_logger.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO api_keys (
                key_hash, key_prefix, user_id, user_name, tier,
                quota_daily_cost_usd, quota_monthly_cost_usd,
                expires_at, notes, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)
            RETURNING id, created_at
            """,
            key_hash,
            key_prefix,
            payload.user_id,
            payload.user_name,
            payload.tier,
            payload.quota_daily_cost_usd,
            payload.quota_monthly_cost_usd,
            payload.expires_at,
            payload.notes,
            payload.metadata,
        )

    # Log admin action
    await log_admin_action(
        db_logger,
        admin_ip,
        "create_key",
        payload.user_id,
        {
            "tier": payload.tier,
            "quota_daily_usd": float(payload.quota_daily_cost_usd),
            "key_prefix": key_prefix,
        },
    )

    return CreateAPIKeyResponse(
        api_key=plaintext_key,
        user_id=payload.user_id,
        key_prefix=key_prefix,
        tier=payload.tier,
        quota_daily_cost_usd=payload.quota_daily_cost_usd,
        quota_monthly_cost_usd=payload.quota_monthly_cost_usd,
        expires_at=payload.expires_at,
        created_at=row["created_at"],
    )


@router.get("/admin/api-keys", response_model=ListAPIKeysResponse)
async def list_api_keys(
    request: Request,
    status: str | None = None,
    tier: str | None = None,
    limit: int = 100,
    offset: int = 0,
    admin_ip: str = Depends(verify_admin_token),
    db_logger=Depends(get_db_logger),
) -> ListAPIKeysResponse:
    """List all API keys with optional filtering.

    Query Parameters:
    - status: Filter by status (active|suspended|revoked)
    - tier: Filter by tier (free|pro|enterprise)
    - limit: Max results (default: 100)
    - offset: Pagination offset

    Requires: Authorization: Bearer {ADMIN_TOKEN}
    """
    if not db_logger or not db_logger.pool:
        raise HTTPException(500, "Database not configured")

    # Build query with filters
    where_clauses = []
    params: list[Any] = []

    if status:
        where_clauses.append(f"status = ${len(params) + 1}")
        params.append(status)

    if tier:
        where_clauses.append(f"tier = ${len(params) + 1}")
        params.append(tier)

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    # Add limit and offset
    params.append(limit)
    params.append(offset)

    async with db_logger.pool.acquire() as conn:
        # Get total count
        count_row = await conn.fetchrow(
            f"SELECT COUNT(*) as total FROM api_keys {where_sql}",
            *params[: len(params) - 2],
        )
        total = count_row["total"] if count_row else 0

        # Get keys
        rows = await conn.fetch(
            f"""
            SELECT
                user_id, user_name, key_prefix, tier, status,
                quota_daily_cost_usd, quota_monthly_cost_usd,
                created_at, last_used_at, expires_at, notes
            FROM api_keys
            {where_sql}
            ORDER BY created_at DESC
            LIMIT ${len(params) - 1} OFFSET ${len(params)}
            """,
            *params,
        )

        if not rows:
            return ListAPIKeysResponse(total=total, keys=[])

        # Batch query: Get usage for all users in one query (avoids N+1 problem)
        user_ids = [row["user_id"] for row in rows]

        # Get today's usage for all users
        usage_today_rows = await conn.fetch(
            """
            SELECT
                user_id,
                COALESCE(SUM(cost_usd), 0) AS cost_spent
            FROM api_logs
            WHERE user_id = ANY($1::text[])
              AND timestamp >= date_trunc('day', NOW() AT TIME ZONE 'UTC')
            GROUP BY user_id
            """,
            user_ids,
        )
        usage_today_map = {row["user_id"]: row["cost_spent"] for row in usage_today_rows}

        # Get this month's usage for all users
        usage_month_rows = await conn.fetch(
            """
            SELECT
                user_id,
                COALESCE(SUM(cost_usd), 0) AS cost_spent
            FROM api_logs
            WHERE user_id = ANY($1::text[])
              AND timestamp >= date_trunc('month', NOW() AT TIME ZONE 'UTC')
            GROUP BY user_id
            """,
            user_ids,
        )
        usage_month_map = {row["user_id"]: row["cost_spent"] for row in usage_month_rows}

        # Build response with O(1) lookup
        keys = []
        for row in rows:
            user_id = row["user_id"]
            keys.append(
                APIKeyListItem(
                    user_id=user_id,
                    user_name=row["user_name"],
                    key_prefix=row["key_prefix"],
                    tier=row["tier"],
                    status=row["status"],
                    quota_daily_cost_usd=row["quota_daily_cost_usd"],
                    quota_monthly_cost_usd=row["quota_monthly_cost_usd"],
                    created_at=row["created_at"],
                    last_used_at=row["last_used_at"],
                    expires_at=row["expires_at"],
                    usage_today_usd=Decimal(str(usage_today_map.get(user_id, 0))),
                    usage_month_usd=Decimal(str(usage_month_map.get(user_id, 0))),
                    notes=row["notes"],
                )
            )

    return ListAPIKeysResponse(total=total, keys=keys)


@router.get("/admin/api-keys/{user_id}", response_model=APIKeyDetailResponse)
async def get_api_key_detail(
    request: Request,
    user_id: str,
    admin_ip: str = Depends(verify_admin_token),
    db_logger=Depends(get_db_logger),
) -> APIKeyDetailResponse:
    """Get detailed information about a specific API key including usage analytics.

    Requires: Authorization: Bearer {ADMIN_TOKEN}
    """
    if not db_logger or not db_logger.pool:
        raise HTTPException(500, "Database not configured")

    async with db_logger.pool.acquire() as conn:
        # Get key info
        row = await conn.fetchrow(
            """
            SELECT
                user_id, user_name, key_prefix, tier, status,
                quota_daily_cost_usd, quota_monthly_cost_usd,
                created_at, last_used_at, expires_at, notes, metadata
            FROM api_keys
            WHERE user_id = $1
            """,
            user_id,
        )

        if not row:
            raise HTTPException(404, f"User '{user_id}' not found")

        # Get today's usage
        usage_today = await conn.fetchrow(
            """
            SELECT
                COALESCE(SUM(cost_usd), 0) AS cost_spent,
                COUNT(*) AS requests
            FROM api_logs
            WHERE user_id = $1
              AND timestamp >= date_trunc('day', NOW() AT TIME ZONE 'UTC')
            """,
            user_id,
        )

        # Get this month's usage
        usage_month = await conn.fetchrow(
            """
            SELECT
                COALESCE(SUM(cost_usd), 0) AS cost_spent,
                COUNT(*) AS requests
            FROM api_logs
            WHERE user_id = $1
              AND timestamp >= date_trunc('month', NOW() AT TIME ZONE 'UTC')
            """,
            user_id,
        )

        # Get models used
        models_rows = await conn.fetch(
            """
            SELECT DISTINCT model_id
            FROM api_logs
            WHERE user_id = $1
              AND timestamp >= NOW() - INTERVAL '30 days'
            ORDER BY model_id
            """,
            user_id,
        )

        # Get last request timestamp
        last_request_row = await conn.fetchrow(
            """
            SELECT MAX(timestamp) AS last_request_at
            FROM api_logs
            WHERE user_id = $1
            """,
            user_id,
        )

    quota_daily = float(row["quota_daily_cost_usd"]) if row["quota_daily_cost_usd"] else 1000.0
    quota_monthly = float(row["quota_monthly_cost_usd"]) if row["quota_monthly_cost_usd"] else None

    cost_today = float(usage_today["cost_spent"]) if usage_today else 0.0
    cost_month = float(usage_month["cost_spent"]) if usage_month else 0.0

    usage = APIKeyDetailUsage(
        today={
            "cost_usd": cost_today,
            "requests": usage_today["requests"] if usage_today else 0,
            "quota_remaining_usd": max(0, quota_daily - cost_today),
        },
        this_month={
            "cost_usd": cost_month,
            "requests": usage_month["requests"] if usage_month else 0,
            "quota_remaining_usd": (max(0, quota_monthly - cost_month) if quota_monthly else None),
        },
        models_used=[r["model_id"] for r in models_rows],
        last_request_at=last_request_row["last_request_at"] if last_request_row else None,
    )

    return APIKeyDetailResponse(
        user_id=row["user_id"],
        user_name=row["user_name"],
        key_prefix=row["key_prefix"],
        tier=row["tier"],
        status=row["status"],
        quota_daily_cost_usd=row["quota_daily_cost_usd"],
        quota_monthly_cost_usd=row["quota_monthly_cost_usd"],
        created_at=row["created_at"],
        last_used_at=row["last_used_at"],
        expires_at=row["expires_at"],
        notes=row["notes"],
        metadata=row["metadata"],
        usage=usage,
    )


@router.patch("/admin/api-keys/{user_id}", response_model=UpdateAPIKeyResponse)
async def update_api_key(
    request: Request,
    user_id: str,
    payload: UpdateAPIKeyRequest,
    admin_ip: str = Depends(verify_admin_token),
    db_logger=Depends(get_db_logger),
) -> UpdateAPIKeyResponse:
    """Update an existing API key's settings.

    All fields are optional - only provided fields will be updated.

    Requires: Authorization: Bearer {ADMIN_TOKEN}
    """
    if not db_logger or not db_logger.pool:
        raise HTTPException(500, "Database not configured")

    # Build dynamic UPDATE query
    updates = []
    params: list[Any] = [user_id]
    param_idx = 2
    updated_fields = []
    new_values = {}

    payload_dict = payload.model_dump(exclude_unset=True)

    for field, value in payload_dict.items():
        updates.append(f"{field} = ${param_idx}")
        params.append(value)
        param_idx += 1
        updated_fields.append(field)
        new_values[field] = value

    if not updates:
        raise HTTPException(422, "No fields to update")

    update_sql = ", ".join(updates)

    async with db_logger.pool.acquire() as conn:
        # Check if user exists
        existing = await conn.fetchrow("SELECT user_id FROM api_keys WHERE user_id = $1", user_id)
        if not existing:
            raise HTTPException(404, f"User '{user_id}' not found")

        # Update
        await conn.execute(
            f"""
            UPDATE api_keys
            SET {update_sql}
            WHERE user_id = $1
            """,
            *params,
        )

    # Log admin action (serialize Decimal/datetime to JSON-safe types)
    await log_admin_action(
        db_logger,
        admin_ip,
        "update_key",
        user_id,
        _serialize_for_audit({"updated_fields": updated_fields, "new_values": new_values}),
    )

    return UpdateAPIKeyResponse(
        user_id=user_id,
        updated_fields=updated_fields,
        new_values=new_values,
    )


@router.delete("/admin/api-keys/{user_id}", response_model=RevokeAPIKeyResponse)
async def revoke_api_key(
    request: Request,
    user_id: str,
    hard_delete: bool = False,
    admin_ip: str = Depends(verify_admin_token),
    db_logger=Depends(get_db_logger),
) -> RevokeAPIKeyResponse:
    """Revoke or delete an API key.

    Query Parameters:
    - hard_delete: If true, permanently delete from database (⚠️ irreversible)
                   If false (default), set status='revoked' (soft delete)

    Requires: Authorization: Bearer {ADMIN_TOKEN}
    """
    if not db_logger or not db_logger.pool:
        raise HTTPException(500, "Database not configured")

    async with db_logger.pool.acquire() as conn:
        # Check if user exists
        existing = await conn.fetchrow("SELECT user_id FROM api_keys WHERE user_id = $1", user_id)
        if not existing:
            raise HTTPException(404, f"User '{user_id}' not found")

        if hard_delete:
            # Permanently delete
            await conn.execute("DELETE FROM api_keys WHERE user_id = $1", user_id)
            action_type = "hard_delete_key"
            response_action = "deleted"
            message = f"API key for user '{user_id}' has been permanently deleted."
        else:
            # Soft delete (set status='revoked')
            await conn.execute(
                "UPDATE api_keys SET status = 'revoked' WHERE user_id = $1",
                user_id,
            )
            action_type = "revoke_key"
            response_action = "revoked"
            message = (
                f"API key for user '{user_id}' has been revoked. User can no longer access the API."
            )

    # Log admin action
    await log_admin_action(
        db_logger,
        admin_ip,
        action_type,
        user_id,
        {"hard_delete": hard_delete},
    )

    return RevokeAPIKeyResponse(
        user_id=user_id,
        action=response_action,
        message=message,
    )


@router.post("/admin/api-keys/{user_id}/regenerate", response_model=RegenerateAPIKeyResponse)
async def regenerate_api_key(
    request: Request,
    user_id: str,
    admin_ip: str = Depends(verify_admin_token),
    db_logger=Depends(get_db_logger),
) -> RegenerateAPIKeyResponse:
    """Regenerate API key for a user (e.g., after suspected compromise).

    This atomically:
    1. Generates a new key
    2. Updates the database
    3. Returns the new plaintext key ONLY ONCE

    The old key is immediately invalidated.

    Requires: Authorization: Bearer {ADMIN_TOKEN}
    """
    if not db_logger or not db_logger.pool:
        raise HTTPException(500, "Database not configured")

    async with db_logger.pool.acquire() as conn:
        # Check if user exists
        old_key_row = await conn.fetchrow(
            "SELECT key_prefix FROM api_keys WHERE user_id = $1",
            user_id,
        )
        if not old_key_row:
            raise HTTPException(404, f"User '{user_id}' not found")

        old_key_prefix = old_key_row["key_prefix"]

        # Generate new key
        new_plaintext_key = generate_api_key()
        new_key_hash = hash_api_key(new_plaintext_key)
        new_key_prefix = new_plaintext_key[:12]

        # Update database atomically
        await conn.execute(
            """
            UPDATE api_keys
            SET key_hash = $1, key_prefix = $2
            WHERE user_id = $3
            """,
            new_key_hash,
            new_key_prefix,
            user_id,
        )

    # Log admin action
    await log_admin_action(
        db_logger,
        admin_ip,
        "regenerate_key",
        user_id,
        {
            "old_key_prefix": old_key_prefix,
            "new_key_prefix": new_key_prefix,
        },
    )

    return RegenerateAPIKeyResponse(
        api_key=new_plaintext_key,
        user_id=user_id,
        key_prefix=new_key_prefix,
        old_key_prefix=old_key_prefix,
    )
