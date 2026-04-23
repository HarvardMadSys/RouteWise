"""Pydantic schemas for admin API endpoints."""

from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field, field_validator


class CreateAPIKeyRequest(BaseModel):  # type: ignore[no-any-unimported]
    """Request payload for creating a new API key."""

    user_id: str = Field(..., min_length=1, max_length=255, description="Unique user identifier")
    user_name: str | None = Field(None, max_length=255, description="Display name for the user")
    tier: str = Field("free", pattern="^(free|pro|enterprise)$", description="User tier")
    quota_daily_cost_usd: Decimal = Field(
        Decimal("1000.00"),
        ge=0,
        description="Daily cost quota in USD",
    )
    quota_monthly_cost_usd: Decimal | None = Field(
        None,
        ge=0,
        description="Monthly cost quota in USD (optional)",
    )
    expires_at: datetime | None = Field(
        None,
        description="Expiration timestamp (optional)",
    )
    notes: str | None = Field(None, description="Admin notes")
    metadata: dict[str, Any] | None = Field(None, description="Custom metadata")

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: str) -> str:
        """Ensure user_id contains only safe characters."""
        if not v.replace("_", "").replace("-", "").replace(".", "").isalnum():
            raise ValueError("user_id must contain only alphanumeric, dash, underscore, or dot")
        return v


class CreateAPIKeyResponse(BaseModel):  # type: ignore[no-any-unimported]
    """Response payload for successful API key creation."""

    api_key: str = Field(..., description="Plaintext API key (shown only once)")
    user_id: str
    key_prefix: str = Field(..., description="First 12 characters for identification")
    tier: str
    quota_daily_cost_usd: Decimal
    quota_monthly_cost_usd: Decimal | None
    expires_at: datetime | None
    created_at: datetime
    warning: str = Field(
        default="⚠️ Save this API key now. It cannot be retrieved later.",
        description="Security warning",
    )


class APIKeyListItem(BaseModel):  # type: ignore[no-any-unimported]
    """Single API key item in list view."""

    user_id: str
    user_name: str | None
    key_prefix: str
    tier: str
    status: str
    quota_daily_cost_usd: Decimal
    quota_monthly_cost_usd: Decimal | None
    created_at: datetime
    last_used_at: datetime | None
    expires_at: datetime | None
    usage_today_usd: Decimal = Field(default=Decimal("0"), description="Cost spent today")
    usage_month_usd: Decimal = Field(default=Decimal("0"), description="Cost spent this month")
    notes: str | None


class ListAPIKeysResponse(BaseModel):  # type: ignore[no-any-unimported]
    """Response payload for listing API keys."""

    total: int
    keys: list[APIKeyListItem]


class APIKeyDetailUsage(BaseModel):  # type: ignore[no-any-unimported]
    """Detailed usage statistics for a user."""

    today: dict[str, Any] = Field(
        default_factory=dict,
        description="Today's usage: cost_usd, requests, quota_remaining_usd",
    )
    this_month: dict[str, Any] = Field(
        default_factory=dict,
        description="This month's usage: cost_usd, requests, quota_remaining_usd",
    )
    models_used: list[str] = Field(default_factory=list, description="List of models accessed")
    last_request_at: datetime | None = Field(None, description="Timestamp of last request")


class APIKeyDetailResponse(BaseModel):  # type: ignore[no-any-unimported]
    """Response payload for detailed API key view."""

    user_id: str
    user_name: str | None
    key_prefix: str
    tier: str
    status: str
    quota_daily_cost_usd: Decimal
    quota_monthly_cost_usd: Decimal | None
    created_at: datetime
    last_used_at: datetime | None
    expires_at: datetime | None
    notes: str | None
    metadata: dict[str, Any] | None
    usage: APIKeyDetailUsage


class UpdateAPIKeyRequest(BaseModel):  # type: ignore[no-any-unimported]
    """Request payload for updating an API key."""

    user_name: str | None = Field(None, max_length=255)
    tier: str | None = Field(None, pattern="^(free|pro|enterprise)$")
    status: str | None = Field(None, pattern="^(active|suspended|revoked)$")
    quota_daily_cost_usd: Decimal | None = Field(None, ge=0)
    quota_monthly_cost_usd: Decimal | None = Field(None, ge=0)
    expires_at: datetime | None = None
    notes: str | None = None
    metadata: dict[str, Any] | None = None


class UpdateAPIKeyResponse(BaseModel):  # type: ignore[no-any-unimported]
    """Response payload for successful update."""

    user_id: str
    updated_fields: list[str]
    new_values: dict[str, Any]


class RevokeAPIKeyResponse(BaseModel):  # type: ignore[no-any-unimported]
    """Response payload for successful revocation."""

    user_id: str
    action: str  # "revoked" or "deleted"
    message: str


class RegenerateAPIKeyResponse(BaseModel):  # type: ignore[no-any-unimported]
    """Response payload for successful key regeneration."""

    api_key: str = Field(..., description="New plaintext API key (shown only once)")
    user_id: str
    key_prefix: str
    old_key_prefix: str
    warning: str = Field(
        default="⚠️ Old key is now revoked. Save this new key immediately.",
        description="Security warning",
    )


# Rebuild models to ensure forward references are resolved when imported via FastAPI
__all__ = [
    "APIKeyDetailResponse",
    "APIKeyDetailUsage",
    "APIKeyListItem",
    "CreateAPIKeyRequest",
    "CreateAPIKeyResponse",
    "ListAPIKeysResponse",
    "RegenerateAPIKeyResponse",
    "RevokeAPIKeyResponse",
    "UpdateAPIKeyRequest",
    "UpdateAPIKeyResponse",
]
