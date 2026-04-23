"""Compatibility wrappers for legacy completion endpoints."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Header, Request

from serving.servers.auth import verify_api_key
from serving.servers.deps import (
    get_db_logger,
    get_rate_limiter,
    get_router,
)

from .completions import chat_completions

router = APIRouter()


@router.post("/completion")
async def single_completion(
    request: Request,
    authorization: str | None = Header(None),
    user_ctx: dict = Depends(verify_api_key),
    router_exec=Depends(get_router),
    rate_limiter=Depends(get_rate_limiter),
    db_logger=Depends(get_db_logger),
):
    """Compatibility alias for single-shot completion requests.

    Forwards to /v1/chat/completions using the provided payload.
    """
    return await chat_completions(
        request,
        authorization=authorization,
        user_ctx=user_ctx,
        router_exec=router_exec,
        rate_limiter=rate_limiter,
        db_logger=db_logger,
    )


@router.post("/v1/completions")
async def legacy_completions(
    request: Request,
    authorization: str | None = Header(None),
    user_ctx: dict = Depends(verify_api_key),
    router_exec=Depends(get_router),
    rate_limiter=Depends(get_rate_limiter),
    db_logger=Depends(get_db_logger),
):
    """OpenAI-style legacy completions endpoint: convert to chat format."""
    body = await request.json()
    prompt = body.get("prompt", "")
    messages = [{"role": "user", "content": prompt}]
    body["messages"] = messages
    body.pop("prompt", None)
    request._body = json.dumps(body).encode()  # type: ignore[attr-defined]
    return await chat_completions(
        request,
        authorization=authorization,
        user_ctx=user_ctx,
        router_exec=router_exec,
        rate_limiter=rate_limiter,
        db_logger=db_logger,
    )
