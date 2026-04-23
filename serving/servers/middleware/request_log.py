from __future__ import annotations

import time
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware

from serving.utils import context as req_ctx
from serving.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi import Request, Response

logger = get_logger(__name__)


class RequestLogMiddleware(BaseHTTPMiddleware):
    """Emit a concise structured log per HTTP request."""

    async def dispatch(self, request: Request, call_next: Callable):  # type: ignore[override]
        start = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = int((time.perf_counter() - start) * 1000)
        ctx = req_ctx.get()
        # Enrich logs to help identify misrouted or unexpected callers.
        # Note: ``request.client.host`` will be the proxy's IP (e.g., NGINX). The
        # original client should be available via ``X-Forwarded-For`` when the
        # proxy sets it.
        remote_ip = getattr(getattr(request, "client", None), "host", None)
        xff = request.headers.get("x-forwarded-for")
        user_agent = request.headers.get("user-agent")
        host = request.headers.get("host")
        request_id = ctx.get("request_id")
        # Extract canonical session_id for logs (same as DB metadata).
        canonical_session_id = request.headers.get("X-Session-ID")

        logger.info(
            "http_request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": getattr(response, "status_code", 0),
                "duration_ms": duration_ms,
                "model": ctx.get("model"),
                "provider": ctx.get("provider"),
                "remote_ip": remote_ip,
                "x_forwarded_for": xff,
                "user_agent": user_agent,
                "host": host,
                "request_id": request_id,
                "session_id": canonical_session_id,
            },
        )

        # Debug-only: emit a compact headers snapshot with sensitive fields masked.
        # This helps diagnose whether upstream clients (e.g., Cursor) include
        # conversation/session identifiers without flooding logs.
        if logger.isEnabledFor(10):  # logging.DEBUG
            masked_headers: dict[str, str] = {}
            for k, v in request.headers.items():
                key_lower = k.lower()
                if key_lower in {"authorization", "x-api-key"}:
                    masked_headers[k] = "***"
                else:
                    # Truncate very long header values to keep logs readable.
                    val = v if v is not None else ""
                    masked_headers[k] = (val[:256] + "…") if len(val) > 256 else val
            logger.debug("http_request_headers", extra={"headers": masked_headers})
        return response
