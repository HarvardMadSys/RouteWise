from __future__ import annotations

import secrets
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware

from serving.utils import context as req_ctx

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi import Request, Response


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach/propagate an ``X-Request-ID`` header and expose in request.state."""

    header_name = "X-Request-ID"

    async def dispatch(self, request: Request, call_next: Callable):  # type: ignore[override]
        req_id = request.headers.get(self.header_name) or secrets.token_hex(12)
        request.state.request_id = req_id
        # seed request context
        req_ctx.update({"request_id": req_id})
        response: Response = await call_next(request)
        response.headers[self.header_name] = req_id
        return response
