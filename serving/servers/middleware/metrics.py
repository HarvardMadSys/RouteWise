# mypy: disable-error-code=no-any-unimported
"""Prometheus metrics middleware for FastAPI.

Records per-request counters and latency histograms with coarse labels.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse

from serving.observability.metrics import (
    API_CONCURRENCY,
    API_REQUEST_LATENCY,
    API_REQUESTS,
    normalize_route,
    status_class_from_code,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class MetricsMiddleware(BaseHTTPMiddleware):
    """Record per-request counters and latency histograms.

    Labels are intentionally coarse to avoid high cardinality: ``route`` uses
    the request URL path, and we distinguish streaming responses by type.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request, record metrics, and return the response."""
        route = normalize_route(request.url.path)
        method = request.method
        started = time.perf_counter()
        API_CONCURRENCY.inc()
        try:
            timeout_s = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
            response: Response = await asyncio.wait_for(call_next(request), timeout=timeout_s)
        except asyncio.TimeoutError:
            # Record timeout as 5xx and raise 504 to be shaped by error handlers.
            elapsed = time.perf_counter() - started
            API_REQUEST_LATENCY.labels(route=route, method=method, stream="unknown").observe(
                elapsed
            )
            API_REQUESTS.labels(
                route=route, method=method, status_class="5xx", stream="unknown"
            ).inc()
            raise HTTPException(status_code=504, detail="Gateway Timeout") from None
        except Exception:
            # Record as 500 and re-raise
            elapsed = time.perf_counter() - started
            API_REQUEST_LATENCY.labels(route=route, method=method, stream="unknown").observe(
                elapsed
            )
            API_REQUESTS.labels(
                route=route, method=method, status_class="5xx", stream="unknown"
            ).inc()
            raise
        else:
            stream_label = "yes" if isinstance(response, StreamingResponse) else "no"
            elapsed = time.perf_counter() - started
            API_REQUEST_LATENCY.labels(route=route, method=method, stream=stream_label).observe(
                elapsed
            )
            status = int(getattr(response, "status_code", 0))
            API_REQUESTS.labels(
                route=route,
                method=method,
                status_class=status_class_from_code(status),
                stream=stream_label,
            ).inc()
            return response
        finally:
            API_CONCURRENCY.dec()
