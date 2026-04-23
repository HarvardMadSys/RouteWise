from __future__ import annotations

"""Expose Prometheus metrics at /metrics.

If prometheus_client is unavailable or METRICS_ENABLED=0, returns a minimal
payload so the endpoint still exists without crashing.

Note: This endpoint does NOT perform active database queries to avoid blocking
Prometheus scrapes. Database connection status is updated by:
- /health endpoint (active SELECT 1 queries)
- Bootstrap initialization (on startup)
- Auth middleware (on request authentication failures)
"""

from fastapi import APIRouter, Response

from serving.observability.metrics import render_latest

router = APIRouter()


@router.get("/metrics")
async def metrics() -> Response:
    """Expose Prometheus metrics.

    Database connection metric is updated elsewhere to avoid blocking scrapes.
    """
    payload = render_latest()
    return Response(content=payload, media_type="text/plain; version=0.0.4; charset=utf-8")
