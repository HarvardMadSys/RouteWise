"""Simple health-check loop for deployment endpoints."""

# mypy: disable-error-code=no-any-unimported
from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import aiohttp


class HealthMonitor:
    """Health monitoring for deployment endpoints.

    Periodically checks endpoint health via GET /health requests.
    """

    def __init__(self, timeout_s: int, interval_s: int) -> None:
        """Initialize health monitor.

        Args:
            timeout_s: Request timeout in seconds.
            interval_s: Check interval in seconds (0 to disable).
        """
        self.timeout_s = timeout_s
        self.interval_s = interval_s
        self._status: dict[str, bool] = {}
        self._task: asyncio.Task | None = None

    def is_healthy(self, endpoint: str) -> bool:
        """Check if an endpoint is healthy.

        Args:
            endpoint: Endpoint URL to check.

        Returns:
            True if healthy or unknown, False if known unhealthy.
        """
        return self._status.get(endpoint, True)

    async def _check_once(self, session: Any, endpoint: str) -> bool:
        """Perform a single health check against the origin's /health path.

        This ignores any API prefix (e.g., "/v1") present in the endpoint URL
        to avoid 404s such as "/v1/health".
        """
        try:
            from urllib.parse import urlparse, urlunparse

            parsed = urlparse(endpoint)
            # Always probe the origin root at /health.
            health_url = urlunparse((parsed.scheme, parsed.netloc, "/health", "", "", ""))
            async with session.get(
                health_url, timeout=aiohttp.ClientTimeout(total=self.timeout_s)
            ) as resp:
                return bool(resp.status == 200)
        except Exception:
            return False

    async def _run(self, endpoints: list[str]) -> None:
        if self.interval_s <= 0:
            return
        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                for ep in endpoints:
                    ok = await self._check_once(session, ep)
                    self._status[ep] = ok
                await asyncio.sleep(self.interval_s)

    def start(self, endpoints: list[str]) -> None:
        """Start health monitoring for given endpoints.

        Args:
            endpoints: List of endpoint URLs to monitor.
        """
        if self.interval_s <= 0 or self._task is not None:
            return
        self._task = asyncio.create_task(self._run(endpoints))

    async def shutdown(self) -> None:
        """Stop health monitoring and cleanup resources."""
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
