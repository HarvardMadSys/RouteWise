# mypy: disable-error-code=no-any-unimported
"""Lightweight shared async HTTP client for adapters.

Provides a shared aiohttp session with convenience helpers for JSON
requests and streaming responses. Adapters can depend on this instead
of each maintaining their own sessions.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import aiohttp

from serving.observability.metrics import API_RETRIES
from serving.servers.sse import SSEParser
from serving.utils import context as req_ctx

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class AsyncHTTPClient:
    """Shared async HTTP client with a single underlying session."""

    _shared: AsyncHTTPClient | None = None

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None

    @classmethod
    def shared(cls) -> AsyncHTTPClient:
        """Get or create a shared AsyncHTTPClient instance."""
        if cls._shared is None:
            cls._shared = AsyncHTTPClient()
        return cls._shared

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # Set a conservative default timeout; callers can override per request.
            timeout = aiohttp.ClientTimeout(total=60)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def json_post(
        self,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: aiohttp.ClientTimeout | None = None,
    ) -> dict[str, Any]:
        """Send a POST request with JSON payload."""
        session = await self._ensure_session()
        async with session.post(url, json=json, headers=headers, timeout=timeout) as resp:
            resp.raise_for_status()
            from typing import cast

            return cast("dict[str, Any]", await resp.json())

    async def json_post_with_retry(
        self,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: aiohttp.ClientTimeout | None = None,
        retries: int = 3,
        backoff_base: float = 0.5,
        backoff_factor: float = 2.0,
    ) -> dict[str, Any]:
        """POST JSON with simple exponential backoff retries.

        Retries on aiohttp client errors and timeouts. Backoff delays are
        computed as backoff_base * (backoff_factor ** attempt).
        """
        last_err: BaseException | None = None
        for attempt in range(retries):
            try:
                return await self.json_post(url, json=json, headers=headers, timeout=timeout)
            except (aiohttp.ClientError, asyncio.TimeoutError) as err:
                last_err = err
                if attempt == retries - 1:
                    raise
                delay = backoff_base * (backoff_factor**attempt)
                # metrics: retry with context provider label if available
                ctx = req_ctx.get()
                API_RETRIES.labels(
                    provider=str(ctx.get("provider", "unknown")), reason=err.__class__.__name__
                ).inc()
                await asyncio.sleep(delay)
        # Should never reach here, but keep mypy happy.
        assert last_err is not None
        raise last_err

    async def json_get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: aiohttp.ClientTimeout | None = None,
    ) -> dict[str, Any]:
        """Send a GET request and return JSON response."""
        session = await self._ensure_session()
        async with session.get(url, headers=headers, timeout=timeout) as resp:
            resp.raise_for_status()
            from typing import cast

            return cast("dict[str, Any]", await resp.json())

    async def json_get_with_retry(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: aiohttp.ClientTimeout | None = None,
        retries: int = 3,
        backoff_base: float = 0.5,
        backoff_factor: float = 2.0,
    ) -> dict[str, Any]:
        """GET JSON with simple exponential backoff retries."""
        last_err: BaseException | None = None
        for attempt in range(retries):
            try:
                return await self.json_get(url, headers=headers, timeout=timeout)
            except (aiohttp.ClientError, asyncio.TimeoutError) as err:
                last_err = err
                if attempt == retries - 1:
                    raise
                delay = backoff_base * (backoff_factor**attempt)
                # metrics: retry with context provider label if available
                ctx = req_ctx.get()
                API_RETRIES.labels(
                    provider=str(ctx.get("provider", "unknown")), reason=err.__class__.__name__
                ).inc()
                await asyncio.sleep(delay)
        assert last_err is not None
        raise last_err

    async def stream_post(
        self,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: aiohttp.ClientTimeout | None = None,
        mode: str = "sse",
    ) -> AsyncIterator[str]:
        """Yield streaming lines using SSE or NDJSON parsing.

        Args:
            url: Target URL.
            json: Optional JSON payload for the POST request body.
            headers: Optional request headers to include.
            timeout: Optional overall timeout for the request.
            mode: Parsing mode, either "sse" for Server-Sent Events or
                "ndjson" for newline-delimited JSON.

        Yields:
            str: For mode="sse", yields lines prefixed with "data: " and a
            terminal "data: [DONE]". For mode="ndjson", yields complete JSON
            lines without SSE prefix.

        Raises:
            aiohttp.ClientError: If the HTTP request fails.
        """
        session = await self._ensure_session()
        async with session.post(url, json=json, headers=headers, timeout=timeout) as resp:
            resp.raise_for_status()
            # Detect content type for streaming mode if requested
            content_type = str(resp.headers.get("Content-Type", "")).lower()
            detected_mode = mode
            if mode == "auto":
                if "text/event-stream" in content_type:
                    detected_mode = "sse"
                elif (
                    "application/x-ndjson" in content_type
                    or "ndjson" in content_type
                    or "application/json" in content_type
                ):
                    # Many upstreams return a single JSON object for stream endpoints.
                    # Treat it as NDJSON and flush the tail at end.
                    detected_mode = "ndjson"
                else:
                    # Default to SSE when unsure
                    detected_mode = "sse"

            if detected_mode == "sse":
                # Debug logging for SSE streams
                import logging

                logger = logging.getLogger(__name__)
                logger.debug(
                    f"Connected to {url}, status={resp.status}, type={content_type or 'unknown'}"
                )
                logger.debug(f"Response headers: {dict(resp.headers)}")

                parser = SSEParser()
                chunk_count = 0
                message_count = 0
                async for raw in resp.content.iter_chunked(4096):
                    chunk_count += 1
                    if chunk_count <= 5 or chunk_count % 10 == 0:
                        logger.debug(f"Chunk {chunk_count}: received {len(raw)} bytes")
                        # Show first few bytes to debug encoding issues
                        preview = raw[:200].decode("utf-8", errors="replace")
                        logger.debug(f"Chunk {chunk_count} preview: {preview}")

                    messages = list(parser.feed(raw))
                    if messages and chunk_count <= 5:
                        logger.debug(
                            f"Chunk {chunk_count} parser produced {len(messages)} messages"
                        )

                    for msg in messages:
                        if not msg.data:
                            logger.debug("Empty message data, skipping")
                            continue

                        message_count += 1
                        if message_count <= 10 or message_count % 10 == 0:
                            logger.debug(f"Message {message_count} SSE data: {msg.data[:200]}")

                        # Preserve legacy adapter expectations (no trailing newlines)
                        if msg.data.strip() == "[DONE]":
                            logger.debug(
                                f"Received [DONE], total chunks: {chunk_count}, total messages: {message_count}"
                            )
                            yield "data: [DONE]"
                            return

                        output = f"data: {msg.data}"
                        if message_count <= 5:
                            logger.debug(f"Yielding message {message_count}: {output[:200]}")
                        yield output

                logger.info(f"Stream complete: chunks={chunk_count}, messages={message_count}")
            elif detected_mode == "ndjson":
                # Incremental UTF-8 decode + line buffering
                import codecs

                decoder = codecs.getincrementaldecoder("utf-8")()
                buffer = ""
                async for raw in resp.content.iter_chunked(4096):
                    try:
                        text = decoder.decode(raw, final=False)
                    except UnicodeDecodeError:
                        # Wait for next chunk to complete sequence
                        text = ""
                    if text:
                        buffer += text
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if line:
                                yield line
                # Flush tail
                tail = buffer.strip()
                if tail:
                    yield tail
            else:
                # Fallback to raw decoding (legacy behavior)
                async for raw in resp.content:
                    yield raw.decode("utf-8").strip()

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
