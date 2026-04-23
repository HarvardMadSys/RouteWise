from __future__ import annotations

import json

import aiohttp
import pytest

from serving.http import AsyncHTTPClient


@pytest.mark.unit
@pytest.mark.asyncio
async def test_json_post_with_retry_succeeds_on_third_attempt_after_two_failures(monkeypatch):
    client = AsyncHTTPClient.shared()

    calls = {"n": 0}

    async def fake_json_post(self, url: str, *, json=None, headers=None, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            import aiohttp

            raise aiohttp.ClientError("transient")
        return {"ok": True}

    monkeypatch.setattr(AsyncHTTPClient, "json_post", fake_json_post)
    out = await client.json_post_with_retry("http://example/api", json={})
    assert out == {"ok": True}
    assert calls["n"] == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_json_post_with_retry_exhausts_all_attempts(monkeypatch):
    """Verify retry stops and raises after max attempts are exhausted."""
    client = AsyncHTTPClient.shared()

    async def always_fail(self, url: str, *, json=None, headers=None, timeout=None):
        import aiohttp

        raise aiohttp.ClientError("transient")

    monkeypatch.setattr(AsyncHTTPClient, "json_post", always_fail)

    with pytest.raises(aiohttp.ClientError):
        await client.json_post_with_retry("http://example/api", json={}, retries=3)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stream_post_sse_wraps_lines(monkeypatch):
    client = AsyncHTTPClient.shared()

    class _Resp:
        def __init__(self, chunks: list[bytes]):
            self._chunks = chunks
            self.content = self
            self.headers = {}
            self.status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def iter_chunked(self, n: int):
            for c in self._chunks:
                yield c

        def raise_for_status(self):
            return None

    def fake_post(*args, **kwargs):
        # Two events + DONE
        data = [b"id: 1\n\n", b'data: {"x":1}\n\n', b"data: [DONE]\n\n"]
        return _Resp(data)

    async def fake_ensure(self):  # return object with post()
        class S:
            def post(self, *a, **k):
                return fake_post()

        return S()

    monkeypatch.setattr(AsyncHTTPClient, "_ensure_session", fake_ensure)

    lines: list[str] = []
    async for line in client.stream_post("http://example/sse", json={"stream": True}, mode="sse"):
        lines.append(line)

    assert lines[0].startswith("data: ")
    assert lines[-1] == "data: [DONE]"
    # JSON payload is preserved without trailing newlines
    assert json.loads(lines[0][6:]) == {"x": 1}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stream_post_ndjson(monkeypatch):
    client = AsyncHTTPClient.shared()

    class _Resp:
        def __init__(self, chunks: list[bytes]):
            self._chunks = chunks
            self.content = self
            self.headers = {}
            self.status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def iter_chunked(self, n: int):
            for c in self._chunks:
                yield c

        def raise_for_status(self):
            return None

    def fake_post(*args, **kwargs):
        # Two NDJSON lines split across chunks + trailing partial flush
        data = [b'{"a":1}\n{"b":2}', b"\n  ", b"\n"]
        return _Resp(data)

    async def fake_ensure(self):
        class S:
            def post(self, *a, **k):
                return fake_post()

        return S()

    monkeypatch.setattr(AsyncHTTPClient, "_ensure_session", fake_ensure)

    lines = []
    async for line in client.stream_post("http://example/ndjson", json={}, mode="ndjson"):
        lines.append(line)

    assert [json.loads(line) for line in lines] == [{"a": 1}, {"b": 2}]
