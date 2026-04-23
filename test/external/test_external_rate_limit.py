#!/usr/bin/env python3
"""External rate-limit tests (optional), require local server on :8080."""

import asyncio
from typing import Any

import aiohttp
import pytest
import requests

pytestmark = pytest.mark.external


async def _send_request(
    session: aiohttp.ClientSession, url: str, model: str, tokens: int
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "a" * (tokens * 4)}],
        "max_tokens": 32,
    }
    try:
        async with session.post(url, json=payload) as resp:
            body = await resp.json()
            return {"status": resp.status, "body": body}
    except Exception as e:
        return {"status": -1, "error": str(e)}


@pytest.mark.asyncio
async def test_rate_limiting_external():
    base_url = "http://localhost:8080"
    try:
        requests.get(f"{base_url}/health", timeout=0.2)
    except Exception:
        pytest.skip("Local server not running on :8080")

    models = requests.get(f"{base_url}/v1/models", timeout=5).json().get("data", [])
    model_id = None
    for m in models:
        if m["id"].startswith("gemini-"):
            model_id = m["id"]
            break
    if not model_id:
        pytest.skip("No rate-limited model configured; skipping")

    async with aiohttp.ClientSession() as session:
        tasks = [
            _send_request(session, f"{base_url}/v1/chat/completions", model_id, 100000)
            for _ in range(5)
        ]
        results = await asyncio.gather(*tasks)
        statuses = [r["status"] for r in results]
        # Expect some 200 and possibly 429 depending on configured limits
        assert all(s in (200, 429) for s in statuses)
