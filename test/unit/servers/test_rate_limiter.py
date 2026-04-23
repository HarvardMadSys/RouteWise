from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from serving.servers.rate_limiter import PersistentRateLimiter, RateLimitConfig


def _messages(n: int = 0) -> list[dict[str, Any]]:
    if n <= 0:
        return []
    return [{"role": "user", "content": "x" * n}]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rejects_when_wait_exceeds_timeout():
    """Reject when required wait time is larger than provided timeout."""
    limiter = PersistentRateLimiter()
    # Very slow refill: 10 tokens per 1000s => 0.01 t/s
    limiter.configure(RateLimitConfig(model_id="m", window_seconds=1000, capacity_tokens=10))
    await limiter.initialize()

    # Consume all capacity in one go
    ok, _ = await limiter.acquire_tokens("m", messages=[], max_tokens=10)
    assert ok

    # Next request needs 10 tokens but refill is too slow; timeout small
    ok, meta = await limiter.acquire_tokens("m", messages=[], max_tokens=10, timeout=0.05)
    assert ok is False
    assert meta.get("error") == "Rate limit exceeded"
    assert "retry_after" in meta


@pytest.mark.unit
@pytest.mark.asyncio
async def test_wait_then_succeed_with_fast_refill():
    """Fast refill leads to short sleep and eventual success within timeout."""
    limiter = PersistentRateLimiter()
    # Fast refill: 1000 tokens per 1s
    limiter.configure(RateLimitConfig(model_id="m", window_seconds=1, capacity_tokens=1000))
    await limiter.initialize()

    # Deplete the bucket
    ok, _ = await limiter.acquire_tokens("m", messages=[], max_tokens=1000)
    assert ok

    # Request 10 tokens; expected wait ~0.01s; timeout generous
    t0 = time.perf_counter()
    ok, meta = await limiter.acquire_tokens("m", messages=[], max_tokens=10, timeout=0.5)
    t1 = time.perf_counter()
    assert ok is True
    assert (t1 - t0) < 0.2  # Should not wait long on fast refill
    assert "tokens_consumed" in meta


@pytest.mark.unit
@pytest.mark.asyncio
async def test_concurrent_acquire_does_not_oversubscribe():
    limiter = PersistentRateLimiter()
    limiter.configure(RateLimitConfig(model_id="m", window_seconds=60, capacity_tokens=50))
    await limiter.initialize()

    async def acquire():
        return await limiter.acquire_tokens("m", messages=[], max_tokens=10, timeout=0.01)

    results = await asyncio.gather(*(acquire() for _ in range(10)))
    successes = sum(1 for ok, _ in results if ok)
    # With 50 capacity and 10 tokens each, at most 5 succeed
    assert 0 < successes <= 5


@pytest.mark.unit
@pytest.mark.asyncio
async def test_release_tokens_increases_available():
    limiter = PersistentRateLimiter()
    limiter.configure(RateLimitConfig(model_id="m", window_seconds=60, capacity_tokens=100))
    await limiter.initialize()

    ok, _ = await limiter.acquire_tokens("m", messages=[], max_tokens=60)
    assert ok
    before = limiter.get_status("m")["tokens_available"]
    await limiter.release_tokens("m", 20)
    after = limiter.get_status("m")["tokens_available"]
    assert after >= before + 20 - 1  # allow minor float rounding


@pytest.mark.unit
@pytest.mark.asyncio
async def test_persistence_across_restart(tmp_path):
    db_path = tmp_path / "rate_limits.db"

    limiter1 = PersistentRateLimiter(str(db_path))
    limiter1.configure(RateLimitConfig(model_id="m", window_seconds=60, capacity_tokens=100))
    await limiter1.initialize()
    ok, _ = await limiter1.acquire_tokens("m", messages=[], max_tokens=50)
    assert ok
    await limiter1._persist_state()

    limiter2 = PersistentRateLimiter(str(db_path))
    limiter2.configure(RateLimitConfig(model_id="m", window_seconds=60, capacity_tokens=100))
    await limiter2.initialize()
    status = limiter2.get_status("m")
    assert status["configured"] is True
    assert 0 <= status["tokens_available"] <= 100
    assert status["tokens_available"] < 100  # Should reflect prior consumption


@pytest.mark.unit
@pytest.mark.asyncio
async def test_metrics_and_status_shapes():
    limiter = PersistentRateLimiter()
    limiter.configure(RateLimitConfig(model_id="m", window_seconds=60, capacity_tokens=100))
    await limiter.initialize()

    await limiter.acquire_tokens("m", messages=[], max_tokens=10)
    metrics = limiter.get_metrics("m")
    assert "total_requests" in metrics and metrics["total_requests"] >= 1

    status = limiter.get_status("m")
    for key in ["configured", "capacity", "window_seconds", "tokens_available", "metrics"]:
        assert key in status


@pytest.mark.unit
@pytest.mark.asyncio
async def test_token_estimation_influences_consumption():
    """Longer messages should consume more tokens (prompt + completion budget)."""
    limiter = PersistentRateLimiter()
    limiter.configure(RateLimitConfig(model_id="m", window_seconds=60, capacity_tokens=1000))
    await limiter.initialize()

    # Snapshot before
    before = limiter.get_status("m")["tokens_available"]

    short = [{"role": "user", "content": "hi"}]
    long = [{"role": "user", "content": "word " * 100}]

    ok1, _ = await limiter.acquire_tokens("m", messages=short, max_tokens=100)
    assert ok1
    mid = limiter.get_status("m")["tokens_available"]

    ok2, _ = await limiter.acquire_tokens("m", messages=long, max_tokens=100)
    assert ok2
    after = limiter.get_status("m")["tokens_available"]

    # Long message should reduce tokens_available more than short message
    assert (before - mid) < (mid - after)


@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.asyncio
async def test_sliding_window_refill():
    """Tokens should replenish over the configured window."""
    limiter = PersistentRateLimiter()
    limiter.configure(RateLimitConfig(model_id="m", window_seconds=1, capacity_tokens=100))
    await limiter.initialize()

    ok, _ = await limiter.acquire_tokens("m", messages=[], max_tokens=100)
    assert ok
    await asyncio.sleep(1.1)
    # Trigger refill by doing a small acquire
    ok2, meta2 = await limiter.acquire_tokens("m", messages=[], max_tokens=1, timeout=0.1)
    assert ok2 is True
    # After consuming 1 token post-refill, available should be close to capacity - 1
    status = limiter.get_status("m")
    assert status["tokens_available"] >= 98


@pytest.mark.unit
def test_circuit_breaker_state_always_closed():
    """Current implementation keeps circuit breaker closed; reset is a no-op."""
    limiter = PersistentRateLimiter()
    limiter.configure(RateLimitConfig(model_id="m", window_seconds=60, capacity_tokens=10))
    # Not initialized needed for state value; using default
    st = limiter.get_status("m")
    assert st["circuit_breaker_state"] == "closed"
    limiter.reset_circuit_breaker("m")
    st2 = limiter.get_status("m")
    assert st2["circuit_breaker_state"] == "closed"


@pytest.mark.unit
@pytest.mark.perf
@pytest.mark.asyncio
async def test_high_concurrency_performance():
    """Ensure acquire handles many small requests quickly under ample capacity."""
    limiter = PersistentRateLimiter()
    limiter.configure(RateLimitConfig(model_id="m", window_seconds=60, capacity_tokens=10000))
    await limiter.initialize()

    start = time.perf_counter()
    tasks = [
        limiter.acquire_tokens("m", messages=[], max_tokens=1, timeout=0.01) for _ in range(1000)
    ]
    await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0
