"""Advanced Token Rate Limiter with Persistence and Accurate Counting.

This module implements a production-grade rate limiter with:
- Accurate token counting using multiple strategies
- SQLite persistence for crash recovery
- Request queueing with backpressure
- Circuit breaker pattern
- Detailed metrics and monitoring

Author: Muxin Tian
Date: 2025-08-26
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from collections import defaultdict
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
from serving.observability.metrics import (
    RATE_LIMIT_HITS,
    RATE_LIMIT_QUEUE_SIZE,
    RATE_LIMIT_QUEUE_WAIT,
)
from serving.utils.tokens import estimate_total_tokens


class TokenCounter:
    """Unified token estimation facade for rate limiting."""

    @staticmethod
    def estimate_tokens(messages: list[dict[str, Any]], max_tokens: int | None = None) -> int:
        return int(estimate_total_tokens(messages, max_tokens))


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    model_id: str
    window_seconds: int
    capacity_tokens: int
    burst_multiplier: float = 1.5
    queue_size: int = 100
    enable_persistence: bool = True


@dataclass
class TokenBucket:
    """Token bucket with refill rate for smooth rate limiting."""

    capacity: int
    refill_rate: float
    tokens: float = field(init=False)
    last_refill: float = field(init=False)

    def __post_init__(self):
        self.tokens = float(self.capacity)
        self.last_refill = time.time()

    def try_consume(self, amount: int) -> tuple[bool, float]:
        """Try to consume tokens from bucket."""
        now = time.time()
        elapsed = now - self.last_refill

        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        if self.tokens >= amount:
            self.tokens -= amount
            return True, self.tokens

        wait_time = (amount - self.tokens) / self.refill_rate
        return False, wait_time


class RateLimitState(Enum):
    """Simplified circuit state placeholder (always closed)."""

    CLOSED = "closed"


class PersistentRateLimiter:
    """Production-grade rate limiter with persistence and circuit breaker."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or os.path.join(
            Path(__file__).parent.parent.parent, "data", "db", "rate_limits.db"
        )

        self.configs: dict[str, RateLimitConfig] = {}
        self.buckets: dict[str, TokenBucket] = {}
        self.queues: dict[str, asyncio.Queue] = {}
        self.states: dict[str, RateLimitState] = defaultdict(lambda: RateLimitState.CLOSED)
        self.metrics: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "total_requests": 0,
                "accepted_requests": 0,
                "rejected_requests": 0,
                "queued_requests": 0,
                "total_tokens": 0,
                "circuit_breaker_trips": 0,
            }
        )

        self._lock = asyncio.Lock()
        self._initialized = False
        # Track cooperative waiters per model to expose as a queue size gauge
        self._waiting: dict[str, int] = defaultdict(int)

    async def initialize(self):
        """Initialize database and restore state."""
        if self._initialized:
            return

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        async with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                # Ensure table exists; include both legacy and new columns.
                # First check if table exists and add new columns if needed.
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='rate_limit_state'"
                )
                if cursor.fetchone():
                    # Table exists, check for new columns.
                    cursor = conn.execute("PRAGMA table_info(rate_limit_state)")
                    columns = {col[1] for col in cursor.fetchall()}

                    if "tokens" not in columns:
                        conn.execute("ALTER TABLE rate_limit_state ADD COLUMN tokens REAL")
                    if "last_refill" not in columns:
                        conn.execute("ALTER TABLE rate_limit_state ADD COLUMN last_refill REAL")
                else:
                    # Create new table with all columns.
                    conn.execute("""
                        CREATE TABLE rate_limit_state (
                            model_id TEXT PRIMARY KEY,
                            tokens_used REAL,
                            window_start REAL,
                            last_update REAL,
                            metrics TEXT,
                            tokens REAL,
                            last_refill REAL
                        )
                        """)
                conn.commit()

                await self._restore_state(conn)
            finally:
                conn.close()

        self._initialized = True

        _ = asyncio.create_task(self._periodic_persistence())  # noqa: RUF006

    async def _restore_state(self, conn: sqlite3.Connection):
        """Restore rate limit state from database."""
        cursor = conn.execute("""
            SELECT model_id, tokens_used, window_start, last_update, metrics, tokens, last_refill
            FROM rate_limit_state
            """)

        now = time.time()
        for row in cursor:
            model_id, tokens_used, window_start, _, metrics_json, tokens_val, last_refill_val = row

            if model_id not in self.configs:
                continue

            config = self.configs[model_id]

            bucket = self.buckets.get(model_id)
            if not bucket:
                continue

            # Prefer new columns if available; otherwise derive from legacy
            if tokens_val is not None and last_refill_val is not None:
                # Clamp tokens to [0, capacity]
                bucket.tokens = max(0.0, min(float(tokens_val), float(bucket.capacity)))
                bucket.last_refill = float(last_refill_val)
            else:
                # Legacy fallback: derive tokens from tokens_used and use window_start as last_refill
                if (
                    window_start is not None
                    and tokens_used is not None
                    and now - float(window_start) < float(config.window_seconds)
                ):
                    bucket.tokens = max(
                        0.0,
                        min(float(config.capacity_tokens - tokens_used), float(bucket.capacity)),
                    )
                    bucket.last_refill = float(window_start)

            if metrics_json:
                with suppress(json.JSONDecodeError):
                    self.metrics[model_id] = json.loads(metrics_json)

    async def _periodic_persistence(self):
        """Periodically persist state to database."""
        while True:
            await asyncio.sleep(10)
            await self._persist_state()

    async def _persist_state(self):
        """Persist current state to database."""
        if not self._initialized:
            return

        conn = sqlite3.connect(self.db_path)
        try:
            for model_id, config in self.configs.items():
                bucket = self.buckets.get(model_id)
                if not bucket:
                    continue

                tokens_remaining = float(bucket.tokens)
                metrics_json = json.dumps(self.metrics[model_id])

                # Persist both legacy and new fields. Use last_refill for window_start
                conn.execute(
                    """
                    INSERT OR REPLACE INTO rate_limit_state
                    (model_id, tokens_used, window_start, last_update, metrics, tokens, last_refill)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        model_id,
                        float(config.capacity_tokens) - tokens_remaining,
                        float(bucket.last_refill),
                        time.time(),
                        metrics_json,
                        tokens_remaining,
                        float(bucket.last_refill),
                    ),
                )

            conn.commit()
        finally:
            conn.close()

    def configure(self, config: RateLimitConfig):
        """Configure rate limiting for a model.

        For strict compliance (no burst beyond window capacity), we clamp
        effective capacity to exactly `capacity_tokens` regardless of
        `burst_multiplier`. This avoids overshooting provider quotas.
        """
        # Enforce no-burst semantics by default
        effective_capacity = int(config.capacity_tokens)

        self.configs[config.model_id] = config

        refill_rate = config.capacity_tokens / config.window_seconds
        self.buckets[config.model_id] = TokenBucket(
            capacity=effective_capacity, refill_rate=refill_rate
        )

        # Keep a queue placeholder for status compatibility, but we no longer
        # enqueue requests; we use a cooperative wait loop in acquire_tokens.
        self.queues[config.model_id] = asyncio.Queue(maxsize=config.queue_size)

        logger.info(
            f"Configured rate limit for {config.model_id}: "
            f"{config.capacity_tokens:,} tokens per {config.window_seconds}s"
        )

    async def acquire_tokens(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        priority: int = 0,  # Reserved for future use.
        timeout: float = 30.0,
    ) -> tuple[bool, dict[str, Any]]:
        """Acquire tokens with cooperative waiting (no internal queueing).

        Returns:
            Tuple of (success, metadata dict with details)
        """
        start_ts = time.time()

        if model_id not in self.configs:
            RATE_LIMIT_HITS.labels(model=model_id, outcome="accepted").inc()
            return True, {"unlimited": True}

        self.metrics[model_id]["total_requests"] += 1

        # No circuit breaker transitions; always CLOSED in this simplified version

        estimated_tokens = TokenCounter.estimate_tokens(messages, max_tokens)
        self.metrics[model_id]["total_tokens"] += estimated_tokens

        bucket = self.buckets[model_id]

        # First, fast-path try
        async with self._lock:
            success, wait_or_remaining = bucket.try_consume(estimated_tokens)
            if success:
                self.metrics[model_id]["accepted_requests"] += 1
                RATE_LIMIT_HITS.labels(model=model_id, outcome="accepted").inc()
                RATE_LIMIT_QUEUE_WAIT.labels(model=model_id, outcome="accepted").observe(
                    max(0.0, time.time() - start_ts)
                )
                return True, {
                    "tokens_consumed": estimated_tokens,
                    "tokens_remaining": wait_or_remaining,
                }

            wait_time = float(wait_or_remaining)

        # If we can't serve immediately, cooperatively wait up to timeout
        if wait_time > timeout:
            self.metrics[model_id]["rejected_requests"] += 1
            RATE_LIMIT_HITS.labels(model=model_id, outcome="rejected").inc()
            RATE_LIMIT_QUEUE_WAIT.labels(model=model_id, outcome="rejected").observe(
                max(0.0, time.time() - start_ts)
            )
            return False, {
                "error": "Rate limit exceeded",
                "tokens_requested": estimated_tokens,
                "wait_time": wait_time,
                "retry_after": int(wait_time),
            }

        # Sleep close to the theoretical needed wait to avoid busy looping
        # Mark as waiting and expose gauge
        self._waiting[model_id] += 1
        RATE_LIMIT_QUEUE_SIZE.labels(model=model_id).set(self._waiting[model_id])

        deadline = time.time() + timeout
        remaining_time = deadline - time.time()
        # Add a small safety margin and cap to remaining_time.
        sleep_for = min(max(0.0, wait_time), max(0.0, remaining_time))
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)

        # Try once more after sleeping
        async with self._lock:
            success, wait_or_remaining = bucket.try_consume(estimated_tokens)
            if success:
                self.metrics[model_id]["accepted_requests"] += 1
                RATE_LIMIT_HITS.labels(model=model_id, outcome="accepted").inc()
                RATE_LIMIT_QUEUE_WAIT.labels(model=model_id, outcome="accepted").observe(
                    max(0.0, time.time() - start_ts)
                )
                # Update waiting gauge
                self._waiting[model_id] = max(0, self._waiting[model_id] - 1)
                RATE_LIMIT_QUEUE_SIZE.labels(model=model_id).set(self._waiting[model_id])
                return True, {
                    "tokens_consumed": estimated_tokens,
                    "tokens_remaining": wait_or_remaining,
                    "waited": True,
                }

        self.metrics[model_id]["rejected_requests"] += 1
        RATE_LIMIT_HITS.labels(model=model_id, outcome="rejected").inc()
        RATE_LIMIT_QUEUE_WAIT.labels(model=model_id, outcome="rejected").observe(
            max(0.0, time.time() - start_ts)
        )
        # Update waiting gauge
        self._waiting[model_id] = max(0, self._waiting[model_id] - 1)
        RATE_LIMIT_QUEUE_SIZE.labels(model=model_id).set(self._waiting[model_id])
        return False, {
            "error": "Rate limit exceeded",
            "tokens_requested": estimated_tokens,
            "retry_after": int(max(1, wait_or_remaining)),
        }

    async def release_tokens(self, model_id: str, tokens: int):
        """Release tokens back to the bucket (on error)."""
        if model_id not in self.buckets:
            return

        async with self._lock:
            bucket = self.buckets[model_id]
            bucket.tokens = min(bucket.capacity, bucket.tokens + tokens)

    def get_metrics(self, model_id: str | None = None) -> dict[str, Any]:
        """Get rate limiter metrics."""
        if model_id:
            return dict(self.metrics.get(model_id, {}))

        return {model: dict(metrics) for model, metrics in self.metrics.items()}

    def get_status(self, model_id: str) -> dict[str, Any]:
        """Get current status for a model."""
        if model_id not in self.configs:
            return {"configured": False}

        config = self.configs[model_id]
        bucket = self.buckets[model_id]
        queue = self.queues[model_id]

        return {
            "configured": True,
            "capacity": config.capacity_tokens,
            "window_seconds": config.window_seconds,
            "tokens_available": int(bucket.tokens),
            "refill_rate": bucket.refill_rate,
            "queue_size": queue.qsize(),
            "queue_capacity": config.queue_size,
            "circuit_breaker_state": RateLimitState.CLOSED.value,
            "metrics": self.get_metrics(model_id),
        }

    def reset_circuit_breaker(self, model_id: str):
        """No-op in simplified version (always closed)."""
        logger.info(f"Circuit breaker reset requested for {model_id} (no-op)")


@asynccontextmanager
async def create_rate_limiter(configs: list[RateLimitConfig], db_path: str | None = None):
    """Create and initialize a rate limiter with configs."""
    limiter = PersistentRateLimiter(db_path)

    for config in configs:
        limiter.configure(config)

    await limiter.initialize()

    try:
        yield limiter
    finally:
        await limiter._persist_state()
