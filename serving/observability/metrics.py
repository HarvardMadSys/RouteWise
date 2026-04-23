"""Server-side Prometheus metrics with graceful no-op fallback.

Migrated from utils/server_metrics.py.
"""

from __future__ import annotations

import os
import re
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    # Optional runtime collectors; not all client versions expose all of these.
    try:
        from prometheus_client import (
            GCCollector,
            PlatformCollector,
            ProcessCollector,
        )
    except Exception:  # pragma: no cover - older versions may not have these
        ProcessCollector = None
        PlatformCollector = None
        GCCollector = None
except Exception:  # pragma: no cover - fallback when lib missing
    CollectorRegistry = None
    Counter = None
    Histogram = None
    Gauge = None
    generate_latest = None


_ENABLED = os.getenv("METRICS_ENABLED", "1") == "1"


def _noop(*_args: Any, **_kwargs: Any) -> Any:  # pragma: no cover
    return None


if _ENABLED and CollectorRegistry and Counter and Histogram:
    REGISTRY = CollectorRegistry()

    API_REQUESTS = Counter(
        "api_requests_total",
        "Total API requests",
        labelnames=("route", "method", "status_class", "stream"),
        registry=REGISTRY,
    )

    API_REQUEST_LATENCY = Histogram(
        "api_request_duration_seconds",
        "Request duration in seconds",
        labelnames=("route", "method", "stream"),
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 30.0),
        registry=REGISTRY,
    )

    API_TTFT = Histogram(
        "api_ttft_seconds",
        "Time to first token in seconds",
        labelnames=("provider", "model"),
        buckets=(0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0),
        registry=REGISTRY,
    )

    PROVIDER_LATENCY = Histogram(
        "provider_request_duration_seconds",
        "Upstream provider call duration in seconds",
        labelnames=("provider", "model", "operation"),
        buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 20.0),
        registry=REGISTRY,
    )

    API_RETRIES = Counter(
        "api_retries_total",
        "Total upstream retries",
        labelnames=("provider", "reason"),
        registry=REGISTRY,
    )

    API_FALLBACKS = Counter(
        "api_fallbacks_total",
        "Total routing fallbacks between providers",
        labelnames=("from_provider", "to_provider", "reason"),
        registry=REGISTRY,
    )

    API_TOKENS = Counter(
        "api_tokens_total",
        "Total tokens by direction",
        labelnames=("model", "provider", "direction"),
        registry=REGISTRY,
    )

    API_TOKEN_ANOMALIES = Counter(
        "api_token_anomalies_total",
        "Token usage anomalies detected (invalid or extreme values)",
        labelnames=("model", "provider", "reason"),
        registry=REGISTRY,
    )

    # Cost tracking metrics - not implemented yet
    # API_COST_CENTS = Counter(
    #     "api_cost_cents_total",
    #     "Accumulated token cost in cents (estimated)",
    #     labelnames=("model", "provider"),
    #     registry=REGISTRY,
    # )
    #
    # API_TOKEN_COST_DOLLARS = Counter(
    #     "api_token_cost_dollars_total",
    #     "Accumulated token cost in USD (estimated) [deprecated]",
    #     labelnames=("model", "provider"),
    #     registry=REGISTRY,
    # )

    # Model behavior metrics - not implemented yet
    # MODEL_REFUSALS = Counter(
    #     "model_refusals_total",
    #     "Model refusal or safety filter events",
    #     labelnames=("model", "provider", "reason"),
    #     registry=REGISTRY,
    # )
    #
    # MODEL_TRUNCATIONS = Counter(
    #     "model_truncations_total",
    #     "Model truncation events",
    #     labelnames=("model", "provider", "direction"),
    #     registry=REGISTRY,
    # )

    STREAMING_INTERRUPTION = Counter(
        "streaming_interruptions_total",
        "Streaming interruptions/errors",
        labelnames=("model", "provider", "stage"),
        registry=REGISTRY,
    )

    RATE_LIMIT_HITS = Counter(
        "rate_limit_hits_total",
        "Rate limiter hits by model and outcome",
        labelnames=("model", "outcome"),  # outcome=accepted|rejected
        registry=REGISTRY,
    )

    API_CONCURRENCY = Gauge(
        "api_concurrent_requests",
        "Number of in-flight API requests",
        registry=REGISTRY,
    )

    # Pipeline health metrics - not implemented yet
    # TRACKING_QUEUE_DEPTH = Gauge(
    #     "tracking_queue_depth",
    #     "Depth of the persistent tracking queue",
    #     registry=REGISTRY,
    # )
    # EXPORT_FAILURES = Counter(
    #     "export_failures_total",
    #     "Total number of export failures",
    #     registry=REGISTRY,
    # )
    # EXPORT_LAG_SECONDS = Gauge(
    #     "export_lag_seconds",
    #     "Time lag between latest exported record and now",
    #     registry=REGISTRY,
    # )
    # WAL_BACKLOG_BYTES = Gauge(
    #     "wal_backlog_bytes",
    #     "Local WAL backlog size in bytes",
    #     registry=REGISTRY,
    # )
    PROVIDER_AVAILABILITY = Gauge(
        "provider_availability",
        "Availability score (0..1) per provider",
        labelnames=("provider",),
        registry=REGISTRY,
    )

    DATABASE_CONNECTED = Gauge(
        "database_connected",
        "Database connection status (1=connected, 0=disconnected)",
        registry=REGISTRY,
    )

    # Rate limiter queueing and wait time
    RATE_LIMIT_QUEUE_SIZE = Gauge(
        "rate_limit_queue_size",
        "Number of requests currently waiting for tokens (per model)",
        labelnames=("model",),
        registry=REGISTRY,
    )
    RATE_LIMIT_QUEUE_WAIT = Histogram(
        "rate_limit_queue_wait_seconds",
        "Time a request waited in the limiter before being accepted",
        labelnames=("model", "outcome"),  # outcome=accepted|rejected
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
        registry=REGISTRY,
    )

    # Circuit breaker metrics (draft for provider-aware routing)
    CIRCUIT_STATE = Gauge(
        "circuit_state",
        "Circuit state per provider (0 closed, 1 open)",
        labelnames=("provider",),
        registry=REGISTRY,
    )
    CIRCUIT_OPEN_TOTAL = Counter(
        "circuit_open_total",
        "Total circuit open events",
        labelnames=("provider", "reason"),
        registry=REGISTRY,
    )

    # Register runtime collectors for process/GC/platform if available
    try:  # pragma: no cover - environment dependent
        if ProcessCollector:
            ProcessCollector(registry=REGISTRY)
        if GCCollector:
            GCCollector(registry=REGISTRY)
        if PlatformCollector:
            PlatformCollector(registry=REGISTRY)
    except Exception:
        pass

    def render_latest() -> bytes:
        """Render current metrics registry in Prometheus text format."""
        from typing import cast

        return cast("bytes", generate_latest(REGISTRY))

    @contextmanager
    def latency_timer(hist: Any, **labels: str) -> Iterator[None]:
        """Context manager to time and observe latency into a histogram."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            hist.labels(**labels).observe(elapsed)

    def status_class_from_code(code: int) -> str:
        """Return coarse HTTP status class label for a status code."""
        if 200 <= code < 300:
            return "2xx"
        if 400 <= code < 500:
            return "4xx"
        if 500 <= code < 600:
            return "5xx"
        return "other"

    # Precompiled patterns for route normalization (performance)
    _UUID_RE = re.compile(r"/[0-9a-fA-F-]{36}")
    _HEX_RE = re.compile(r"/[0-9a-fA-F]{8,}")
    _ID_RE = re.compile(r"/\d+")

    def normalize_route(path: str) -> str:
        """Normalize dynamic path segments to avoid label cardinality explosion."""
        path = _UUID_RE.sub("/:uuid", path)
        path = _HEX_RE.sub("/:hex", path)
        path = _ID_RE.sub("/:id", path)
        return path

    def _sanitize_label_value(value: str, max_len: int = 80) -> str:
        # Replace whitespace, limit charset, and truncate to keep cardinality safer
        import re

        v = value.strip()
        v = re.sub(r"[^a-zA-Z0-9:_\-/\.]+", "_", v)
        if len(v) > max_len:
            return v[: max_len - 3] + "..."
        return v

    def normalize_model_label(model: str) -> str:
        """Normalize model label value for safe Prometheus label usage."""
        import re

        mode = os.getenv("METRICS_MODEL_LABEL", "full")  # full|family
        m = model
        if mode == "family":
            m = re.sub(r"([@:-])(\d{4}-\d{2}-\d{2}|v\d+)$", r"\1", m)
            m = m.rstrip("-:@")
        return _sanitize_label_value(m)

    def normalize_provider_label(provider: str) -> str:
        """Normalize provider label value for safe Prometheus label usage."""
        return _sanitize_label_value(provider)

else:  # No-op fallbacks to avoid hard dependency during tests
    REGISTRY = None
    API_REQUESTS = type("Noop", (), {"labels": lambda *a, **k: type("L", (), {"inc": _noop})()})()
    API_REQUEST_LATENCY = type(
        "NoopH",
        (),
        {
            "labels": lambda *a, **k: type("L", (), {"observe": _noop})(),
        },
    )()
    API_TTFT = API_REQUEST_LATENCY
    PROVIDER_LATENCY = API_REQUEST_LATENCY
    API_RETRIES = type("Noop", (), {"labels": lambda *a, **k: type("L", (), {"inc": _noop})()})()
    API_FALLBACKS = type("Noop", (), {"labels": lambda *a, **k: type("L", (), {"inc": _noop})()})()
    API_TOKENS = type("Noop", (), {"labels": lambda *a, **k: type("L", (), {"inc": _noop})()})()
    API_TOKEN_ANOMALIES = type(
        "Noop", (), {"labels": lambda *a, **k: type("L", (), {"inc": _noop})()}
    )()
    # Commented out unused metrics
    # API_COST_CENTS = type("Noop", (), {"labels": lambda *a, **k: type("L", (), {"inc": _noop})()})()
    # API_TOKEN_COST_DOLLARS = type(
    #     "Noop", (), {"labels": lambda *a, **k: type("L", (), {"inc": _noop})()}
    # )()
    # MODEL_REFUSALS = type("Noop", (), {"labels": lambda *a, **k: type("L", (), {"inc": _noop})()})()
    # MODEL_TRUNCATIONS = type(
    #     "Noop", (), {"labels": lambda *a, **k: type("L", (), {"inc": _noop})()}
    # )()
    STREAMING_INTERRUPTION = type(
        "Noop", (), {"labels": lambda *a, **k: type("L", (), {"inc": _noop})()}
    )()
    RATE_LIMIT_HITS = type(
        "Noop", (), {"labels": lambda *a, **k: type("L", (), {"inc": _noop})()}
    )()
    API_CONCURRENCY = type("NoopGauge", (), {"inc": _noop, "dec": _noop})()
    RATE_LIMIT_QUEUE_SIZE = type(
        "NoopGauge", (), {"labels": lambda *a, **k: type("L", (), {"set": _noop})()}
    )()
    RATE_LIMIT_QUEUE_WAIT = API_REQUEST_LATENCY
    CIRCUIT_STATE = type(
        "NoopGauge", (), {"labels": lambda *a, **k: type("L", (), {"set": _noop})()}
    )()
    CIRCUIT_OPEN_TOTAL = type(
        "Noop", (), {"labels": lambda *a, **k: type("L", (), {"inc": _noop})()}
    )()
    DATABASE_CONNECTED = type("NoopGauge", (), {"set": _noop})()

    def render_latest() -> bytes:  # pragma: no cover
        """Return a minimal body when metrics are disabled."""
        return b"# metrics disabled\n"

    @contextmanager
    def latency_timer(_hist: Any, **_labels: str) -> Iterator[None]:  # pragma: no cover
        """No-op latency timer when metrics are disabled."""
        yield

    def status_class_from_code(code: int) -> str:  # pragma: no cover
        """Return coarse HTTP status class label for a status code."""
        if 200 <= code < 300:
            return "2xx"
        if 400 <= code < 500:
            return "4xx"
        if 500 <= code < 600:
            return "5xx"
        return "other"

    def normalize_route(path: str) -> str:  # pragma: no cover
        """Return path unchanged when metrics are disabled."""
        return path

    def normalize_model_label(model: str) -> str:  # pragma: no cover
        """Return model unchanged when metrics are disabled."""
        return model

    def normalize_provider_label(provider: str) -> str:  # pragma: no cover
        """Return provider unchanged when metrics are disabled."""
        return provider


__all__ = [
    # Core API metrics
    "API_CONCURRENCY",
    "API_FALLBACKS",
    "API_REQUESTS",
    "API_REQUEST_LATENCY",
    "API_RETRIES",
    "API_TOKENS",
    "API_TOKEN_ANOMALIES",
    "API_TTFT",
    # Circuit breaker metrics
    "CIRCUIT_OPEN_TOTAL",
    "CIRCUIT_STATE",
    # Database metrics
    "DATABASE_CONNECTED",
    # Provider metrics
    "PROVIDER_AVAILABILITY",
    "PROVIDER_LATENCY",
    # Rate limiting metrics
    "RATE_LIMIT_HITS",
    "RATE_LIMIT_QUEUE_SIZE",
    "RATE_LIMIT_QUEUE_WAIT",
    # Streaming metrics
    "STREAMING_INTERRUPTION",
    # Helper functions
    "latency_timer",
    "normalize_model_label",
    "normalize_provider_label",
    "normalize_route",
    "render_latest",
    "status_class_from_code",
    # Commented out - not implemented yet:
    # "API_COST_CENTS",
    # "API_TOKEN_COST_DOLLARS",
    # "MODEL_REFUSALS",
    # "MODEL_TRUNCATIONS",
    # "EXPORT_FAILURES",
    # "EXPORT_LAG_SECONDS",
    # "TRACKING_QUEUE_DEPTH",
    # "WAL_BACKLOG_BYTES",
]
