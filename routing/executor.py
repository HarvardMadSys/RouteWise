"""Execution logic for routing requests to AI model adapters."""

from __future__ import annotations

import contextlib
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from serving.adapters.base import BaseAdapter

from serving.observability.metrics import (
    API_FALLBACKS,
    API_TTFT,
    CIRCUIT_OPEN_TOTAL,
    CIRCUIT_STATE,
    PROVIDER_AVAILABILITY,
    PROVIDER_LATENCY,
    STREAMING_INTERRUPTION,
    normalize_model_label,
    normalize_provider_label,
)
from serving.utils import context as req_ctx


@dataclass
class RouteConfig:
    """Weighted adapter list for a model."""

    adapters: list[tuple[BaseAdapter, float]]


class RouteExecutor:
    """Weighted routing executor with automatic fallback.

    Responsibilities:
    - Maintain a mapping from model_id -> weighted adapter list
    - Select an adapter by weight for each request
    - On failure, try remaining adapters in order
    """

    def __init__(self) -> None:
        self.routes: dict[str, RouteConfig] = {}
        # Provider health and circuit breakers shared across models.
        self._health: dict[str, _ProviderHealth] = {}
        self._circuits: dict[str, _CircuitBreaker] = {}
        # Short critical-section lock protecting shared dictionaries/state changes.
        # Use RLock to allow nested locking in helper methods.
        self._lock = threading.RLock()

    def register_route(
        self, model_id: str, adapters_with_weights: list[tuple[BaseAdapter, float]]
    ) -> None:
        """Register a weighted route for a model.

        Args:
            model_id: Model identifier.
            adapters_with_weights: List of (adapter, weight) tuples.
                Weights will be normalized to sum to 1.0.
        """
        total_weight = sum(weight for _, weight in adapters_with_weights)
        if total_weight <= 0:
            return
        normalized = [(adapter, weight / total_weight) for adapter, weight in adapters_with_weights]
        self.routes[model_id] = RouteConfig(adapters=normalized)

    def _select_adapter(self, model_id: str) -> BaseAdapter | None:
        """Select an adapter using weighted random selection.

        Args:
            model_id: Model identifier.

        Returns:
            Selected adapter or None if no route configured.
        """
        route = self.routes.get(model_id)
        if not route or not route.adapters:
            return None
        # Build a snapshot of (adapter, weight, circuit) under a short lock, then
        # decide allow_request() outside the lock to minimize contention.
        with self._lock:
            snapshot: list[tuple[BaseAdapter, float, _CircuitBreaker]] = []
            for adapter, weight in route.adapters:
                provider = adapter.config.provider
                cb = self._circuits.get(provider)
                if not cb:
                    cb = self._circuits[provider] = _CircuitBreaker(provider)
                snapshot.append((adapter, weight, cb))

        allowed: list[tuple[BaseAdapter, float]] = [
            (adapter, weight) for (adapter, weight, cb) in snapshot if cb.allow_request()
        ]

        pool = allowed if allowed else [(a, w) for (a, w, _cb) in snapshot]

        rand = random.random()
        cumulative = 0.0
        for adapter, weight in pool:
            cumulative += weight
            if rand <= cumulative:
                return adapter
        return pool[-1][0]

    async def chat_completion(
        self, model_id: str, messages: list[dict[str, Any]], **params: Any
    ) -> dict[str, Any]:
        """Execute chat completion with automatic fallback.

        Args:
            model_id: Model identifier.
            messages: Chat messages in OpenAI format.
            **params: Additional parameters for the adapter.

        Returns:
            Chat completion response with routing metadata.

        Raises:
            ValueError: If no route configured for model.
        """
        primary = self._select_adapter(model_id)
        if not primary:
            raise ValueError(f"No route configured for model {model_id}")
        try:
            with req_ctx.push(model=model_id, provider=primary.config.provider):
                provider = primary.config.provider
                self._ensure_health(provider)
                started = time.perf_counter()
                resp = await primary.chat_completion(messages, **params)
                PROVIDER_LATENCY.labels(
                    provider=normalize_provider_label(provider),
                    model=normalize_model_label(model_id),
                    operation="chat_completion",
                ).observe(time.perf_counter() - started)
                self._on_success(provider)
            resp["_routing"] = {
                "provider": primary.config.provider,
                "base_url": primary.config.base_url,
            }
            return resp
        except Exception as primary_error:
            route = self.routes[model_id]
            for adapter, _ in route.adapters:
                if adapter == primary:
                    continue
                try:
                    with req_ctx.push(model=model_id, provider=adapter.config.provider):
                        provider = adapter.config.provider
                        self._ensure_health(provider)
                        started = time.perf_counter()
                        resp = await adapter.chat_completion(messages, **params)
                        PROVIDER_LATENCY.labels(
                            provider=normalize_provider_label(provider),
                            model=normalize_model_label(model_id),
                            operation="chat_completion",
                        ).observe(time.perf_counter() - started)
                        self._on_success(provider)
                    resp["_routing"] = {
                        "provider": adapter.config.provider,
                        "base_url": adapter.config.base_url,
                        "fallback": True,
                    }
                    API_FALLBACKS.labels(
                        from_provider=primary.config.provider,
                        to_provider=adapter.config.provider,
                        reason=primary_error.__class__.__name__,
                    ).inc()
                    return resp
                except Exception:
                    self._on_failure(adapter.config.provider, reason="chat_exception")
                    continue
            raise primary_error

    async def stream_chat_completion(
        self, model_id: str, messages: list[dict[str, Any]], **params: Any
    ) -> AsyncIterator[Any]:
        """Stream chat completion with automatic fallback.

        Args:
            model_id: Model identifier.
            messages: Chat messages in OpenAI format.
            **params: Additional parameters for the adapter.

        Yields:
            SSE chunks from the adapter.

        Raises:
            ValueError: If no route configured for model.
        """
        primary = self._select_adapter(model_id)
        if not primary:
            raise ValueError(f"No route configured for model {model_id}")
        try:
            with req_ctx.push(model=model_id, provider=primary.config.provider):
                first = True
                started = time.perf_counter()
                async for chunk in primary.stream_chat_completion(messages, **params):
                    if first and _has_non_empty_content(chunk):
                        # Observe TTFT only when the first non-empty content arrives.
                        # Providers may emit keep-alives or empty terminal chunks.
                        first = False
                        API_TTFT.labels(
                            provider=normalize_provider_label(primary.config.provider),
                            model=normalize_model_label(model_id),
                        ).observe(time.perf_counter() - started)
                        # Consider first non-empty token as a success signal for availability.
                        self._on_success(primary.config.provider)
                    yield chunk
            return
        except Exception as primary_error:
            # record streaming interruption for primary provider
            STREAMING_INTERRUPTION.labels(
                model=model_id,
                provider=primary.config.provider,
                stage="adapter_stream",
            ).inc()
            self._on_failure(primary.config.provider, reason="stream_exception")
            route = self.routes[model_id]
            for adapter, _ in route.adapters:
                if adapter == primary:
                    continue
                try:
                    with req_ctx.push(model=model_id, provider=adapter.config.provider):
                        first = True
                        started = time.perf_counter()
                        async for chunk in adapter.stream_chat_completion(messages, **params):
                            if first and _has_non_empty_content(chunk):
                                first = False
                                API_TTFT.labels(
                                    provider=normalize_provider_label(adapter.config.provider),
                                    model=normalize_model_label(model_id),
                                ).observe(time.perf_counter() - started)
                                self._on_success(adapter.config.provider)
                            yield chunk
                    API_FALLBACKS.labels(
                        from_provider=primary.config.provider,
                        to_provider=adapter.config.provider,
                        reason=primary_error.__class__.__name__,
                    ).inc()
                    return
                except Exception:
                    STREAMING_INTERRUPTION.labels(
                        model=model_id,
                        provider=adapter.config.provider,
                        stage="adapter_stream",
                    ).inc()
                    self._on_failure(adapter.config.provider, reason="stream_exception")
                    continue
            raise primary_error


def _has_non_empty_content(chunk: Any) -> bool:
    r"""Return True if the SSE ``chunk`` carries a non-empty delta content.

    The streaming protocol emits lines like ``"data: {json}\n\n"`` and a
    terminal ``"data: [DONE]\n\n"``. We only consider a chunk as the first
    token when the JSON "choices[0].delta.content" is a non-empty string.
    """
    try:
        if not isinstance(chunk, str | bytes):
            return True  # Unknown type; assume it carries content
        s = chunk.decode() if isinstance(chunk, bytes) else chunk
        if "[DONE]" in s:
            return False
        prefix = "data: "
        if not s.startswith(prefix):
            return True  # Non-standard; assume content
        import json as _json

        payload = s[len(prefix) :].strip()
        obj = _json.loads(payload)
        choices = obj.get("choices") or []
        if not choices:
            return False
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        return isinstance(content, str) and len(content) > 0
    except Exception:
        # Be conservative and treat as content to avoid missing TTFT altogether
        return True


class _ProviderHealth:
    """Track provider availability via exponentially weighted counters."""

    def __init__(self, provider: str, alpha: float | None = None) -> None:
        self.provider = provider
        env_alpha = os.getenv("ROUTER_HEALTH_EWMA_ALPHA")
        self.alpha = (
            float(env_alpha) if env_alpha is not None else (alpha if alpha is not None else 0.2)
        )
        self.ewma_success = 1.0
        self.ewma_total = 1.0
        self._lock = threading.Lock()

    def record(self, success: bool) -> None:
        inc_s = 1.0 if success else 0.0
        with self._lock:
            self.ewma_success = (1 - self.alpha) * self.ewma_success + self.alpha * inc_s
            self.ewma_total = (1 - self.alpha) * self.ewma_total + self.alpha * 1.0

    @property
    def availability(self) -> float:
        if self.ewma_total <= 0:
            return 1.0
        return max(0.0, min(1.0, self.ewma_success / self.ewma_total))


def _providers_status(self: RouteExecutor) -> dict[str, dict[str, Any]]:
    """Return a snapshot of provider availability and circuit state."""
    out: dict[str, dict[str, Any]] = {}
    with self._lock:
        for provider, h in self._health.items():
            state = self._circuits.get(provider).state if provider in self._circuits else "closed"
            out[provider] = {
                "availability": h.availability,
                "circuit_state": state,
            }
    return out


class _CircuitState:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class _CircuitBreaker:
    """Simple circuit breaker per provider.

    - Open when consecutive failures exceed threshold or availability too low.
    - Remain open for a cooldown, then transition to HALF_OPEN to allow a trial.
    - On trial success, close; on failure, reopen and reset cooldown.
    """

    def __init__(
        self,
        provider: str,
        *,
        failure_threshold: int | None = None,
        cooldown_seconds: float | None = None,
        min_availability: float | None = None,
    ) -> None:
        self.provider = provider
        self.state = _CircuitState.CLOSED
        # Read configuration from environment with sensible defaults.
        self.failure_threshold = int(
            os.getenv(
                "CIRCUIT_FAILURE_THRESHOLD",
                str(failure_threshold if failure_threshold is not None else 3),
            )
        )
        self.cooldown_seconds = float(
            os.getenv(
                "CIRCUIT_COOLDOWN_SECONDS",
                str(cooldown_seconds if cooldown_seconds is not None else 30.0),
            )
        )
        self.min_availability = float(
            os.getenv(
                "CIRCUIT_MIN_AVAILABILITY",
                str(min_availability if min_availability is not None else 0.7),
            )
        )
        self.consecutive_failures = 0
        self.last_opened: float | None = None
        self._lock = threading.Lock()
        CIRCUIT_STATE.labels(provider=normalize_provider_label(provider)).set(0)

    def allow_request(self) -> bool:
        with self._lock:
            if self.state == _CircuitState.CLOSED:
                return True
            if self.state == _CircuitState.OPEN:
                if self.last_opened is None:
                    return False
                if (time.perf_counter() - self.last_opened) >= self.cooldown_seconds:
                    # Move to half-open for a trial request.
                    self.state = _CircuitState.HALF_OPEN
                    CIRCUIT_STATE.labels(provider=normalize_provider_label(self.provider)).set(0)
                    return True
                return False
            # HALF_OPEN allows a single trial at a time; conservative approach: allow.
            return True

    def on_success(self) -> None:
        with self._lock:
            self.consecutive_failures = 0
            if self.state in (_CircuitState.OPEN, _CircuitState.HALF_OPEN):
                self.state = _CircuitState.CLOSED
                CIRCUIT_STATE.labels(provider=normalize_provider_label(self.provider)).set(0)

    def on_failure(self, *, availability: float | None = None, reason: str = "error") -> None:
        with self._lock:
            self.consecutive_failures += 1
            trip = False
            if self.consecutive_failures >= self.failure_threshold:
                trip = True
            if availability is not None and availability < self.min_availability:
                trip = True
            if trip:
                self.state = _CircuitState.OPEN
                self.last_opened = time.perf_counter()
                CIRCUIT_STATE.labels(provider=normalize_provider_label(self.provider)).set(1)
                CIRCUIT_OPEN_TOTAL.labels(
                    provider=normalize_provider_label(self.provider), reason=reason
                ).inc()


# Internal helpers on RouteExecutor
def _safe_set_availability(provider: str, value: float) -> None:
    with contextlib.suppress(Exception):
        PROVIDER_AVAILABILITY.labels(provider=normalize_provider_label(provider)).set(value)


def _now() -> float:
    return time.perf_counter()


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _reason_str(s: str) -> str:
    return s if s and len(s) < 64 else "error"


def _log_debug(_: str) -> None:  # reserved for future detailed logs
    return


def _format_provider(p: str) -> str:
    return normalize_provider_label(p)


def _format_model(m: str) -> str:
    return normalize_model_label(m)


# Bind methods to RouteExecutor for clarity
def _ensure_health(self: RouteExecutor, provider: str) -> None:
    with self._lock:
        if provider not in self._health:
            self._health[provider] = _ProviderHealth(provider)
        if provider not in self._circuits:
            self._circuits[provider] = _CircuitBreaker(provider)


def _on_success(self: RouteExecutor, provider: str) -> None:
    with self._lock:
        self._ensure_health(provider)
        self._health[provider].record(True)
        avail = self._health[provider].availability
        self._circuits[provider].on_success()
    # Emit metrics outside the lock to minimize contention.
    _safe_set_availability(provider, avail)


def _on_failure(self: RouteExecutor, provider: str, *, reason: str = "error") -> None:
    with self._lock:
        self._ensure_health(provider)
        self._health[provider].record(False)
        avail = self._health[provider].availability
        self._circuits[provider].on_failure(availability=avail, reason=_reason_str(reason))
    # Emit metrics outside the lock to minimize contention.
    _safe_set_availability(provider, avail)


# Attach helper methods to the class
RouteExecutor._ensure_health = _ensure_health  # type: ignore[attr-defined]
RouteExecutor._on_success = _on_success  # type: ignore[attr-defined]
RouteExecutor._on_failure = _on_failure  # type: ignore[attr-defined]
RouteExecutor.get_provider_status = _providers_status  # type: ignore[attr-defined]
