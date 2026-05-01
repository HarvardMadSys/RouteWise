"""Transport abstraction for real online evaluation.

Each transport knows how to send a single streaming chat-completion request
to one provider backend and parse TTFT + token usage.

Supported backends:
    - openrouter       : OpenRouter (optionally with provider hint or sort mode)
    - chutes           : Chutes direct subscription
    - featherless      : Featherless concurrency subscription
    - minimax_native   : MiniMax native token plan endpoint
    - ollama_cloud     : Ollama Cloud Pro subscription

All transports return a common ``SingleRequestResult`` so the runner stays
transport-agnostic.

Migrated from ``NSDI2027_RouteWise/experiment/strategies/joint_online_transport.py``.
The OpenRouter ``sort_mode`` parameter is a first-class field (replaces the
old ``__sort_latency__`` sentinel string the runner used to monkeypatch).
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any

import requests

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MINIMAX_NATIVE_BASE_URL = os.environ.get(
    "MINIMAX_BASE_URL", "https://api.minimax.io/v1"
)
OLLAMA_CLOUD_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL", "https://ollama.com/v1"
)

OPENROUTER_SORT_MODES: frozenset[str] = frozenset({"price", "throughput", "latency"})
RATE_LIMIT_STATUS = 429
DEFAULT_RATE_LIMIT_BACKOFF_SEC = 0.25
MAX_RATE_LIMIT_BACKOFF_SEC = 5.0
logger = logging.getLogger(__name__)


def _ensure_v1_suffix(url: str) -> str:
    """Ensure the base URL ends in ``/v1``; many ``.env`` files omit it."""
    cleaned = url.rstrip("/")
    if cleaned.endswith("/v1"):
        return cleaned
    return cleaned + "/v1"


@dataclass
class SingleRequestResult:
    """Result of one streaming chat-completion request, transport-agnostic."""

    ttft_ms: float
    e2e_ms: float
    status: str
    provider: str
    error_message: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    billed_cost_usd: float = 0.0
    start_ts: float = 0.0
    first_token_ts: float | None = None
    http_status: int | None = None
    retry_count: int = 0
    retry_sleep_ms: float = 0.0
    rate_limited: bool = False


@dataclass
class TransportConfig:
    """Per-provider transport-level configuration.

    ``provider_hint`` and ``sort_mode`` are mutually exclusive — provider_hint
    pins routing to one OpenRouter sub-provider, while sort_mode lets
    OpenRouter pick by ``price`` / ``throughput`` / ``latency``.
    """

    name: str
    transport: str
    model: str | None = None
    provider_hint: str | None = None
    sort_mode: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    input_price_per_m: float = 0.0
    output_price_per_m: float = 0.0
    extra_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sort_mode is not None and self.sort_mode not in OPENROUTER_SORT_MODES:
            raise ValueError(
                f"sort_mode={self.sort_mode!r} not in {sorted(OPENROUTER_SORT_MODES)}"
            )
        if self.sort_mode is not None and self.provider_hint is not None:
            raise ValueError(
                f"{self.name!r}: sort_mode and provider_hint are mutually exclusive"
            )
        if self.sort_mode is not None and self.transport != "openrouter":
            raise ValueError(
                f"{self.name!r}: sort_mode only applies to openrouter transport"
            )


def compute_request_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    input_price_per_m: float,
    output_price_per_m: float,
) -> float:
    """Compute the linear-pricing cost of one request in USD."""
    return (
        input_price_per_m * prompt_tokens + output_price_per_m * completion_tokens
    ) / 1_000_000.0


def _parse_retry_after_sec(value: str | None) -> float | None:
    if not value:
        return None
    stripped = value.strip()
    try:
        return max(0.0, float(stripped))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(stripped)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if retry_at is None:
        return None
    return max(0.0, retry_at.timestamp() - time.time())


def _rate_limit_backoff_sec(response: requests.Response, retry_count: int) -> float:
    retry_after = _parse_retry_after_sec(response.headers.get("Retry-After"))
    if retry_after is not None:
        return min(retry_after, MAX_RATE_LIMIT_BACKOFF_SEC)
    backoff = DEFAULT_RATE_LIMIT_BACKOFF_SEC * (2 ** min(retry_count, 6))
    return min(backoff, MAX_RATE_LIMIT_BACKOFF_SEC)


class BaseTransport:
    """Base class for a streaming chat-completion transport."""

    def __init__(self, cfg: TransportConfig, session: requests.Session):
        self.cfg = cfg
        self.session = session

    def _api_key(self) -> str:
        key = os.environ.get(self.cfg.api_key_env or "", "")
        if not key:
            raise RuntimeError(
                f"Missing API key for {self.cfg.name}: set env {self.cfg.api_key_env}"
            )
        return key

    def _endpoint(self) -> str:
        return (self.cfg.base_url or "").rstrip("/")

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return compute_request_cost_usd(
            prompt_tokens,
            completion_tokens,
            self.cfg.input_price_per_m,
            self.cfg.output_price_per_m,
        )

    def send(
        self,
        prompt: str,
        max_tokens: int,
        timeout: int = 60,
        ttft_event: threading.Event | None = None,
        ttft_info: dict[str, Any] | None = None,
    ) -> SingleRequestResult:
        raise NotImplementedError


class OpenAICompatStreamingTransport(BaseTransport):
    """Streaming chat-completions transport for OpenAI-compatible endpoints.

    Used by chutes, featherless, minimax_native, ollama_cloud, and openrouter.
    Provider-specific differences are handled via ``cfg.model``,
    ``cfg.provider_hint`` (openrouter only), ``cfg.sort_mode`` (openrouter only),
    and ``cfg.extra_headers``.
    """

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key()}",
        }
        headers.update(self.cfg.extra_headers)
        return headers

    def _build_payload(self, prompt: str, max_tokens: int) -> dict[str, Any]:
        # Force greedy decoding so realized output length is dominated by
        # the model + prompt rather than each provider's default sampling
        # temperature (which varies: OpenAI/DeepInfra ≈ 1.0, Together ≈ 0.7,
        # Gemini ≈ 0.4). Without this, cross-provider cost comparisons mix
        # routing differences with sampling-temperature differences.
        # ``seed`` is honored by some OR sub-providers and ignored by
        # others, so it's a best-effort stability hint rather than a
        # guarantee.
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": 42,
            "stream": True,
        }

        if self.cfg.transport == "openrouter":
            payload["stream_options"] = {"include_usage": True}
            if self.cfg.sort_mode is not None:
                payload["provider"] = {"sort": self.cfg.sort_mode}
            elif self.cfg.provider_hint is not None:
                payload["provider"] = {
                    "order": [self.cfg.provider_hint],
                    "allow_fallbacks": False,
                }
        return payload

    def send(
        self,
        prompt: str,
        max_tokens: int,
        timeout: int = 60,
        ttft_event: threading.Event | None = None,
        ttft_info: dict[str, Any] | None = None,
    ) -> SingleRequestResult:
        headers = self._build_headers()
        payload = self._build_payload(prompt, max_tokens)
        url = f"{self._endpoint()}/chat/completions"

        start_perf = time.perf_counter()
        start_ts = time.time()
        deadline_perf = start_perf + max(float(timeout), 1e-6)

        ttft_ms = -1.0
        e2e_ms = -1.0
        first_token_ts: float | None = None
        status = "success"
        error_message: str | None = None
        prompt_tokens = 0
        completion_tokens = 0
        http_status: int | None = None
        reported_cost_usd: float | None = None
        reported_provider: str | None = None
        retry_count = 0
        retry_sleep_sec = 0.0
        saw_rate_limit = False

        if ttft_info is not None:
            ttft_info.setdefault("ttft_ms", -1.0)
            ttft_info.setdefault("first_token_ts", None)
            ttft_info.setdefault("status", None)

        try:
            while True:
                remaining_timeout = max(1e-3, deadline_perf - time.perf_counter())
                response = self.session.post(
                    url=url,
                    headers=headers,
                    json=payload,
                    timeout=remaining_timeout,
                    stream=True,
                )
                http_status = response.status_code

                if response.status_code != 200:
                    error_data: dict[str, Any] = {}
                    try:
                        error_data = response.json() if response.text else {}
                    except Exception:
                        error_data = {}
                    error_message = (
                        error_data.get("error", {}).get("message")
                        if isinstance(error_data.get("error"), dict)
                        else str(error_data)
                    )
                    response.close()
                    if response.status_code == RATE_LIMIT_STATUS:
                        saw_rate_limit = True
                        remaining = deadline_perf - time.perf_counter()
                        if remaining > 0:
                            sleep_sec = min(
                                _rate_limit_backoff_sec(response, retry_count),
                                remaining,
                            )
                            if sleep_sec > 0:
                                retry_count += 1
                                retry_sleep_sec += sleep_sec
                                logger.info(
                                    "%s hit HTTP 429; retrying after %.3fs "
                                    "(attempt %d)",
                                    self.cfg.name,
                                    sleep_sec,
                                    retry_count,
                                )
                                time.sleep(sleep_sec)
                                continue
                        if ttft_event is not None:
                            ttft_event.set()
                    if ttft_info is not None:
                        ttft_info["status"] = f"HTTP {response.status_code}"
                    return SingleRequestResult(
                        ttft_ms=-1.0,
                        e2e_ms=(time.perf_counter() - start_perf) * 1000.0,
                        status=f"HTTP {response.status_code}",
                        provider=self.cfg.name,
                        error_message=error_message or "http_error",
                        prompt_tokens=0,
                        completion_tokens=0,
                        billed_cost_usd=0.0,
                        start_ts=start_ts,
                        first_token_ts=None,
                        http_status=http_status,
                        retry_count=retry_count,
                        retry_sleep_ms=retry_sleep_sec * 1000.0,
                        rate_limited=saw_rate_limit,
                    )

                for raw_line in response.iter_lines(decode_unicode=True):
                    if time.perf_counter() >= deadline_perf:
                        status = "timeout"
                        error_message = "total request timeout"
                        response.close()
                        break
                    if not raw_line:
                        continue
                    if not raw_line.startswith("data:"):
                        continue
                    data_str = raw_line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    usage = chunk.get("usage") or {}
                    if isinstance(usage, dict):
                        prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                        completion_tokens = int(
                            usage.get("completion_tokens") or completion_tokens
                        )
                        if usage.get("cost") is not None:
                            try:
                                reported_cost_usd = float(usage["cost"])
                            except (TypeError, ValueError):
                                pass

                    provider_field = chunk.get("provider")
                    if isinstance(provider_field, str):
                        reported_provider = provider_field

                    choices = chunk.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        # Reasoning models stream visible tokens in
                        # delta.reasoning / reasoning_content / thinking before
                        # delta.content. Treat any non-empty stream field as
                        # the user-visible first token.
                        visible_token = (
                            delta.get("content")
                            or delta.get("reasoning")
                            or delta.get("reasoning_content")
                            or delta.get("thinking")
                        )
                        if visible_token and first_token_ts is None:
                            first_token_ts = time.time()
                            first_token_perf = time.perf_counter()
                            ttft_ms = (first_token_perf - start_perf) * 1000.0
                            if ttft_info is not None:
                                ttft_info["ttft_ms"] = ttft_ms
                                ttft_info["first_token_ts"] = first_token_ts
                                ttft_info["status"] = "success"
                            if ttft_event is not None:
                                ttft_event.set()

                    if time.perf_counter() >= deadline_perf:
                        status = "timeout"
                        error_message = "total request timeout"
                        response.close()
                        break

                end_perf = time.perf_counter()
                e2e_ms = (end_perf - start_perf) * 1000.0
                response.close()
                break

        except requests.exceptions.Timeout:
            status = "timeout"
            error_message = "timeout"
        except requests.exceptions.RequestException as exc:
            status = "error"
            error_message = f"request_exception: {exc}"
        except Exception as exc:
            status = "error"
            error_message = f"unknown_exception: {exc}"
        finally:
            if ttft_event is not None:
                ttft_event.set()
            if ttft_info is not None and ttft_info.get("status") is None:
                ttft_info["status"] = status

        if first_token_ts is None and status == "success":
            status = "no_tokens"
            error_message = "no_tokens_received"

        # Prefer OpenRouter-reported actual cost when present (handles the
        # case where openrouter_auto / sort=latency picks a sub-provider whose
        # price differs from this transport's default pricing).
        if reported_cost_usd is not None and reported_cost_usd > 0:
            billed = reported_cost_usd
        else:
            billed = (
                self._estimate_cost(prompt_tokens, completion_tokens)
                if self.cfg.input_price_per_m > 0 or self.cfg.output_price_per_m > 0
                else 0.0
            )

        logical_provider = self.cfg.name
        if reported_provider:
            logical_provider = f"{self.cfg.name}@{reported_provider}"

        return SingleRequestResult(
            ttft_ms=ttft_ms,
            e2e_ms=e2e_ms,
            status=status,
            provider=logical_provider,
            error_message=error_message,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            billed_cost_usd=billed,
            start_ts=start_ts,
            first_token_ts=first_token_ts,
            http_status=http_status,
            retry_count=retry_count,
            retry_sleep_ms=retry_sleep_sec * 1000.0,
            rate_limited=saw_rate_limit,
        )


def build_transport(cfg: TransportConfig, session: requests.Session) -> BaseTransport:
    """Dispatch to the concrete transport implementation for one provider."""
    if cfg.transport in (
        "openrouter",
        "chutes",
        "featherless",
        "minimax_native",
        "ollama_cloud",
    ):
        return OpenAICompatStreamingTransport(cfg, session)
    raise ValueError(f"Unknown transport: {cfg.transport}")


def resolve_transport_config(provider_entry: dict[str, Any]) -> TransportConfig:
    """Convert one entry from the inventory JSON into a ``TransportConfig``.

    Recognized fields:
        name              : provider display name (required)
        transport         : one of openrouter/chutes/featherless/minimax_native/ollama_cloud
        model             : provider-specific model id (or openrouter_model_id for OR)
        provider_hint     : optional OpenRouter sub-provider pin
        sort_mode         : optional OpenRouter ``sort`` (price/throughput/latency)
        input_price_per_m : optional, USD per 1M input tokens
        output_price_per_m: optional, USD per 1M output tokens
    """
    transport = provider_entry["transport"]
    model = provider_entry.get("model")
    provider_hint = provider_entry.get("provider_hint")
    sort_mode = provider_entry.get("sort_mode")

    if transport == "openrouter":
        base_url = OPENROUTER_BASE_URL
        api_key_env = "OPENROUTER_API_KEY"
        model = (
            model
            or provider_entry.get("openrouter_model_id")
            or "minimax/minimax-m2.5"
        )
        extra_headers = {
            "HTTP-Referer": "https://github.com/HarvardSys/hybridInference",
            "X-Title": "RouteWise real online evaluation",
        }
    elif transport == "chutes":
        base_url = _ensure_v1_suffix(
            os.environ.get("CHUTES_BASE_URL", "https://llm.chutes.ai")
        )
        api_key_env = "CHUTES_API_KEY"
        extra_headers = {}
    elif transport == "featherless":
        base_url = _ensure_v1_suffix(
            os.environ.get("FEATHERLESS_BASE_URL", "https://api.featherless.ai")
        )
        api_key_env = "FEATHERLESS_API_KEY"
        extra_headers = {}
    elif transport == "minimax_native":
        base_url = _ensure_v1_suffix(MINIMAX_NATIVE_BASE_URL)
        api_key_env = "MINIMAX_API_KEY"
        extra_headers = {}
    elif transport == "ollama_cloud":
        base_url = _ensure_v1_suffix(OLLAMA_CLOUD_BASE_URL)
        api_key_env = "OLLAMA_API_KEY"
        extra_headers = {}
    else:
        raise ValueError(f"Unknown transport: {transport}")

    input_price_per_m, output_price_per_m = _resolve_provider_prices(provider_entry)

    return TransportConfig(
        name=provider_entry["name"],
        transport=transport,
        model=model,
        provider_hint=provider_hint,
        sort_mode=sort_mode,
        base_url=base_url,
        api_key_env=api_key_env,
        input_price_per_m=input_price_per_m,
        output_price_per_m=output_price_per_m,
        extra_headers=extra_headers,
    )


def _resolve_provider_prices(provider_entry: dict[str, Any]) -> tuple[float, float]:
    name = str(provider_entry.get("name", "<unnamed>"))
    tier = provider_entry.get("tier")
    transport = provider_entry.get("transport")
    requires_positive_price = tier == "api" or transport == "openrouter"

    input_price = _price_field(
        provider_entry,
        "input_price_per_m",
        required=requires_positive_price,
    )
    output_price = _price_field(
        provider_entry, "output_price_per_m", required=requires_positive_price
    )

    if requires_positive_price and (input_price <= 0.0 or output_price <= 0.0):
        raise ValueError(
            f"{name}: api/openrouter providers require positive input/output "
            f"prices, got input_price_per_m={input_price!r}, "
            f"output_price_per_m={output_price!r}"
        )
    return input_price, output_price


def _price_field(
    provider_entry: dict[str, Any],
    field_name: str,
    *,
    required: bool,
) -> float:
    name = str(provider_entry.get("name", "<unnamed>"))
    value = provider_entry.get(field_name)
    if value is None:
        if required:
            raise ValueError(f"{name}: missing required {field_name}")
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name}: {field_name} must be numeric, got {value!r}"
        ) from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name}: {field_name} must be finite and >= 0")
    return parsed


__all__ = [
    "BaseTransport",
    "OPENROUTER_SORT_MODES",
    "OpenAICompatStreamingTransport",
    "SingleRequestResult",
    "TransportConfig",
    "build_transport",
    "compute_request_cost_usd",
    "resolve_transport_config",
]
