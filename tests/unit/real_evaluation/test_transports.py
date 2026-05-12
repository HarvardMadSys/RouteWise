"""Transport-level streaming semantics."""

from __future__ import annotations

import json
import threading
import time

import pytest

from experiments.real_evaluation.transports import (
    OpenAICompatStreamingTransport,
    TransportConfig,
    observed_cache_read_tokens,
    resolve_transport_config,
)


class _FakeResponse:
    def __init__(
        self,
        status_code: int,
        *,
        headers: dict[str, str] | None = None,
        payload: dict | None = None,
        stream_chunks: list[dict] | None = None,
    ):
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = payload or {}
        self._stream_chunks = stream_chunks or []
        self.closed = False
        self.text = json.dumps(self._payload) if self._payload else ""

    def json(self):
        return self._payload

    def iter_lines(self, decode_unicode: bool = True):
        for chunk in self._stream_chunks:
            yield "data: " + json.dumps(chunk)
        yield "data: [DONE]"

    def close(self) -> None:
        self.closed = True


class _SlowStreamResponse(_FakeResponse):
    def iter_lines(self, decode_unicode: bool = True):
        yield "data: " + json.dumps({"choices": [{"delta": {"content": "hello"}}]})
        time.sleep(0.02)
        yield "data: " + json.dumps({"choices": [{"delta": {"content": "world"}}]})
        yield "data: [DONE]"


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def post(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("no fake response left")
        return self.responses.pop(0)


def _transport(session: _FakeSession) -> OpenAICompatStreamingTransport:
    cfg = TransportConfig(
        name="OR_Test",
        transport="openrouter",
        model="test/model",
        base_url="https://example.test/v1",
        api_key_env="OPENROUTER_API_KEY",
        stream_cancel_billing="stops",
        stream_cancel_billing_by_provider={"Friendli": "stops"},
    )
    return OpenAICompatStreamingTransport(cfg, session)  # type: ignore[arg-type]


def test_http_429_returns_immediately_without_same_provider_retry(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    response = _FakeResponse(
        429,
        headers={"Retry-After": "0.01"},
        payload={"error": {"message": "rate limited"}},
    )

    def fail_sleep(seconds: float) -> None:
        raise AssertionError(f"429 should not sleep before retrying, got {seconds}")

    monkeypatch.setattr("experiments.real_evaluation.transports.time.sleep", fail_sleep)
    session = _FakeSession([response])
    event = threading.Event()
    ttft_info: dict = {}

    result = _transport(session).send(
        prompt="x",
        max_tokens=8,
        timeout=5,
        ttft_event=event,
        ttft_info=ttft_info,
    )

    assert len(session.calls) == 1
    assert response.closed is True
    assert result.status == "HTTP 429"
    assert result.rate_limited is True
    assert result.retry_count == 0
    assert result.retry_sleep_ms == 0.0
    assert result.ttft_ms == -1.0
    assert result.e2e_ms >= 0.0
    assert result.billed_cost_usd == 0.0
    assert result.physical_cost_usd == 0.0
    assert result.provider == "OR_Test"
    assert event.is_set()
    assert ttft_info["status"] == "HTTP 429"


def test_transport_parses_observed_cached_input_tokens(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    session = _FakeSession(
        [
            _FakeResponse(
                200,
                stream_chunks=[
                    {
                        "choices": [{"delta": {"content": "hello"}}],
                        "usage": {
                            "prompt_tokens": 20,
                            "completion_tokens": 5,
                            "cost": 0.000123,
                            "prompt_tokens_details": {"cached_tokens": 11},
                        },
                    }
                ],
            )
        ]
    )

    result = _transport(session).send(prompt="x", max_tokens=8, timeout=5)

    assert result.cache_read_tokens_observed == 11


def test_observed_cache_read_tokens_accepts_anthropic_usage_shape() -> None:
    assert observed_cache_read_tokens({"cache_read_input_tokens": 13}) == 13


def test_http_429_retry_after_header_does_not_delay_provider_failure(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    response = _FakeResponse(
        429,
        headers={"Retry-After": "10"},
        payload={"error": {"message": "rate limited"}},
    )
    session = _FakeSession(
        [
            response,
        ]
    )

    result = _transport(session).send(prompt="x", max_tokens=8, timeout=60)

    assert len(session.calls) == 1
    assert response.closed is True
    assert result.status == "HTTP 429"
    assert result.rate_limited is True
    assert result.retry_count == 0
    assert result.retry_sleep_ms == 0.0
    assert result.ttft_ms == -1.0


def test_streaming_request_honors_total_wall_clock_timeout(monkeypatch) -> None:
    """Streaming providers may keep the socket active with periodic chunks.

    The real-eval timeout is a total request deadline, not just a socket idle
    timeout; otherwise reasoning-heavy responses can outlive the runner's join
    window and disappear from the CSV.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    session = _FakeSession([_SlowStreamResponse(200)])

    result = _transport(session).send(prompt="x", max_tokens=8, timeout=0.005)

    assert result.status == "timeout"
    assert result.error_message == "total request timeout"
    assert result.ttft_ms > 0
    assert result.e2e_ms >= result.ttft_ms


def test_openrouter_provider_filter_payload_preserves_auto_routing() -> None:
    """``provider.only`` filters OpenRouter auto without pinning an order."""
    cfg = TransportConfig(
        name="OR_auto_filtered",
        transport="openrouter",
        model="test/model",
        provider_only=("Chutes", "DeepInfra"),
        base_url="https://example.test/v1",
        api_key_env="OPENROUTER_API_KEY",
        input_price_per_m=0.1,
        output_price_per_m=1.0,
        stream_cancel_billing="stops",
    )
    transport = OpenAICompatStreamingTransport(cfg, _FakeSession([]))  # type: ignore[arg-type]

    payload = transport._build_payload("x", 8)

    assert payload["provider"] == {"only": ["Chutes", "DeepInfra"]}


def test_openrouter_provider_filter_payload_combines_with_sort_mode() -> None:
    """Filtered sort baselines should sort inside the filtered provider set."""
    cfg = TransportConfig(
        name="OR_latency_filtered",
        transport="openrouter",
        model="test/model",
        sort_mode="latency",
        provider_only=("Chutes", "DeepInfra"),
        provider_ignore=("BadProvider",),
        base_url="https://example.test/v1",
        api_key_env="OPENROUTER_API_KEY",
        input_price_per_m=0.1,
        output_price_per_m=1.0,
        stream_cancel_billing="stops",
    )
    transport = OpenAICompatStreamingTransport(cfg, _FakeSession([]))  # type: ignore[arg-type]

    payload = transport._build_payload("x", 8)

    assert payload["provider"] == {
        "only": ["Chutes", "DeepInfra"],
        "ignore": ["BadProvider"],
        "sort": "latency",
    }


def test_resolve_transport_config_parses_openrouter_provider_filters() -> None:
    cfg = resolve_transport_config(
        {
            "name": "OR_filtered",
            "tier": "api",
            "transport": "openrouter",
            "model": "test/model",
            "provider_only": ["Chutes", "Chutes", "DeepInfra"],
            "provider_ignore": "BadProvider",
            "input_price_per_m": 0.1,
            "output_price_per_m": 1.0,
            "stream_cancel_billing": "stops",
        }
    )

    assert cfg.provider_only == ("Chutes", "DeepInfra")
    assert cfg.provider_ignore == ("BadProvider",)


def test_resolve_transport_config_parses_cached_input_price() -> None:
    cfg = resolve_transport_config(
        {
            "name": "OR_cached",
            "tier": "api",
            "transport": "openrouter",
            "model": "test/model",
            "input_price_per_m": 0.1,
            "cached_input_price_per_m": 0.02,
            "output_price_per_m": 1.0,
            "stream_cancel_billing": "stops",
        }
    )

    assert cfg.cached_input_price_per_m == 0.02


def test_openrouter_requires_explicit_stream_cancel_billing() -> None:
    with pytest.raises(ValueError, match="missing required stream_cancel_billing"):
        resolve_transport_config(
            {
                "name": "OR_missing_cancel_mode",
                "tier": "api",
                "transport": "openrouter",
                "model": "test/model",
                "input_price_per_m": 0.1,
                "output_price_per_m": 1.0,
            }
        )


def test_openrouter_rejects_invalid_stream_cancel_billing() -> None:
    with pytest.raises(ValueError, match="stream_cancel_billing must be"):
        resolve_transport_config(
            {
                "name": "OR_bad_cancel_mode",
                "tier": "api",
                "transport": "openrouter",
                "model": "test/model",
                "input_price_per_m": 0.1,
                "output_price_per_m": 1.0,
                "stream_cancel_billing": "unknown",
            }
        )


def test_non_openrouter_rejects_stream_cancel_billing_map() -> None:
    with pytest.raises(ValueError, match="stream_cancel_billing_by_provider only applies"):
        TransportConfig(
            name="Chutes",
            transport="chutes",
            model="test/model",
            stream_cancel_billing_by_provider={"Chutes": "stops"},
        )


def test_resolve_transport_config_honors_api_key_env_override() -> None:
    cfg = resolve_transport_config(
        {
            "name": "Featherless_SC",
            "tier": "concurrency",
            "transport": "featherless",
            "model": "test/model",
            "api_key_env": "FEATHERLESS_API_KEY_POLICY_0",
            "input_price_per_m": 0.0,
            "output_price_per_m": 0.0,
            "billing_mode": "subscription",
        }
    )

    assert cfg.api_key_env == "FEATHERLESS_API_KEY_POLICY_0"


def test_openrouter_api_provider_requires_positive_prices() -> None:
    """OpenRouter/API providers must not silently fall back to zero cost."""
    with pytest.raises(ValueError, match="require positive input/output prices"):
        resolve_transport_config(
            {
                "name": "OR_bad_price",
                "tier": "api",
                "transport": "openrouter",
                "model": "test/model",
                "input_price_per_m": 0.0,
                "output_price_per_m": 1.0,
                "stream_cancel_billing": "stops",
            }
        )


def test_subscription_provider_allows_zero_marginal_price() -> None:
    """Subscription providers can have zero marginal price; their cost is
    represented by quota/concurrency shadow prices instead."""
    cfg = resolve_transport_config(
        {
            "name": "Chutes_SQ",
            "tier": "quota",
            "transport": "chutes",
            "model": "provider/model",
            "input_price_per_m": 0.0,
            "output_price_per_m": 0.0,
        }
    )

    assert cfg.input_price_per_m == 0.0
    assert cfg.output_price_per_m == 0.0


def test_openrouter_subscription_provider_allows_zero_paper_price() -> None:
    cfg = resolve_transport_config(
        {
            "name": "Chutes_SQ",
            "tier": "quota",
            "transport": "openrouter",
            "model": "test/model",
            "provider_hint": "Chutes",
            "input_price_per_m": 0.0,
            "output_price_per_m": 0.0,
            "billing_mode": "subscription",
            "stream_cancel_billing": "stops",
        }
    )

    assert cfg.transport == "openrouter"
    assert cfg.billing_mode == "subscription"
    assert cfg.input_price_per_m == 0.0
    assert cfg.output_price_per_m == 0.0


def test_openrouter_subscription_reports_physical_but_zero_paper_cost(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    session = _FakeSession(
        [
            _FakeResponse(
                200,
                stream_chunks=[
                    {
                        "choices": [{"delta": {"content": "hello"}}],
                        "usage": {
                            "prompt_tokens": 3,
                            "completion_tokens": 5,
                            "cost": 0.000123,
                        },
                        "provider": "Chutes",
                    }
                ],
            )
        ]
    )
    cfg = TransportConfig(
        name="Chutes_SQ",
        transport="openrouter",
        model="test/model",
        provider_hint="Chutes",
        base_url="https://example.test/v1",
        api_key_env="OPENROUTER_API_KEY",
        billing_mode="subscription",
        stream_cancel_billing="stops",
    )

    result = OpenAICompatStreamingTransport(cfg, session).send(
        prompt="x",
        max_tokens=8,
        timeout=5,
    )

    assert result.billed_cost_usd == 0.0
    assert result.physical_cost_usd == 0.000123
    assert result.cost_source == "subscription_zero_marginal+reported_physical"


class _CancellableStreamResponse(_FakeResponse):
    """Streams chunks slowly so a cancel mid-flight is observable."""

    def __init__(
        self,
        chunk_count: int = 10,
        chunk_sleep_sec: float = 0.02,
        provider: str = "Friendli",
    ):
        super().__init__(status_code=200)
        self._chunk_count = chunk_count
        self._chunk_sleep_sec = chunk_sleep_sec
        self._provider = provider
        self.chunks_yielded = 0

    def iter_lines(self, decode_unicode: bool = True):
        # First chunk delivers a visible token so the transport sets ttft.
        yield "data: " + json.dumps(
            {"provider": self._provider, "choices": [{"delta": {"content": "hello"}}]}
        )
        self.chunks_yielded += 1
        for _ in range(self._chunk_count - 1):
            time.sleep(self._chunk_sleep_sec)
            yield "data: " + json.dumps(
                {"provider": self._provider, "choices": [{"delta": {"content": "more"}}]}
            )
            self.chunks_yielded += 1
        yield "data: [DONE]"


def test_cancel_event_closes_stream_mid_flight(monkeypatch) -> None:
    """Setting ``cancel_event`` mid-stream should close the response and
    return ``status='canceled'``. Token chunks after the cancel point must
    not be billed because the response was closed before draining them."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    response = _CancellableStreamResponse(chunk_count=20, chunk_sleep_sec=0.02)
    session = _FakeSession([response])
    transport = _transport(session)
    cancel = threading.Event()
    ttft = threading.Event()
    ttft_info: dict[str, object] = {}

    # Fire cancel ~50ms in, after first token but well before all chunks
    # would have been streamed (20 chunks * 20ms = 400ms).
    def trigger_cancel() -> None:
        ttft.wait(timeout=1.0)  # wait for first token
        time.sleep(0.05)
        cancel.set()

    canceler = threading.Thread(target=trigger_cancel, daemon=True)
    canceler.start()

    result = transport.send(
        prompt="x",
        max_tokens=8,
        timeout=5,
        ttft_event=ttft,
        ttft_info=ttft_info,
        cancel_event=cancel,
    )

    canceler.join(timeout=2.0)
    assert result.status == "canceled"
    assert result.error_message == "canceled_by_hedge_winner"
    assert result.billed_cost_usd == 0.0
    assert result.physical_cost_usd == 0.0
    assert result.cost_source == "canceled_no_usage_no_charge"
    assert response.closed is True
    assert response.chunks_yielded < 20  # ended early


def test_cancel_event_marks_continues_billing_provider_unmeasured(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    response = _CancellableStreamResponse(
        chunk_count=20,
        chunk_sleep_sec=0.02,
        provider="Minimax",
    )
    session = _FakeSession([response])
    cfg = TransportConfig(
        name="OR_Minimax",
        transport="openrouter",
        model="test/model",
        provider_hint="Minimax",
        base_url="https://example.test/v1",
        api_key_env="OPENROUTER_API_KEY",
        input_price_per_m=0.3,
        output_price_per_m=1.2,
        stream_cancel_billing="continues",
    )
    transport = OpenAICompatStreamingTransport(cfg, session)  # type: ignore[arg-type]
    cancel = threading.Event()
    ttft = threading.Event()
    ttft_info: dict[str, object] = {}

    def trigger_cancel() -> None:
        ttft.wait(timeout=1.0)
        time.sleep(0.05)
        cancel.set()

    canceler = threading.Thread(target=trigger_cancel, daemon=True)
    canceler.start()

    result = transport.send(
        prompt="x",
        max_tokens=8,
        timeout=5,
        ttft_event=ttft,
        ttft_info=ttft_info,
        cancel_event=cancel,
    )

    canceler.join(timeout=2.0)
    assert result.status == "canceled"
    assert result.billed_cost_usd == 0.0
    assert result.cost_source == "canceled_no_usage_billing_continues_unmeasured"


def test_unpinned_openrouter_cancel_uses_reported_provider_billing_mode(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    response = _CancellableStreamResponse(
        chunk_count=20,
        chunk_sleep_sec=0.02,
        provider="Friendli",
    )
    session = _FakeSession([response])
    cfg = TransportConfig(
        name="__or_auto__",
        transport="openrouter",
        model="test/model",
        base_url="https://example.test/v1",
        api_key_env="OPENROUTER_API_KEY",
        input_price_per_m=0.3,
        output_price_per_m=1.2,
        stream_cancel_billing="continues",
        stream_cancel_billing_by_provider={
            "Friendli": "stops",
            "Minimax": "continues",
        },
    )
    transport = OpenAICompatStreamingTransport(cfg, session)  # type: ignore[arg-type]
    cancel = threading.Event()
    ttft = threading.Event()

    def trigger_cancel() -> None:
        ttft.wait(timeout=1.0)
        time.sleep(0.05)
        cancel.set()

    canceler = threading.Thread(target=trigger_cancel, daemon=True)
    canceler.start()

    result = transport.send(
        prompt="x",
        max_tokens=8,
        timeout=5,
        ttft_event=ttft,
        ttft_info={},
        cancel_event=cancel,
    )

    canceler.join(timeout=2.0)
    assert result.status == "canceled"
    assert result.provider == "__or_auto__@Friendli"
    assert result.cost_source == "canceled_no_usage_no_charge"


def test_unpinned_openrouter_cancel_defaults_to_continues_when_provider_unmapped(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    response = _CancellableStreamResponse(
        chunk_count=20,
        chunk_sleep_sec=0.02,
        provider="Unmapped",
    )
    session = _FakeSession([response])
    cfg = TransportConfig(
        name="__or_auto__",
        transport="openrouter",
        model="test/model",
        base_url="https://example.test/v1",
        api_key_env="OPENROUTER_API_KEY",
        input_price_per_m=0.3,
        output_price_per_m=1.2,
        stream_cancel_billing="continues",
        stream_cancel_billing_by_provider={"Friendli": "stops"},
    )
    transport = OpenAICompatStreamingTransport(cfg, session)  # type: ignore[arg-type]
    cancel = threading.Event()
    ttft = threading.Event()

    def trigger_cancel() -> None:
        ttft.wait(timeout=1.0)
        time.sleep(0.05)
        cancel.set()

    canceler = threading.Thread(target=trigger_cancel, daemon=True)
    canceler.start()

    result = transport.send(
        prompt="x",
        max_tokens=8,
        timeout=5,
        ttft_event=ttft,
        ttft_info={},
        cancel_event=cancel,
    )

    canceler.join(timeout=2.0)
    assert result.status == "canceled"
    assert result.provider == "__or_auto__@Unmapped"
    assert result.cost_source == "canceled_no_usage_billing_continues_unmeasured"


def test_cancel_event_not_set_completes_normally(monkeypatch) -> None:
    """Baseline: same response with no cancel signal should drain to DONE."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    response = _CancellableStreamResponse(chunk_count=3, chunk_sleep_sec=0.001)
    session = _FakeSession([response])
    transport = _transport(session)

    result = transport.send(prompt="x", max_tokens=8, timeout=5)

    assert result.status == "success"
    assert response.chunks_yielded == 3
