"""Transport-level retry semantics."""

from __future__ import annotations

import json
import threading
import time

import pytest

from experiments.real_evaluation.transports import (
    OpenAICompatStreamingTransport,
    TransportConfig,
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
    )
    return OpenAICompatStreamingTransport(cfg, session)  # type: ignore[arg-type]


def test_http_429_is_retried_and_retry_delay_counts_toward_ttft(monkeypatch) -> None:
    """A transient 429 is an execution retry, not a final provider miss.

    The final success keeps ``status=success`` and TTFT/E2E are measured from
    the first attempt, including rate-limit backoff.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    sleep_calls: list[float] = []
    real_sleep = __import__("time").sleep

    def short_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        real_sleep(min(seconds, 0.01))

    monkeypatch.setattr("experiments.real_evaluation.transports.time.sleep", short_sleep)
    session = _FakeSession(
        [
            _FakeResponse(
                429,
                headers={"Retry-After": "0.01"},
                payload={"error": {"message": "rate limited"}},
            ),
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
                        "provider": "actual-provider",
                    }
                ],
            ),
        ]
    )
    event = threading.Event()
    ttft_info: dict = {}

    result = _transport(session).send(
        prompt="x",
        max_tokens=8,
        timeout=5,
        ttft_event=event,
        ttft_info=ttft_info,
    )

    assert len(session.calls) == 2
    assert sleep_calls == [0.01]
    assert result.status == "success"
    assert result.rate_limited is True
    assert result.retry_count == 1
    assert result.retry_sleep_ms == 10.0
    assert result.ttft_ms >= result.retry_sleep_ms
    assert result.e2e_ms >= result.ttft_ms
    assert result.billed_cost_usd == 0.000123
    assert result.provider == "OR_Test@actual-provider"
    assert event.is_set()
    assert ttft_info["status"] == "success"


def test_http_429_final_failure_only_after_retry_budget_exhausted(monkeypatch) -> None:
    """If the rate limit never clears within the request timeout, only then
    does the transport return a final HTTP 429 failure sample."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    session = _FakeSession(
        [
            _FakeResponse(
                429,
                headers={"Retry-After": "10"},
                payload={"error": {"message": "rate limited"}},
            ),
            _FakeResponse(
                429,
                headers={"Retry-After": "10"},
                payload={"error": {"message": "still limited"}},
            ),
        ]
    )

    result = _transport(session).send(prompt="x", max_tokens=8, timeout=0.005)

    assert result.status == "HTTP 429"
    assert result.rate_limited is True
    assert result.retry_count >= 1
    assert result.retry_sleep_ms >= 1.0
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
