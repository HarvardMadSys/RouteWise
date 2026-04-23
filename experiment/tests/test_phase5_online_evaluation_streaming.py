#!/usr/bin/env python3
"""Unit tests for Phase 5 online evaluation streaming and cost parsing.

These tests focus on the highest-risk logic that affects real-money evaluation:
- SSE parsing for TTFT, provider, and OpenRouter usage.cost
- stream_options retry behavior when unsupported
- smart_hedge TTFT aggregation relative to primary start time
- analysis stats for real vs estimated cost
"""

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import phase5_online_evaluation
from phase5_online_evaluation import (
    EvaluationConfig,
    Phase5OnlineEvaluator,
    RequestResult,
    SingleRequestResult,
)


class _FakeResponse:
    """Minimal fake requests.Response for unit testing."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        lines: list[bytes] | None = None,
        json_data: dict[str, Any] | None = None,
        text: str = "",
    ):
        self.status_code = status_code
        self.headers = headers or {}
        self._lines = lines or []
        self._json_data = json_data or {}
        self.text = text

    def iter_lines(self):
        yield from self._lines

    def json(self):
        return self._json_data

    def close(self) -> None:
        return


class _FakeSession:
    """Minimal fake requests.Session for unit testing."""

    def __init__(self, responses: list[_FakeResponse]):
        self._responses = list(responses)
        self.payloads: list[dict[str, Any]] = []

    def post(
        self, *, url: str, headers: dict[str, str], json: dict[str, Any], timeout: int, stream: bool
    ):
        del url, headers, timeout, stream
        # The evaluator mutates the payload dict (e.g., pops stream_options for retry),
        # so store a deep copy for stable assertions.
        self.payloads.append(copy.deepcopy(json))
        if not self._responses:
            raise RuntimeError("No more fake responses configured.")
        return self._responses.pop(0)


@pytest.fixture
def sample_pricing() -> dict[str, float]:
    """Provide sample pricing data."""
    return {"Groq": 0.00003, "Parasail": 0.00002}


@pytest.fixture
def evaluator(tmp_path: Path, sample_pricing: dict[str, float]) -> Phase5OnlineEvaluator:
    """Provide a Phase5OnlineEvaluator instance."""
    config = EvaluationConfig(slo_sec=2.0, warmup_sec=0, duration_sec=0, dispatch_overhead_sec=0.0)
    return Phase5OnlineEvaluator(config=config, pricing=sample_pricing, output_dir=tmp_path)


def _sse(obj: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(obj)}".encode()


class TestSendRequestStreaming:
    """Tests for streaming send request functionality."""

    def test_send_request_parses_ttft_and_cost(self, evaluator: Phase5OnlineEvaluator):
        """Test that send request correctly parses TTFT and cost."""
        lines = [
            _sse({"choices": [{"delta": {"content": "Hello"}}]}),
            _sse({"usage": {"prompt_tokens": 5, "completion_tokens": 7, "cost": 0.00003059}}),
            b"data: [DONE]",
        ]
        fake_session = _FakeSession(
            [
                _FakeResponse(
                    status_code=200, headers={"X-OpenRouter-Provider": "Groq"}, lines=lines
                ),
            ]
        )

        with (
            patch.object(evaluator, "_get_session", return_value=fake_session),
            patch(
                "phase5_online_evaluation.time.time", side_effect=[1000.0, 1000.5, 1000.7, 1000.8]
            ),
        ):
            result = evaluator._send_request(provider="Groq", timeout=10)

        assert result.status == "success"
        assert result.actual_provider == "Groq"
        assert result.prompt_tokens == 5
        assert result.completion_tokens == 7
        assert result.cost == pytest.approx(0.00003059, rel=1e-9)
        assert result.ttft_ms == pytest.approx(500.0, rel=1e-6)
        assert result.e2e_ms == pytest.approx(700.0, rel=1e-6)
        assert len(fake_session.payloads) == 1

    def test_send_request_retries_without_stream_options(self, evaluator: Phase5OnlineEvaluator):
        """Test that send request retries without stream options on error."""
        error = _FakeResponse(
            status_code=400,
            json_data={"error": {"message": "stream_options is not supported"}},
            text="error",
        )
        lines = [
            _sse({"choices": [{"delta": {"content": "OK"}}]}),
            _sse({"usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.00001}}),
            b"data: [DONE]",
        ]
        success = _FakeResponse(status_code=200, lines=lines)
        fake_session = _FakeSession([error, success])

        with (
            patch.object(evaluator, "_get_session", return_value=fake_session),
            patch(
                "phase5_online_evaluation.time.time", side_effect=[1000.0, 1000.1, 1000.2, 1000.3]
            ),
        ):
            result = evaluator._send_request(provider="Groq", timeout=10)

        assert result.status == "success"
        assert len(fake_session.payloads) == 2
        assert "stream_options" in fake_session.payloads[0]
        assert "stream_options" not in fake_session.payloads[1]

    def test_send_request_falls_back_to_requested_provider_if_unrecognized(
        self, evaluator: Phase5OnlineEvaluator
    ):
        """Test fallback to requested provider when actual is unrecognized."""
        lines = [
            _sse({"choices": [{"delta": {"content": "Hi"}}]}),
            _sse({"usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.00001}}),
            b"data: [DONE]",
        ]
        fake_session = _FakeSession(
            [
                _FakeResponse(status_code=200, headers={"X-Provider": "NotInPricing"}, lines=lines),
            ]
        )

        with (
            patch.object(evaluator, "_get_session", return_value=fake_session),
            patch(
                "phase5_online_evaluation.time.time", side_effect=[1000.0, 1000.2, 1000.3, 1000.4]
            ),
        ):
            result = evaluator._send_request(provider="Groq", timeout=10)

        assert result.status == "success"
        assert result.actual_provider == "Groq"


class TestSmartHedgeAggregation:
    """Tests for smart hedge aggregation logic."""

    def test_smart_hedge_ttft_is_relative_to_primary_start(self, evaluator: Phase5OnlineEvaluator):
        """Test that smart hedge TTFT is relative to primary start time."""
        now = 1234.0
        evaluator.router.route = lambda _now: "Parasail"

        primary = SingleRequestResult(
            ttft_ms=900.0,
            e2e_ms=1200.0,
            status="success",
            actual_provider="Parasail",
            error_message=None,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.00002,
            start_ts=1000.0,
            first_token_ts=1000.9,
        )
        backup = SingleRequestResult(
            ttft_ms=200.0,
            e2e_ms=300.0,
            status="success",
            actual_provider="Groq",
            error_message=None,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.00003,
            start_ts=1000.5,
            first_token_ts=1000.7,
        )

        with (
            patch.object(phase5_online_evaluation, "select_backup", return_value="Groq"),
            patch.object(evaluator.hedger, "compute_hedge_time", return_value=0.2),
            patch.object(
                evaluator, "_send_hedged_request", return_value=(primary, backup, "backup", True)
            ),
        ):
            result = evaluator._execute_policy_request("smart_hedge", now)

        assert result.hedge_triggered
        assert result.hedge_winner == "backup"
        assert result.selected_provider == "Parasail"
        assert result.actual_provider == "Groq"
        assert result.ttft_ms == pytest.approx(700.0, rel=1e-9)  # 1000.7 - 1000.0
        assert result.e2e_ms == pytest.approx(800.0, rel=1e-9)  # 1000.8 - 1000.0
        assert result.cost_usd == pytest.approx(0.00005, rel=1e-12)
        assert not result.cost_is_estimated
        assert result.prompt_tokens == 20
        assert result.completion_tokens == 10

    def test_smart_hedge_cost_is_estimated_if_any_cost_missing(
        self, evaluator: Phase5OnlineEvaluator
    ):
        """Test that cost is marked estimated when any cost is missing."""
        now = 1234.0
        evaluator.router.route = lambda _now: "Parasail"

        primary = SingleRequestResult(
            ttft_ms=900.0,
            e2e_ms=1200.0,
            status="success",
            actual_provider="Parasail",
            error_message=None,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.00002,
            start_ts=1000.0,
            first_token_ts=1000.9,
        )
        backup = SingleRequestResult(
            ttft_ms=200.0,
            e2e_ms=300.0,
            status="success",
            actual_provider="Groq",
            error_message=None,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.0,  # Missing usage.cost
            start_ts=1000.5,
            first_token_ts=1000.7,
        )

        with (
            patch.object(phase5_online_evaluation, "select_backup", return_value="Groq"),
            patch.object(evaluator.hedger, "compute_hedge_time", return_value=0.2),
            patch.object(
                evaluator, "_send_hedged_request", return_value=(primary, backup, "backup", True)
            ),
        ):
            result = evaluator._execute_policy_request("smart_hedge", now)

        assert result.hedge_triggered
        assert result.cost_usd == pytest.approx(0.00002, rel=1e-12)
        assert result.cost_is_estimated


class TestAnalysisRealVsEstimatedCost:
    """Tests for analysis of real vs estimated cost."""

    def test_avg_cost_real_excludes_estimated_costs(self, evaluator: Phase5OnlineEvaluator):
        """Test that average cost calculation excludes estimated costs."""
        results = [
            RequestResult(
                request_id="r1",
                timestamp=1.0,
                policy="lp_mix",
                selected_provider="Groq",
                actual_provider="Groq",
                ttft_ms=100.0,
                e2e_ms=120.0,
                status="success",
                slo_violated=False,
                cost_usd=1.0,
                cost_is_estimated=False,
            ),
            RequestResult(
                request_id="r2",
                timestamp=2.0,
                policy="lp_mix",
                selected_provider="Groq",
                actual_provider="Groq",
                ttft_ms=100.0,
                e2e_ms=120.0,
                status="success",
                slo_violated=False,
                cost_usd=2.0,
                cost_is_estimated=False,
            ),
            RequestResult(
                request_id="r3",
                timestamp=3.0,
                policy="lp_mix",
                selected_provider="Groq",
                actual_provider="Groq",
                ttft_ms=100.0,
                e2e_ms=120.0,
                status="success",
                slo_violated=False,
                cost_usd=0.0,
                cost_is_estimated=True,
            ),
        ]
        evaluator.policies["lp_mix"].results.extend(results)
        evaluator.policies["lp_mix"].total_cost = sum(r.cost_usd for r in results)

        stats = evaluator.analyze_results()
        assert stats["lp_mix"]["real_cost_count"] == 2
        assert stats["lp_mix"]["estimated_cost_count"] == 1
        assert stats["lp_mix"]["avg_cost_real"] == pytest.approx(1.5, rel=1e-12)
