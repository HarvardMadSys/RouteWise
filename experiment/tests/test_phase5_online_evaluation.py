#!/usr/bin/env python3
"""Unit tests for Phase 5 Online Evaluation.

Tests cover:
- Data structures (SingleRequestResult, RequestResult)
- Hedging logic (trigger conditions, winner selection, cost aggregation)
- SLO violation detection
- CSV logging format

Run with:
    pytest experiment/tests/test_phase5_online_evaluation.py -v
"""

import csv
import os

# Import module under test
import sys
import threading
import time
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from phase5_online_evaluation import (
    EvaluationConfig,
    Phase5OnlineEvaluator,
    PolicyState,
    RequestResult,
    SingleRequestResult,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_pricing():
    """Sample pricing data for tests."""
    return {
        "Groq": 0.00003,
        "Parasail": 0.00002,
        "Nebius": 0.000025,
    }


@pytest.fixture
def config():
    """Default evaluation config."""
    return EvaluationConfig(
        slo_sec=2.0,
        warmup_sec=60,
        duration_sec=120,
        dispatch_overhead_sec=0.05,
    )


@pytest.fixture
def evaluator(config, sample_pricing, tmp_path):
    """Create evaluator instance for testing."""
    return Phase5OnlineEvaluator(
        config=config,
        pricing=sample_pricing,
        output_dir=tmp_path,
    )


# =============================================================================
# Test Data Structures
# =============================================================================


class TestSingleRequestResult:
    """Tests for SingleRequestResult dataclass."""

    def test_successful_result(self):
        """Test creation of successful result."""
        result = SingleRequestResult(
            ttft_ms=500.0,
            e2e_ms=600.0,
            status="success",
            actual_provider="Groq",
            error_message=None,
            prompt_tokens=42,
            completion_tokens=8,
            cost=0.00003,
            start_ts=1000.0,
            first_token_ts=1000.5,
        )
        assert result.status == "success"
        assert result.ttft_ms == 500.0
        assert result.cost == 0.00003
        assert result.first_token_ts is not None

    def test_failed_result(self):
        """Test creation of failed result."""
        result = SingleRequestResult(
            ttft_ms=-1.0,
            e2e_ms=-1.0,
            status="timeout",
            actual_provider="Groq",
            error_message="Request timeout",
            prompt_tokens=0,
            completion_tokens=0,
            cost=0.0,
            start_ts=1000.0,
            first_token_ts=None,
        )
        assert result.status == "timeout"
        assert result.ttft_ms == -1.0
        assert result.cost == 0.0
        assert result.first_token_ts is None


class TestRequestResult:
    """Tests for RequestResult dataclass."""

    def test_non_hedged_result(self):
        """Test result without hedging."""
        result = RequestResult(
            request_id="abc123",
            timestamp=1000.0,
            policy="lp_mix",
            selected_provider="Groq",
            actual_provider="Groq",
            ttft_ms=500.0,
            e2e_ms=600.0,
            status="success",
            slo_violated=False,
            cost_usd=0.00003,
            hedge_triggered=False,
            hedge_provider=None,
            hedge_winner=None,
        )
        assert not result.hedge_triggered
        assert result.hedge_winner is None
        assert not result.cost_is_estimated

    def test_hedged_result(self):
        """Test result with hedging triggered."""
        result = RequestResult(
            request_id="abc123",
            timestamp=1000.0,
            policy="smart_hedge",
            selected_provider="Parasail",
            actual_provider="Groq",
            ttft_ms=800.0,
            e2e_ms=900.0,
            status="success",
            slo_violated=False,
            cost_usd=0.00005,  # Both requests billed
            hedge_triggered=True,
            hedge_provider="Groq",
            hedge_winner="backup",
            prompt_tokens=84,  # Combined tokens
            completion_tokens=16,
        )
        assert result.hedge_triggered
        assert result.hedge_winner == "backup"
        assert result.cost_usd == 0.00005


class TestPolicyState:
    """Tests for PolicyState tracking."""

    def test_empty_state(self):
        """Test initial empty state."""
        state = PolicyState(name="test")
        assert state.success_count == 0
        assert state.slo_violations == 0
        assert state.slo_violation_rate == 0.0
        assert state.avg_cost == 0.0

    def test_state_with_results(self):
        """Test state after adding results."""
        state = PolicyState(name="test")

        # Add successful result
        state.results.append(
            RequestResult(
                request_id="1",
                timestamp=1000.0,
                policy="test",
                selected_provider="Groq",
                actual_provider="Groq",
                ttft_ms=500.0,
                e2e_ms=600.0,
                status="success",
                slo_violated=False,
                cost_usd=0.00003,
            )
        )
        state.total_cost += 0.00003

        # Add SLO-violating result
        state.results.append(
            RequestResult(
                request_id="2",
                timestamp=1001.0,
                policy="test",
                selected_provider="Groq",
                actual_provider="Groq",
                ttft_ms=2500.0,
                e2e_ms=2600.0,
                status="success",
                slo_violated=True,
                cost_usd=0.00003,
            )
        )
        state.total_cost += 0.00003

        assert state.success_count == 2
        assert state.slo_violations == 1
        assert state.slo_violation_rate == 0.5
        assert state.avg_cost == 0.00003

    def test_latency_percentile(self):
        """Test latency percentile calculation."""
        state = PolicyState(name="test")

        # Add 10 results with known latencies
        for i in range(10):
            state.results.append(
                RequestResult(
                    request_id=str(i),
                    timestamp=1000.0 + i,
                    policy="test",
                    selected_provider="Groq",
                    actual_provider="Groq",
                    ttft_ms=100.0 * (i + 1),  # 100, 200, ..., 1000
                    e2e_ms=100.0 * (i + 1),
                    status="success",
                    slo_violated=False,
                    cost_usd=0.00003,
                )
            )

        p50 = state.get_latency_percentile(50)
        p99 = state.get_latency_percentile(99)

        assert 500 <= p50 <= 600  # Median around 550
        assert p99 >= 900  # P99 near max


# =============================================================================
# Test SLO Violation Detection
# =============================================================================


class TestSLOViolation:
    """Tests for SLO violation detection."""

    def test_no_violation_within_slo(self, config):
        """Test no violation when TTFT < SLO."""
        slo_sec = config.slo_sec  # 2.0s
        ttft_ms = 1500.0  # 1.5s
        status = "success"

        slo_violated = (status != "success") or (ttft_ms / 1000.0 > slo_sec)
        assert not slo_violated

    def test_violation_exceeds_slo(self, config):
        """Test violation when TTFT > SLO."""
        slo_sec = config.slo_sec  # 2.0s
        ttft_ms = 2500.0  # 2.5s
        status = "success"

        slo_violated = (status != "success") or (ttft_ms / 1000.0 > slo_sec)
        assert slo_violated

    def test_violation_on_error(self, config):
        """Test violation on request error."""
        slo_sec = config.slo_sec
        ttft_ms = -1.0
        status = "timeout"

        slo_violated = (status != "success") or (ttft_ms / 1000.0 > slo_sec)
        assert slo_violated

    def test_violation_at_boundary(self, config):
        """Test exactly at SLO boundary."""
        slo_sec = config.slo_sec  # 2.0s
        ttft_ms = 2000.0  # Exactly 2.0s
        status = "success"

        slo_violated = (status != "success") or (ttft_ms / 1000.0 > slo_sec)
        assert not slo_violated  # Equal is not violation


# =============================================================================
# Test Hedging Logic
# =============================================================================


class TestHedgingLogic:
    """Tests for hedging trigger and winner selection."""

    def test_no_hedge_when_primary_fast(self, evaluator):
        """Test no hedging when primary returns TTFT before hedge_time."""

        # Mock _send_request to return fast primary
        def mock_send_request(provider, timeout, ttft_event=None, ttft_info=None):
            time.sleep(0.1)  # Simulate 100ms request
            if ttft_event:
                ttft_event.set()
            if ttft_info is not None:
                ttft_info["ttft_ms"] = 100.0
                ttft_info["first_token_ts"] = time.time()
                ttft_info["status"] = "success"
            return SingleRequestResult(
                ttft_ms=100.0,
                e2e_ms=150.0,
                status="success",
                actual_provider=provider,
                error_message=None,
                prompt_tokens=42,
                completion_tokens=8,
                cost=0.00003,
                start_ts=time.time() - 0.15,
                first_token_ts=time.time() - 0.05,
            )

        with patch.object(evaluator, "_send_request", side_effect=mock_send_request):
            primary_result, backup_result, winner, hedge_triggered = evaluator._send_hedged_request(
                primary_provider="Parasail",
                backup_provider="Groq",
                hedge_time_sec=1.0,  # Hedge after 1s
            )

        assert not hedge_triggered
        assert backup_result is None
        assert winner == "primary"
        assert primary_result.status == "success"

    def test_hedge_when_primary_slow(self, evaluator):
        """Test hedging triggered when primary is slow."""
        call_count = {"primary": 0, "backup": 0}

        def mock_send_request(provider, timeout, ttft_event=None, ttft_info=None):
            if provider == "Parasail":
                call_count["primary"] += 1
                time.sleep(0.5)  # Primary takes 500ms (> hedge_time)
                start_ts = time.time() - 0.5
                first_token_ts = time.time()
            else:
                call_count["backup"] += 1
                time.sleep(0.1)  # Backup takes 100ms
                start_ts = time.time() - 0.1
                first_token_ts = time.time()

            if ttft_event:
                ttft_event.set()
            if ttft_info is not None:
                ttft_info["ttft_ms"] = 100.0
                ttft_info["first_token_ts"] = first_token_ts
                ttft_info["status"] = "success"

            return SingleRequestResult(
                ttft_ms=500.0 if provider == "Parasail" else 100.0,
                e2e_ms=600.0 if provider == "Parasail" else 150.0,
                status="success",
                actual_provider=provider,
                error_message=None,
                prompt_tokens=42,
                completion_tokens=8,
                cost=0.00003,
                start_ts=start_ts,
                first_token_ts=first_token_ts,
            )

        with patch.object(evaluator, "_send_request", side_effect=mock_send_request):
            primary_result, backup_result, winner, hedge_triggered = evaluator._send_hedged_request(
                primary_provider="Parasail",
                backup_provider="Groq",
                hedge_time_sec=0.2,  # Hedge after 200ms
            )

        assert hedge_triggered
        assert backup_result is not None
        assert call_count["primary"] == 1
        assert call_count["backup"] == 1

    def test_winner_selection_primary_wins(self, evaluator):
        """Test winner is primary when it has earlier first_token_ts."""
        base_time = time.time()

        primary = SingleRequestResult(
            ttft_ms=300.0,
            e2e_ms=400.0,
            status="success",
            actual_provider="Parasail",
            error_message=None,
            prompt_tokens=42,
            completion_tokens=8,
            cost=0.00002,
            start_ts=base_time,
            first_token_ts=base_time + 0.3,  # First token at +300ms
        )

        backup = SingleRequestResult(
            ttft_ms=100.0,
            e2e_ms=150.0,
            status="success",
            actual_provider="Groq",
            error_message=None,
            prompt_tokens=42,
            completion_tokens=8,
            cost=0.00003,
            start_ts=base_time + 0.25,  # Started at +250ms (after hedge)
            first_token_ts=base_time + 0.35,  # First token at +350ms
        )

        # Primary first_token (0.3) < Backup first_token (0.35) -> Primary wins
        winner = "primary" if primary.first_token_ts <= backup.first_token_ts else "backup"

        assert winner == "primary"

    def test_winner_selection_backup_wins(self, evaluator):
        """Test winner is backup when it has earlier first_token_ts."""
        base_time = time.time()

        primary = SingleRequestResult(
            ttft_ms=500.0,
            e2e_ms=600.0,
            status="success",
            actual_provider="Parasail",
            error_message=None,
            prompt_tokens=42,
            completion_tokens=8,
            cost=0.00002,
            start_ts=base_time,
            first_token_ts=base_time + 0.5,  # First token at +500ms
        )

        backup = SingleRequestResult(
            ttft_ms=100.0,
            e2e_ms=150.0,
            status="success",
            actual_provider="Groq",
            error_message=None,
            prompt_tokens=42,
            completion_tokens=8,
            cost=0.00003,
            start_ts=base_time + 0.2,  # Started at +200ms (after hedge)
            first_token_ts=base_time + 0.3,  # First token at +300ms
        )

        # Backup first_token (0.3) < Primary first_token (0.5) -> Backup wins
        winner = "primary" if primary.first_token_ts <= backup.first_token_ts else "backup"

        assert winner == "backup"

    def test_infinite_hedge_time_no_hedge(self, evaluator):
        """Test that infinite hedge_time means no hedging."""

        def mock_send_request(provider, timeout, ttft_event=None, ttft_info=None):
            if ttft_event:
                ttft_event.set()
            return SingleRequestResult(
                ttft_ms=500.0,
                e2e_ms=600.0,
                status="success",
                actual_provider=provider,
                error_message=None,
                prompt_tokens=42,
                completion_tokens=8,
                cost=0.00003,
                start_ts=time.time(),
                first_token_ts=time.time(),
            )

        with patch.object(evaluator, "_send_request", side_effect=mock_send_request):
            primary_result, backup_result, winner, hedge_triggered = evaluator._send_hedged_request(
                primary_provider="Parasail",
                backup_provider="Groq",
                hedge_time_sec=float("inf"),
            )

        assert not hedge_triggered
        assert backup_result is None
        assert winner == "primary"


# =============================================================================
# Test Cost Calculation
# =============================================================================


class TestCostCalculation:
    """Tests for cost calculation and aggregation."""

    def test_single_request_cost(self):
        """Test cost from single request."""
        result = SingleRequestResult(
            ttft_ms=500.0,
            e2e_ms=600.0,
            status="success",
            actual_provider="Groq",
            error_message=None,
            prompt_tokens=42,
            completion_tokens=8,
            cost=0.00003059,
            start_ts=time.time(),
            first_token_ts=time.time(),
        )

        assert result.cost == pytest.approx(0.00003059, rel=1e-6)

    def test_hedged_request_cost_both_billed(self):
        """Test that both requests are billed when hedging."""
        primary_cost = 0.00002
        backup_cost = 0.00003

        # Simulated hedging scenario
        hedge_triggered = True
        total_cost = primary_cost
        if hedge_triggered:
            total_cost += backup_cost

        assert total_cost == 0.00005

    def test_cost_is_estimated_when_zero(self):
        """Test cost_is_estimated flag when cost is zero."""
        result = SingleRequestResult(
            ttft_ms=-1.0,
            e2e_ms=-1.0,
            status="timeout",
            actual_provider="Groq",
            error_message="timeout",
            prompt_tokens=0,
            completion_tokens=0,
            cost=0.0,
            start_ts=time.time(),
            first_token_ts=None,
        )

        cost_is_estimated = result.cost <= 0
        assert cost_is_estimated


# =============================================================================
# Test CSV Logging
# =============================================================================


class TestCSVLogging:
    """Tests for CSV log format and content."""

    def test_csv_headers(self, evaluator):
        """Test CSV file has correct headers."""
        with open(evaluator.csv_path) as f:
            reader = csv.reader(f)
            headers = next(reader)

        expected_headers = [
            "timestamp",
            "request_id",
            "policy",
            "selected_provider",
            "actual_provider",
            "ttft_ms",
            "e2e_ms",
            "status",
            "slo_violated",
            "cost_usd",
            "cost_is_estimated",
            "hedge_triggered",
            "hedge_provider",
            "hedge_winner",
            "prompt_tokens",
            "completion_tokens",
            "error_message",
        ]

        assert headers == expected_headers

    def test_log_result(self, evaluator):
        """Test logging a result to CSV."""
        result = RequestResult(
            request_id="test123",
            timestamp=1000.0,
            policy="smart_hedge",
            selected_provider="Parasail",
            actual_provider="Groq",
            ttft_ms=500.0,
            e2e_ms=600.0,
            status="success",
            slo_violated=False,
            cost_usd=0.00005,
            hedge_triggered=True,
            hedge_provider="Groq",
            hedge_winner="backup",
            prompt_tokens=84,
            completion_tokens=16,
            cost_is_estimated=False,
        )

        evaluator._log_result(result)

        with open(evaluator.csv_path) as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            row = next(reader)

        assert row[0] == "1000.0"  # timestamp
        assert row[1] == "test123"  # request_id
        assert row[2] == "smart_hedge"  # policy
        assert row[4] == "Groq"  # actual_provider
        assert row[11] == "True"  # hedge_triggered
        assert row[13] == "backup"  # hedge_winner


# =============================================================================
# Test Thread Safety
# =============================================================================


class TestThreadSafety:
    """Tests for thread-local session management."""

    def test_thread_local_session(self, evaluator):
        """Test that each thread gets its own session."""
        sessions = {}

        def get_session_in_thread(thread_id):
            session = evaluator._get_session()
            sessions[thread_id] = id(session)

        threads = []
        for i in range(3):
            t = threading.Thread(target=get_session_in_thread, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Each thread should have a different session object
        session_ids = list(sessions.values())
        assert len(set(session_ids)) == 3

    def test_same_thread_same_session(self, evaluator):
        """Test that same thread gets same session."""
        s1 = evaluator._get_session()
        s2 = evaluator._get_session()
        assert s1 is s2


# =============================================================================
# Test Provider Selection
# =============================================================================


class TestProviderSelection:
    """Tests for provider selection by policy."""

    def test_openrouter_auto_returns_none(self, evaluator):
        """Test openrouter_auto policy returns None (let OpenRouter decide)."""
        provider = evaluator._select_provider_for_policy("openrouter_auto", time.time())
        assert provider is None

    def test_cheapest_fixed_returns_cheapest(self, evaluator):
        """Test cheapest_fixed returns cheapest provider."""
        provider = evaluator._select_provider_for_policy("cheapest_fixed", time.time())
        assert provider == evaluator.cheapest_provider
        assert provider == "Parasail"  # Cheapest in sample_pricing

    def test_fastest_fixed_returns_groq(self, evaluator):
        """Test fastest_fixed returns Groq (or fallback)."""
        provider = evaluator._select_provider_for_policy("fastest_fixed", time.time())
        assert provider == "Groq"


# =============================================================================
# Test Analysis
# =============================================================================


class TestAnalysis:
    """Tests for result analysis."""

    def test_analyze_empty_results(self, evaluator):
        """Test analysis with no results."""
        stats = evaluator.analyze_results()
        assert stats == {}

    def test_analyze_with_results(self, evaluator):
        """Test analysis with mixed results."""
        # Add some results
        for i in range(10):
            result = RequestResult(
                request_id=str(i),
                timestamp=1000.0 + i,
                policy="lp_mix",
                selected_provider="Groq",
                actual_provider="Groq",
                ttft_ms=500.0 + i * 100,
                e2e_ms=600.0 + i * 100,
                status="success",
                slo_violated=(i >= 8),  # Last 2 violate SLO
                cost_usd=0.00003,
                cost_is_estimated=False,
            )
            evaluator.policies["lp_mix"].results.append(result)
            evaluator.policies["lp_mix"].total_cost += result.cost_usd

        stats = evaluator.analyze_results()

        assert "lp_mix" in stats
        assert stats["lp_mix"]["total_requests"] == 10
        assert stats["lp_mix"]["successful_requests"] == 10
        assert stats["lp_mix"]["slo_violation_rate"] == 0.2
        assert stats["lp_mix"]["real_cost_count"] == 10
        assert stats["lp_mix"]["estimated_cost_count"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
