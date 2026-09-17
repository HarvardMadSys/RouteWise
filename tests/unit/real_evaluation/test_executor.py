"""Checkpoint-hedging executor regressions."""

from __future__ import annotations

import time

import pytest

from experiments.real_evaluation.executor import send_checkpoint_hedged_request
from experiments.real_evaluation.transports import SingleRequestResult
from llm_routewise.core import CheckpointBackupDispatch


def _success_send(provider: str, ttft_ms: float = 200.0) -> SingleRequestResult:
    return SingleRequestResult(
        ttft_ms=ttft_ms,
        e2e_ms=ttft_ms + 100.0,
        status="success",
        provider=provider,
        prompt_tokens=10,
        completion_tokens=50,
        first_token_ts=time.time(),
        start_ts=time.time(),
    )


def test_checkpoint_hedge_dispatches_at_selector_selected_checkpoint() -> None:
    checkpoint_calls: list[tuple[float, float]] = []
    released: list[str] = []
    sent: list[str] = []

    def fake_send(
        provider, prompt, max_tokens, timeout, ttft_event, ttft_info, cancel_event=None
    ):
        del prompt, max_tokens, timeout, cancel_event
        sent.append(provider)
        if provider == "primary":
            time.sleep(0.08)
            if ttft_info is not None:
                ttft_info.update(
                    ttft_ms=80.0,
                    first_token_ts=time.time(),
                    status="success",
                )
            if ttft_event is not None:
                ttft_event.set()
            return _success_send(provider, ttft_ms=80.0)
        if ttft_info is not None:
            ttft_info.update(ttft_ms=5.0, first_token_ts=time.time(), status="success")
        if ttft_event is not None:
            ttft_event.set()
        return _success_send(provider, ttft_ms=5.0)

    def select_checkpoint_backup(
        elapsed_sec: float,
        checkpoint_ts: float,
    ) -> CheckpointBackupDispatch[str] | None:
        checkpoint_calls.append((elapsed_sec, checkpoint_ts))
        if elapsed_sec < 0.01:
            return None
        return CheckpointBackupDispatch(
            backup="backup",
            elapsed_sec=elapsed_sec,
            release=lambda: released.append("backup"),
        )

    hedged = send_checkpoint_hedged_request(
        send_fn=fake_send,
        primary_provider="primary",
        hedge_checkpoints_sec=(0.01, 0.02),
        checkpoint_backup_selector=select_checkpoint_backup,
        prompt="x",
        max_tokens=8,
        timeout=5,
    )

    assert hedged.hedge_triggered is True
    assert hedged.backup_provider == "backup"
    assert hedged.hedge_delay_sec == pytest.approx(0.01)
    assert hedged.hedge_checkpoint_ts == pytest.approx(checkpoint_calls[0][1])
    assert hedged.backup_dispatch_ts is not None
    assert sent[:2] == ["primary", "backup"]
    assert released == ["backup"]


def _streaming_send(first_token_after_sec: float, total_sec: float):
    """Fake transport: first token at ``first_token_after_sec``, stream end at
    ``total_sec``; honors ``cancel_event`` between 5 ms polls like the SSE loop."""

    def send(provider, prompt, max_tokens, timeout, ttft_event, ttft_info, cancel_event=None):
        del prompt, max_tokens, timeout
        start = time.time()
        first_token_ts = None
        while True:
            elapsed = time.time() - start
            if cancel_event is not None and cancel_event.is_set():
                status = "canceled"
                break
            if first_token_ts is None and elapsed >= first_token_after_sec:
                first_token_ts = time.time()
                if ttft_info is not None:
                    ttft_info.update(ttft_ms=elapsed * 1000.0, first_token_ts=first_token_ts, status="success")
                if ttft_event is not None:
                    ttft_event.set()
            if elapsed >= total_sec:
                status = "success"
                break
            time.sleep(0.005)
        if ttft_event is not None:
            ttft_event.set()
        return SingleRequestResult(
            ttft_ms=(first_token_ts - start) * 1000.0 if first_token_ts else -1.0,
            e2e_ms=(time.time() - start) * 1000.0,
            status=status,
            provider=provider,
            error_message="canceled_by_hedge_winner" if status == "canceled" else None,
            start_ts=start,
            first_token_ts=first_token_ts,
        )

    return send


def _run_race(observe: bool):
    sends = {"primary": _streaming_send(0.25, 0.6), "backup": _streaming_send(0.02, 0.3)}

    def fake_send(provider, **kwargs):
        return sends[provider](provider, **kwargs)

    return send_checkpoint_hedged_request(
        send_fn=fake_send,
        primary_provider="primary",
        hedge_checkpoints_sec=(0.05,),
        checkpoint_backup_selector=lambda elapsed, ts: CheckpointBackupDispatch(
            backup="backup", elapsed_sec=elapsed
        ),
        prompt="x",
        max_tokens=8,
        timeout=5,
        observe_loser_first_token=observe,
    )


def test_hedge_loser_is_canceled_before_first_token_by_default() -> None:
    hedged = _run_race(observe=False)

    assert hedged.hedge_triggered and hedged.winner == "backup"
    assert hedged.backup_result is not None and hedged.backup_result.status == "success"
    assert hedged.primary_result.status == "canceled"
    assert hedged.primary_result.first_token_ts is None  # counterfactual TTFT unobserved


def test_hedge_loser_first_token_is_observed_when_requested() -> None:
    hedged = _run_race(observe=True)

    assert hedged.hedge_triggered and hedged.winner == "backup"
    assert hedged.primary_result.status == "canceled"  # still canceled, just later
    assert hedged.primary_result.first_token_ts is not None
    assert hedged.primary_result.ttft_ms == pytest.approx(250.0, abs=60.0)
    assert hedged.primary_result.e2e_ms < 550.0  # canceled before the stream would have ended
    assert hedged.primary_ttft_info.get("ttft_ms", -1.0) > 0
