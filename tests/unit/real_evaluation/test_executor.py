"""Checkpoint-hedging executor regressions."""

from __future__ import annotations

import time

import pytest

from experiments.real_evaluation.executor import send_checkpoint_hedged_request
from experiments.real_evaluation.transports import SingleRequestResult
from rwsim.core import CheckpointBackupDispatch


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
