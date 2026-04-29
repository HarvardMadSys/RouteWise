"""Hedging-executor regressions.

Two behaviours pinned here:

1. Phase5 trigger semantics — backup must fire when primary returns an
   HTTP error before any visible token, even though the transport sets
   ``ttft_event`` in its ``finally`` block.

2. ``on_backup_dispatch`` callback — must run *before* the backup
   thread starts, so the runner can charge concurrency / quota at the
   moment the backup actually goes on the wire (not after the hedged
   request returns).
"""

from __future__ import annotations

import threading
import time

from experiments.real_evaluation.executor import send_hedged_request
from experiments.real_evaluation.transports import SingleRequestResult


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


def _http_error_send(provider: str) -> SingleRequestResult:
    return SingleRequestResult(
        ttft_ms=-1.0,
        e2e_ms=-1.0,
        status="HTTP 500",
        provider=provider,
        error_message="server_error",
        start_ts=time.time(),
    )


def test_primary_http_error_triggers_backup() -> None:
    """Primary fails BEFORE any token: ttft_event is set but ttft_ms<=0.
    The phase5 trigger check must still launch the backup."""

    def fake_send(
        provider, prompt, max_tokens, timeout, ttft_event, ttft_info
    ):
        if provider == "primary":
            if ttft_event is not None:
                ttft_event.set()
            if ttft_info is not None:
                ttft_info["status"] = "HTTP 500"
            return _http_error_send(provider)
        if ttft_info is not None:
            ttft_info.update(
                ttft_ms=200.0, first_token_ts=time.time(), status="success"
            )
        if ttft_event is not None:
            ttft_event.set()
        return _success_send(provider)

    hedged = send_hedged_request(
        send_fn=fake_send,
        primary_provider="primary",
        backup_provider="backup",
        hedge_delay_sec=0.5,
        prompt="x",
        max_tokens=8,
        timeout=5,
        dispatch_overhead_sec=0.0,
    )
    assert hedged.hedge_triggered is True
    assert hedged.winner == "backup"
    assert hedged.backup_result is not None
    assert hedged.backup_result.status == "success"


def test_primary_succeeds_quickly_no_hedge() -> None:
    """When primary returns a valid TTFT before the deadline, backup
    must NOT be dispatched."""

    def fake_send(
        provider, prompt, max_tokens, timeout, ttft_event, ttft_info
    ):
        # Primary signals fast.
        if ttft_info is not None:
            ttft_info.update(
                ttft_ms=50.0, first_token_ts=time.time(), status="success"
            )
        if ttft_event is not None:
            ttft_event.set()
        return _success_send(provider, ttft_ms=50.0)

    hedged = send_hedged_request(
        send_fn=fake_send,
        primary_provider="primary",
        backup_provider="backup",
        hedge_delay_sec=2.0,
        prompt="x",
        max_tokens=8,
        timeout=5,
        dispatch_overhead_sec=0.0,
    )
    assert hedged.hedge_triggered is False
    assert hedged.winner == "primary"
    assert hedged.backup_result is None


def test_on_backup_dispatch_fires_before_backup_send() -> None:
    """The ``on_backup_dispatch`` callback must run *before* the backup
    HTTP send, so capacity tracking sees the slot occupied during the
    backup's lifetime — not only after it returns."""

    callback_ts: list[float] = []
    backup_send_ts: list[float] = []

    def fake_send(
        provider, prompt, max_tokens, timeout, ttft_event, ttft_info
    ):
        if provider == "primary":
            if ttft_event is not None:
                ttft_event.set()
            if ttft_info is not None:
                ttft_info["status"] = "HTTP 500"
            return _http_error_send(provider)
        # backup — record the moment send actually starts
        backup_send_ts.append(time.time())
        if ttft_info is not None:
            ttft_info.update(
                ttft_ms=200.0, first_token_ts=time.time(), status="success"
            )
        if ttft_event is not None:
            ttft_event.set()
        return _success_send(provider)

    def on_dispatch(ts: float) -> None:
        callback_ts.append(ts)

    send_hedged_request(
        send_fn=fake_send,
        primary_provider="primary",
        backup_provider="backup",
        hedge_delay_sec=0.2,
        prompt="x",
        max_tokens=8,
        timeout=5,
        dispatch_overhead_sec=0.0,
        on_backup_dispatch=on_dispatch,
    )
    assert len(callback_ts) == 1
    assert len(backup_send_ts) == 1
    # Callback fires no later than the backup HTTP send starts.
    assert callback_ts[0] <= backup_send_ts[0] + 0.05  # generous slack


def test_on_backup_dispatch_not_called_when_primary_succeeds() -> None:
    """No backup → no callback."""
    calls: list[float] = []

    def fake_send(
        provider, prompt, max_tokens, timeout, ttft_event, ttft_info
    ):
        if ttft_info is not None:
            ttft_info.update(
                ttft_ms=20.0, first_token_ts=time.time(), status="success"
            )
        if ttft_event is not None:
            ttft_event.set()
        return _success_send(provider, ttft_ms=20.0)

    send_hedged_request(
        send_fn=fake_send,
        primary_provider="primary",
        backup_provider="backup",
        hedge_delay_sec=1.0,
        prompt="x",
        max_tokens=8,
        timeout=5,
        on_backup_dispatch=lambda ts: calls.append(ts),
    )
    assert calls == []


def test_callback_exception_does_not_abort_hedge() -> None:
    """A buggy callback must not prevent the backup from firing."""

    def fake_send(
        provider, prompt, max_tokens, timeout, ttft_event, ttft_info
    ):
        if provider == "primary":
            if ttft_event is not None:
                ttft_event.set()
            if ttft_info is not None:
                ttft_info["status"] = "HTTP 500"
            return _http_error_send(provider)
        if ttft_info is not None:
            ttft_info.update(
                ttft_ms=100.0, first_token_ts=time.time(), status="success"
            )
        if ttft_event is not None:
            ttft_event.set()
        return _success_send(provider)

    def bad_callback(ts: float) -> None:
        raise RuntimeError("callback intentionally broken")

    hedged = send_hedged_request(
        send_fn=fake_send,
        primary_provider="primary",
        backup_provider="backup",
        hedge_delay_sec=0.2,
        prompt="x",
        max_tokens=8,
        timeout=5,
        dispatch_overhead_sec=0.0,
        on_backup_dispatch=bad_callback,
    )
    assert hedged.hedge_triggered is True
    assert hedged.winner == "backup"
