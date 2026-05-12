"""Hedging-executor regressions.

Two behaviours pinned here:

1. Phase5 trigger semantics — backup must fire when primary returns an
   HTTP error before any visible token, even though the transport sets
   ``ttft_event`` in its ``finally`` block.

2. ``on_backup_dispatch`` callback — must run *before* the backup
   thread starts, so the runner can charge concurrency / quota at the
   moment the backup actually goes on the wire (not after the hedged
   request returns).

3. Cancel-aware racing — once a winner emerges (first visible token),
   the loser's ``cancel_event`` must be set so its transport can close
   the SSE stream and stop accumulating per-token billing.
"""

from __future__ import annotations

import time

import pytest

from experiments.real_evaluation.executor import (
    send_checkpoint_hedged_request,
    send_hedged_request,
)
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
        provider, prompt, max_tokens, timeout, ttft_event, ttft_info, cancel_event=None
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
    )
    assert hedged.hedge_triggered is True
    assert hedged.winner == "backup"
    assert hedged.backup_result is not None
    assert hedged.backup_result.status == "success"


def test_primary_succeeds_quickly_no_hedge() -> None:
    """When primary returns a valid TTFT before the deadline, backup
    must NOT be dispatched."""

    def fake_send(
        provider, prompt, max_tokens, timeout, ttft_event, ttft_info, cancel_event=None
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
        provider, prompt, max_tokens, timeout, ttft_event, ttft_info, cancel_event=None
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

    hedged = send_hedged_request(
        send_fn=fake_send,
        primary_provider="primary",
        backup_provider="backup",
        hedge_delay_sec=0.2,
        prompt="x",
        max_tokens=8,
        timeout=5,
        on_backup_dispatch=on_dispatch,
    )
    assert len(callback_ts) == 1
    assert len(backup_send_ts) == 1
    assert hedged.backup_dispatch_ts == pytest.approx(callback_ts[0])
    # Callback fires no later than the backup HTTP send starts.
    assert callback_ts[0] <= backup_send_ts[0] + 0.05  # generous slack


def test_checkpoint_hedge_dispatches_at_callback_selected_checkpoint() -> None:
    checkpoint_calls: list[tuple[float, float]] = []
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

    def checkpoint_fn(elapsed_sec: float, checkpoint_ts: float) -> str | None:
        checkpoint_calls.append((elapsed_sec, checkpoint_ts))
        return "backup" if elapsed_sec >= 0.01 else None

    hedged = send_checkpoint_hedged_request(
        send_fn=fake_send,
        primary_provider="primary",
        hedge_checkpoints_sec=(0.01, 0.02),
        checkpoint_fn=checkpoint_fn,
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


def test_on_backup_dispatch_not_called_when_primary_succeeds() -> None:
    """No backup → no callback."""
    calls: list[float] = []

    def fake_send(
        provider, prompt, max_tokens, timeout, ttft_event, ttft_info, cancel_event=None
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
        provider, prompt, max_tokens, timeout, ttft_event, ttft_info, cancel_event=None
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
        on_backup_dispatch=bad_callback,
    )
    assert hedged.hedge_triggered is True
    assert hedged.winner == "backup"


def _hedge_race_send_factory(
    primary_first_token_delay: float,
    backup_first_token_delay: float,
    seen_cancel: dict[str, bool],
) -> callable:
    """Build a fake send_fn that simulates two providers racing.

    Each side sleeps until its first-token delay, then signals the
    ``ttft_event``. After signalling it polls ``cancel_event`` for up to
    300ms; if the cancel fires we record it via ``seen_cancel``.
    """

    def fake_send(
        provider, prompt, max_tokens, timeout, ttft_event, ttft_info, cancel_event=None
    ):
        delay = (
            primary_first_token_delay
            if provider == "primary"
            else backup_first_token_delay
        )
        time.sleep(delay)
        first_token_ts = time.time()  # capture AT first-token, like the real transport
        if ttft_info is not None:
            ttft_info.update(
                ttft_ms=delay * 1000.0,
                first_token_ts=first_token_ts,
                status="success",
            )
        if ttft_event is not None:
            ttft_event.set()

        # Now linger like a real streaming response. Watch for cancel.
        deadline = time.time() + 0.3
        while time.time() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                seen_cancel[provider] = True
                break
            time.sleep(0.005)
        return SingleRequestResult(
            ttft_ms=delay * 1000.0,
            e2e_ms=(time.time() - first_token_ts + delay) * 1000.0,
            status="success",
            provider=provider,
            prompt_tokens=10,
            completion_tokens=20,
            first_token_ts=first_token_ts,
            start_ts=first_token_ts - delay,
        )

    return fake_send


def test_cancel_loser_after_first_token_primary_wins() -> None:
    """Primary returns first token first → backup gets cancel signal."""
    seen: dict[str, bool] = {}
    fake = _hedge_race_send_factory(
        primary_first_token_delay=0.05,  # primary fast
        backup_first_token_delay=0.30,   # backup slow
        seen_cancel=seen,
    )
    hedged = send_hedged_request(
        send_fn=fake,
        primary_provider="primary",
        backup_provider="backup",
        hedge_delay_sec=0.0,  # force backup to fire immediately
        prompt="x",
        max_tokens=8,
        timeout=5,
        cancel_loser_after_first_token=True,
    )
    assert hedged.hedge_triggered is True
    assert hedged.winner == "primary"
    assert seen.get("backup") is True
    assert seen.get("primary") is None  # winner not canceled


def test_cancel_loser_after_first_token_backup_wins() -> None:
    """Backup returns first token first → primary gets cancel signal."""
    seen: dict[str, bool] = {}
    fake = _hedge_race_send_factory(
        primary_first_token_delay=0.30,  # primary slow
        backup_first_token_delay=0.05,   # backup fast
        seen_cancel=seen,
    )
    hedged = send_hedged_request(
        send_fn=fake,
        primary_provider="primary",
        backup_provider="backup",
        hedge_delay_sec=0.0,
        prompt="x",
        max_tokens=8,
        timeout=5,
        cancel_loser_after_first_token=True,
    )
    assert hedged.hedge_triggered is True
    assert hedged.winner == "backup"
    assert seen.get("primary") is True
    assert seen.get("backup") is None


def test_cancel_loser_disabled_no_cancel_event_set() -> None:
    """When the cancel knob is False, neither side gets canceled."""
    seen: dict[str, bool] = {}
    fake = _hedge_race_send_factory(
        primary_first_token_delay=0.05,
        backup_first_token_delay=0.30,
        seen_cancel=seen,
    )
    hedged = send_hedged_request(
        send_fn=fake,
        primary_provider="primary",
        backup_provider="backup",
        hedge_delay_sec=0.0,
        prompt="x",
        max_tokens=8,
        timeout=5,
        cancel_loser_after_first_token=False,
    )
    assert hedged.hedge_triggered is True
    assert hedged.winner == "primary"
    assert seen == {}  # nobody saw a cancel
