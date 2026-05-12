"""Hedged request execution.

Standalone hedging-race logic, decoupled from any specific runner. Takes
a ``send_fn`` callable and dispatches primary / backup threads, returning
both results plus winner.

Migrated from ``NSDI2027_RouteWise/experiment/scripts/phase5_online_evaluation.py``
lines 903-1007 (``_send_hedged_request``). The phase5 trigger semantics are
**preserved deliberately** (and differ from phase6's later version):

    Backup is dispatched whenever the primary fails to produce a *valid*
    TTFT before ``hedge_delay_sec``. Specifically:

        if (not primary_ttft_event.is_set()) or primary_ttft_info["ttft_ms"] <= 0:
            launch backup

    Phase6's version (``_send_hedged`` in phase6_joint) only checked
    ``primary_ttft_event.is_set()``. That's a bug: when the primary returns
    an HTTP error, the transport sets the event in its ``finally`` block
    *without* populating ``ttft_ms``, so phase6 silently skips the backup.
    The phase5 check catches this case.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from experiments.real_evaluation.transports import SingleRequestResult

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def logger_log_dispatch_callback_error() -> None:
    """Log + swallow ``on_backup_dispatch`` callback exceptions.

    A misbehaving callback must not abort the hedge — the backup thread
    still needs to fire so the user gets a response.
    """
    logger.warning("on_backup_dispatch callback raised", exc_info=True)


def _ttft_succeeded(
    ttft_event: threading.Event, ttft_info: dict[str, Any]
) -> bool:
    """Did this side reach a successful first token?

    The transport sets ``ttft_event`` in two situations: a successful first
    visible token arrived (``ttft_info["ttft_ms"] > 0``), and the SSE
    finished/errored without ever producing one (``ttft_ms`` stays at -1).
    Only the former should win the race; the latter must let the other side
    keep streaming.
    """
    if not ttft_event.is_set():
        return False
    return ttft_info.get("ttft_ms", -1.0) > 0


def _race_monitor_loop(
    primary_ttft: threading.Event,
    primary_ttft_info: dict[str, Any],
    backup_ttft: threading.Event,
    backup_ttft_info: dict[str, Any],
    primary_thread: threading.Thread,
    backup_thread: threading.Thread,
    primary_cancel: threading.Event,
    backup_cancel: threading.Event,
    deadline_sec: float,
    poll_sec: float,
) -> None:
    """Cancel the hedge loser once the winner produces a visible token.

    Polls both ``ttft_info`` slots; whichever reports a successful first
    token first wins, and we set the *other* side's ``cancel_event``. If
    both reach first token in the same poll interval we cancel neither
    (effectively a tie — both will return shortly anyway). Exits early when
    both transport threads have died (both errored / both finished cleanly).
    """
    deadline = time.time() + deadline_sec
    while time.time() < deadline:
        p_won = _ttft_succeeded(primary_ttft, primary_ttft_info)
        b_won = _ttft_succeeded(backup_ttft, backup_ttft_info)
        if p_won and not b_won:
            backup_cancel.set()
            return
        if b_won and not p_won:
            primary_cancel.set()
            return
        if p_won and b_won:
            # Photo finish — both produced a visible token within one poll
            # interval. No useful cancellation; let both stream out.
            return
        if not primary_thread.is_alive() and not backup_thread.is_alive():
            return
        time.sleep(poll_sec)


class SendFn(Protocol):
    """Callable signature accepted by ``send_hedged_request``.

    A wrapper around one transport's ``send`` that adds the provider
    selection. Typical implementations look up the right
    :class:`BaseTransport` for ``provider`` and call its ``send`` method.

    ``cancel_event`` is propagated to the transport so the SSE loop can
    abort the loser of a hedge race after the winner returns its first
    visible token.
    """

    def __call__(
        self,
        provider: str,
        prompt: str,
        max_tokens: int,
        timeout: int,
        ttft_event: threading.Event | None,
        ttft_info: dict[str, Any] | None,
        cancel_event: threading.Event | None = None,
    ) -> SingleRequestResult: ...


class CheckpointHedgeFn(Protocol):
    """Callable used to decide whether to dispatch a backup at a checkpoint."""

    def __call__(self, elapsed_sec: float, checkpoint_ts: float) -> str | None: ...


@dataclass
class HedgedResult:
    """Outcome of a hedged dispatch.

    ``winner`` is ``"primary"`` if the primary's first token arrived first
    (or no hedge was triggered); ``"backup"`` if the backup's first token
    arrived first; ``"primary"`` as a fallback when neither produced a
    visible token.
    """

    primary_result: SingleRequestResult
    backup_result: SingleRequestResult | None
    winner: str  # "primary" | "backup"
    hedge_triggered: bool
    backup_provider: str | None = None
    hedge_delay_sec: float | None = None
    hedge_checkpoint_ts: float | None = None
    backup_dispatch_ts: float | None = None
    primary_ttft_info: dict[str, Any] = field(default_factory=dict)
    backup_ttft_info: dict[str, Any] = field(default_factory=dict)

    @property
    def chosen_result(self) -> SingleRequestResult:
        """Return the result that won the race (the user-visible response)."""
        if self.winner == "backup" and self.backup_result is not None:
            return self.backup_result
        return self.primary_result


def send_request(
    send_fn: SendFn,
    provider: str,
    prompt: str,
    max_tokens: int,
    timeout: int = 60,
) -> SingleRequestResult:
    """Send a single request via ``send_fn`` (no hedging)."""
    return send_fn(
        provider=provider,
        prompt=prompt,
        max_tokens=max_tokens,
        timeout=timeout,
        ttft_event=None,
        ttft_info=None,
        cancel_event=None,
    )


def send_hedged_request(
    send_fn: SendFn,
    primary_provider: str,
    backup_provider: str,
    hedge_delay_sec: float,
    prompt: str,
    max_tokens: int,
    timeout: int = 60,
    on_backup_dispatch: Callable[[float], None] | None = None,
    cancel_loser_after_first_token: bool = True,
    race_monitor_poll_sec: float = 0.005,
) -> HedgedResult:
    """Race a primary against a delayed backup.

    Sequence:
      1. Dispatch primary in a thread; pass it a ``ttft_event`` and
         ``ttft_info`` so it can signal first-token arrival.
      2. Block on ``primary_ttft.wait(timeout=hedge_delay_sec)``.
      3. If the event did not fire OR ``ttft_info["ttft_ms"] <= 0``,
         invoke ``on_backup_dispatch(dispatch_ts)`` *before* the backup
         thread starts (so capacity tracking sees the slot occupied
         during the backup's lifetime, not only after it returns), then
         dispatch the backup. (See module docstring for why the second
         clause matters — this is the phase5 fix to phase6's bug.)
      4. If ``cancel_loser_after_first_token`` (default True), spawn a
         race-monitor thread that watches both ``ttft_info`` slots and
         sets the loser's ``cancel_event`` once the winner produces a
         visible token. The transport's SSE loop polls this event and
         closes the response, so the loser stops streaming early and
         we avoid paying the rest of its output-token cost on
         providers that bill per streamed token (OpenRouter et al.).
         Subscription / quota providers may still consume capacity
         locally even after a cancel; this knob only controls the
         client-side stream close.
      5. Wait for both threads to finish.
      6. Pick the winner by ``first_token_ts``.

    ``hedge_delay_sec == math.inf`` disables hedging (primary-only).

    ``on_backup_dispatch`` receives the unix timestamp at which the
    backup thread is about to start. Use it to charge concurrency /
    quota at the *actual* dispatch instant; without this the runner
    can only charge once the hedged request returns, which leaves
    a race window where in-flight backups are invisible to other
    routing decisions.
    """
    if not math.isfinite(hedge_delay_sec):
        primary_result = send_request(
            send_fn,
            provider=primary_provider,
            prompt=prompt,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return HedgedResult(
            primary_result=primary_result,
            backup_result=None,
            winner="primary",
            hedge_triggered=False,
        )

    hedge_delay_sec = max(0.0, float(hedge_delay_sec))

    primary_holder: dict[str, SingleRequestResult] = {}
    backup_holder: dict[str, SingleRequestResult | None] = {"r": None}
    primary_ttft = threading.Event()
    backup_ttft = threading.Event()
    primary_ttft_info: dict[str, Any] = {}
    backup_ttft_info: dict[str, Any] = {}
    primary_cancel = threading.Event()
    backup_cancel = threading.Event()

    def run_primary() -> None:
        primary_holder["r"] = send_fn(
            provider=primary_provider,
            prompt=prompt,
            max_tokens=max_tokens,
            timeout=timeout,
            ttft_event=primary_ttft,
            ttft_info=primary_ttft_info,
            cancel_event=primary_cancel,
        )

    def run_backup() -> None:
        backup_holder["r"] = send_fn(
            provider=backup_provider,
            prompt=prompt,
            max_tokens=max_tokens,
            timeout=timeout,
            ttft_event=backup_ttft,
            ttft_info=backup_ttft_info,
            cancel_event=backup_cancel,
        )

    primary_thread = threading.Thread(target=run_primary, daemon=True)
    primary_thread.start()

    primary_ttft.wait(timeout=hedge_delay_sec)

    # Phase5 trigger semantics (NOT phase6): also dispatch backup when the
    # primary's event was set due to an early HTTP / transport error before
    # any visible token arrived. ``ttft_info["ttft_ms"]`` stays at -1 in
    # that case, so the second clause catches it.
    primary_failed_early = (
        not primary_ttft.is_set()
        or primary_ttft_info.get("ttft_ms", -1.0) <= 0
    )

    hedge_triggered = False
    monitor_thread: threading.Thread | None = None
    backup_dispatch_ts: float | None = None
    if primary_failed_early:
        hedge_triggered = True
        backup_dispatch_ts = time.time()
        if on_backup_dispatch is not None:
            try:
                on_backup_dispatch(backup_dispatch_ts)
            except Exception:
                logger_log_dispatch_callback_error()
        backup_thread = threading.Thread(target=run_backup, daemon=True)
        backup_thread.start()

        if cancel_loser_after_first_token:
            monitor_thread = threading.Thread(
                target=_race_monitor_loop,
                args=(
                    primary_ttft,
                    primary_ttft_info,
                    backup_ttft,
                    backup_ttft_info,
                    primary_thread,
                    backup_thread,
                    primary_cancel,
                    backup_cancel,
                    timeout + 5,
                    race_monitor_poll_sec,
                ),
                name="hedge-race-monitor",
                daemon=True,
            )
            monitor_thread.start()

        primary_thread.join(timeout=timeout + 5)
        backup_thread.join(timeout=timeout + 5)
        if monitor_thread is not None:
            # Monitor exits as soon as a winner emerges or both threads die,
            # so this join is essentially instant.
            monitor_thread.join(timeout=1.0)
    else:
        primary_thread.join(timeout=timeout + 5)

    primary_result = primary_holder.get("r") or SingleRequestResult(
        ttft_ms=-1.0,
        e2e_ms=-1.0,
        status="error",
        provider=primary_provider,
        error_message="primary_thread_missing_result",
    )
    backup_result = backup_holder.get("r")

    def first_token_or_inf(r: SingleRequestResult | None) -> float:
        if r is None or r.status != "success" or r.first_token_ts is None:
            return float("inf")
        return float(r.first_token_ts)

    winner = "primary"
    if hedge_triggered:
        primary_first = first_token_or_inf(primary_result)
        backup_first = first_token_or_inf(backup_result)
        if backup_first < primary_first:
            winner = "backup"

    return HedgedResult(
        primary_result=primary_result,
        backup_result=backup_result,
        winner=winner,
        hedge_triggered=hedge_triggered,
        backup_provider=backup_provider if hedge_triggered else None,
        hedge_delay_sec=hedge_delay_sec if hedge_triggered else None,
        hedge_checkpoint_ts=backup_dispatch_ts,
        backup_dispatch_ts=backup_dispatch_ts,
        primary_ttft_info=dict(primary_ttft_info),
        backup_ttft_info=dict(backup_ttft_info),
    )


def send_checkpoint_hedged_request(
    send_fn: SendFn,
    primary_provider: str,
    hedge_checkpoints_sec: tuple[float, ...],
    checkpoint_fn: CheckpointHedgeFn,
    prompt: str,
    max_tokens: int,
    timeout: int = 60,
    cancel_loser_after_first_token: bool = True,
    race_monitor_poll_sec: float = 0.005,
) -> HedgedResult:
    """Dispatch a primary and evaluate hedge decisions at SLO checkpoints.

    Unlike ``send_hedged_request``, this does not receive a pre-selected backup
    or a fixed delay. The caller supplies the RouteWise checkpoint schedule and
    a callback that re-evaluates the probability target using current state.
    """
    checkpoints = tuple(
        sorted(
            float(checkpoint)
            for checkpoint in hedge_checkpoints_sec
            if math.isfinite(checkpoint) and checkpoint >= 0.0
        )
    )
    if not checkpoints:
        primary_result = send_request(
            send_fn,
            provider=primary_provider,
            prompt=prompt,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return HedgedResult(
            primary_result=primary_result,
            backup_result=None,
            winner="primary",
            hedge_triggered=False,
        )

    primary_holder: dict[str, SingleRequestResult] = {}
    backup_holder: dict[str, SingleRequestResult | None] = {"r": None}
    primary_ttft = threading.Event()
    backup_ttft = threading.Event()
    primary_ttft_info: dict[str, Any] = {}
    backup_ttft_info: dict[str, Any] = {}
    primary_cancel = threading.Event()
    backup_cancel = threading.Event()

    def run_primary() -> None:
        primary_holder["r"] = send_fn(
            provider=primary_provider,
            prompt=prompt,
            max_tokens=max_tokens,
            timeout=timeout,
            ttft_event=primary_ttft,
            ttft_info=primary_ttft_info,
            cancel_event=primary_cancel,
        )

    def run_backup(provider: str) -> None:
        backup_holder["r"] = send_fn(
            provider=provider,
            prompt=prompt,
            max_tokens=max_tokens,
            timeout=timeout,
            ttft_event=backup_ttft,
            ttft_info=backup_ttft_info,
            cancel_event=backup_cancel,
        )

    primary_thread = threading.Thread(target=run_primary, daemon=True)
    schedule_start_ts = time.time()
    primary_thread.start()

    hedge_triggered = False
    hedge_delay_sec: float | None = None
    hedge_checkpoint_ts: float | None = None
    backup_dispatch_ts: float | None = None
    backup_provider: str | None = None
    backup_thread: threading.Thread | None = None
    monitor_thread: threading.Thread | None = None

    for checkpoint_sec in checkpoints:
        wait_until = schedule_start_ts + checkpoint_sec
        wait_remaining = max(0.0, wait_until - time.time())
        if _ttft_succeeded(primary_ttft, primary_ttft_info):
            break
        if wait_remaining > 0.0:
            if primary_ttft.is_set():
                time.sleep(wait_remaining)
            else:
                primary_ttft.wait(timeout=wait_remaining)
        if _ttft_succeeded(primary_ttft, primary_ttft_info):
            break

        checkpoint_ts = time.time()
        try:
            selected_backup = checkpoint_fn(checkpoint_sec, checkpoint_ts)
        except Exception:
            logger.warning("checkpoint hedge callback raised", exc_info=True)
            selected_backup = None
        if selected_backup is None:
            continue

        hedge_triggered = True
        hedge_delay_sec = checkpoint_sec
        hedge_checkpoint_ts = checkpoint_ts
        backup_provider = selected_backup
        backup_dispatch_ts = time.time()
        backup_thread = threading.Thread(
            target=run_backup,
            args=(selected_backup,),
            daemon=True,
        )
        backup_thread.start()

        if cancel_loser_after_first_token:
            monitor_thread = threading.Thread(
                target=_race_monitor_loop,
                args=(
                    primary_ttft,
                    primary_ttft_info,
                    backup_ttft,
                    backup_ttft_info,
                    primary_thread,
                    backup_thread,
                    primary_cancel,
                    backup_cancel,
                    timeout + 5,
                    race_monitor_poll_sec,
                ),
                name="hedge-race-monitor",
                daemon=True,
            )
            monitor_thread.start()
        break

    primary_thread.join(timeout=timeout + 5)
    if backup_thread is not None:
        backup_thread.join(timeout=timeout + 5)
    if monitor_thread is not None:
        monitor_thread.join(timeout=1.0)

    primary_result = primary_holder.get("r") or SingleRequestResult(
        ttft_ms=-1.0,
        e2e_ms=-1.0,
        status="error",
        provider=primary_provider,
        error_message="primary_thread_missing_result",
    )
    backup_result = backup_holder.get("r")

    def first_token_or_inf(r: SingleRequestResult | None) -> float:
        if r is None or r.status != "success" or r.first_token_ts is None:
            return float("inf")
        return float(r.first_token_ts)

    winner = "primary"
    if hedge_triggered:
        primary_first = first_token_or_inf(primary_result)
        backup_first = first_token_or_inf(backup_result)
        if backup_first < primary_first:
            winner = "backup"

    return HedgedResult(
        primary_result=primary_result,
        backup_result=backup_result,
        winner=winner,
        hedge_triggered=hedge_triggered,
        backup_provider=backup_provider,
        hedge_delay_sec=hedge_delay_sec,
        hedge_checkpoint_ts=hedge_checkpoint_ts,
        backup_dispatch_ts=backup_dispatch_ts,
        primary_ttft_info=dict(primary_ttft_info),
        backup_ttft_info=dict(backup_ttft_info),
    )


__all__ = [
    "CheckpointHedgeFn",
    "HedgedResult",
    "SendFn",
    "send_checkpoint_hedged_request",
    "send_hedged_request",
    "send_request",
]
