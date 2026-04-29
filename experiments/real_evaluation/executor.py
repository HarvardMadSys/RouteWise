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
from typing import Any, Callable, Protocol

from experiments.real_evaluation.transports import SingleRequestResult

logger = logging.getLogger(__name__)


def logger_log_dispatch_callback_error() -> None:
    """Log + swallow ``on_backup_dispatch`` callback exceptions.

    A misbehaving callback must not abort the hedge — the backup thread
    still needs to fire so the user gets a response.
    """
    logger.warning("on_backup_dispatch callback raised", exc_info=True)


class SendFn(Protocol):
    """Callable signature accepted by ``send_hedged_request``.

    A wrapper around one transport's ``send`` that adds the provider
    selection. Typical implementations look up the right
    :class:`BaseTransport` for ``provider`` and call its ``send`` method.
    """

    def __call__(
        self,
        provider: str,
        prompt: str,
        max_tokens: int,
        timeout: int,
        ttft_event: threading.Event | None,
        ttft_info: dict[str, Any] | None,
    ) -> SingleRequestResult: ...


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
    )


def send_hedged_request(
    send_fn: SendFn,
    primary_provider: str,
    backup_provider: str,
    hedge_delay_sec: float,
    prompt: str,
    max_tokens: int,
    timeout: int = 60,
    dispatch_overhead_sec: float = 0.05,
    on_backup_dispatch: Callable[[float], None] | None = None,
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
      4. Wait for both threads to finish.
      5. Pick the winner by ``first_token_ts``.

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

    def run_primary() -> None:
        primary_holder["r"] = send_fn(
            provider=primary_provider,
            prompt=prompt,
            max_tokens=max_tokens,
            timeout=timeout,
            ttft_event=primary_ttft,
            ttft_info=primary_ttft_info,
        )

    def run_backup() -> None:
        backup_holder["r"] = send_fn(
            provider=backup_provider,
            prompt=prompt,
            max_tokens=max_tokens,
            timeout=timeout,
            ttft_event=backup_ttft,
            ttft_info=backup_ttft_info,
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
    if primary_failed_early:
        hedge_triggered = True
        if dispatch_overhead_sec > 0:
            time.sleep(dispatch_overhead_sec)
        backup_dispatch_ts = time.time()
        if on_backup_dispatch is not None:
            try:
                on_backup_dispatch(backup_dispatch_ts)
            except Exception:
                logger_log_dispatch_callback_error()
        backup_thread = threading.Thread(target=run_backup, daemon=True)
        backup_thread.start()
        primary_thread.join(timeout=timeout + 5)
        backup_thread.join(timeout=timeout + 5)
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
        primary_ttft_info=dict(primary_ttft_info),
        backup_ttft_info=dict(backup_ttft_info),
    )


__all__ = [
    "HedgedResult",
    "SendFn",
    "send_hedged_request",
    "send_request",
]
