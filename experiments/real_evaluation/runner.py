"""Trace-replay runner for real online evaluation.

Drives one or more policies through a synthetic / real trace against live
provider APIs. Each request is dispatched in a daemon thread; per-policy
profiles + capacity state are updated by feedback after each completion.

Migrated from
``NSDI2027_RouteWise/experiment/scripts/phase6_joint_online_evaluation.py``
lines 1046-1600. Differences:

- Per-policy CSV via :class:`Recorder` (not the old inline writer)
- Hedge dispatch uses :func:`executor.send_hedged_request` (with the
  phase5 trigger-semantics fix)
- Sentinel handling for OpenRouter native modes is centralized here
- ``--max-cost-usd`` guardrail per policy and globally
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from experiments.real_evaluation.executor import (
    HedgedResult,
    send_hedged_request,
    send_request,
)
from experiments.real_evaluation.inventory import (
    InventoryConfig,
    ProviderSpec,
    load_inventory,
)
from experiments.real_evaluation.policies import (
    HEDGE_DISPATCH_OVERHEAD_SEC,
    OR_AUTO_SENTINEL,
    OR_SORT_SENTINEL_TO_MODE,
    BasePolicy,
    RequestContext,
    RoutingDecision,
    build_policy,
    compute_hedge_time_sec,
    select_safe_cheapest_backup,
)
from experiments.real_evaluation.recorder import Recorder
from experiments.real_evaluation.transports import (
    BaseTransport,
    SingleRequestResult,
    TransportConfig,
    build_transport,
)

DEFAULT_TIMEOUT_SEC: int = 60
DEFAULT_PROBE_PROMPT: str = "Write a one-sentence greeting."
DEFAULT_MAX_TOKENS: int = 128

logger = logging.getLogger(__name__)


@dataclass
class TraceRequest:
    """One row from a trace JSONL."""

    arrival_time_sec: float
    prompt: str
    prompt_tokens: int
    max_tokens: int
    use_real_prompt: bool = True


def load_trace_jsonl(
    path: Path,
    *,
    max_requests: int | None = None,
    time_compression: float = 1.0,
    trace_start_sec: float = 0.0,
    trace_end_sec: float = float("inf"),
) -> list[TraceRequest]:
    """Load an arrival-paced trace from a JSONL file.

    Recognized fields (first non-null wins):
        arrival_time_sec : ``arrived_at``
        prompt           : ``prompt_text`` | ``prompt``
        prompt_tokens    : ``num_prefill_tokens`` | ``prompt_tokens``
        max_tokens       : ``num_decode_tokens`` | ``max_tokens``

    ``time_compression`` divides arrival times so a long trace can be
    replayed in less wall-clock time.
    """
    out: list[TraceRequest] = []
    first_ts: float | None = None
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            arrived = float(
                rec.get("arrived_at")
                or rec.get("arrival_time_sec")
                or 0.0
            )
            if first_ts is None:
                first_ts = arrived
            relative = (arrived - first_ts) / max(time_compression, 1e-6)
            if relative < trace_start_sec:
                continue
            if relative > trace_end_sec:
                break
            prompt = rec.get("prompt_text") or rec.get("prompt") or DEFAULT_PROBE_PROMPT
            prompt_tokens = int(
                rec.get("num_prefill_tokens") or rec.get("prompt_tokens") or 0
            )
            max_tokens = int(
                rec.get("num_decode_tokens")
                or rec.get("max_tokens")
                or DEFAULT_MAX_TOKENS
            )
            out.append(
                TraceRequest(
                    arrival_time_sec=relative,
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                    max_tokens=max_tokens,
                )
            )
            if max_requests is not None and len(out) >= max_requests:
                break
    return out


def make_synthetic_trace(
    n_requests: int,
    rate_per_sec: float = 1.0,
    prompt: str = DEFAULT_PROBE_PROMPT,
    prompt_tokens: int = 50,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[TraceRequest]:
    """Build a fixed-rate synthetic trace for smoke testing."""
    interval = 1.0 / max(rate_per_sec, 1e-6)
    return [
        TraceRequest(
            arrival_time_sec=i * interval,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            use_real_prompt=False,
        )
        for i in range(n_requests)
    ]


# ---------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------


class RealExperimentRunner:
    """Owns transports, policies, recorder, and the trace-replay loop.

    Threading model: trace dispatch happens on the main thread (sleep until
    arrival, then spawn a daemon thread). All HTTP calls happen in those
    daemon threads. The recorder + per-policy state hold their own locks.
    """

    def __init__(
        self,
        inventory: InventoryConfig,
        policy_names: list[str],
        recorder: Recorder,
        *,
        slo_ms: float | None = None,
        max_cost_usd: float = 5.0,
        timeout_sec: int = DEFAULT_TIMEOUT_SEC,
        profile_window_sec: float = 15 * 60.0,
    ) -> None:
        self.inventory = inventory
        self.slo_ms = slo_ms if slo_ms is not None else inventory.primary_slo_ms
        self.slo_sec = self.slo_ms / 1000.0
        self.max_cost_usd = max_cost_usd
        self.timeout_sec = timeout_sec
        self.recorder = recorder

        self.policies: dict[str, BasePolicy] = {
            name: build_policy(
                name,
                specs=inventory.providers,
                slo_ms=self.slo_ms,
                profile_window_sec=profile_window_sec,
            )
            for name in policy_names
        }

        self._spec_by_name: dict[str, ProviderSpec] = {
            spec.name: spec for spec in inventory.providers
        }
        self._transports: dict[str, BaseTransport] = self._build_transports()
        self._or_base_cfg: TransportConfig | None = self._first_openrouter_cfg()

        self._cost_lock = threading.Lock()
        self._cost_per_policy: dict[str, float] = {n: 0.0 for n in policy_names}
        self._total_cost_usd: float = 0.0
        self._stop_event = threading.Event()
        # ``threading.local`` must live on the runner instance so each
        # worker thread sees a stable storage; a fresh ``threading.local()``
        # in ``_session()`` would defeat the per-thread reuse contract.
        self._thread_local = threading.local()

    # ------------------------------------------------------------------
    # Transport plumbing.
    # ------------------------------------------------------------------

    def _session(self) -> requests.Session:
        """Return one ``requests.Session`` per worker thread.

        Persists across calls thanks to the instance-level ``_thread_local``;
        the previous implementation declared ``threading.local()`` inside
        the function, which created a fresh storage on every call and
        therefore never reused a session.
        """
        sess = getattr(self._thread_local, "session", None)
        if sess is None:
            sess = requests.Session()
            self._thread_local.session = sess
        return sess

    def _build_transports(self) -> dict[str, BaseTransport]:
        out: dict[str, BaseTransport] = {}
        session = requests.Session()
        for spec in self.inventory.providers:
            out[spec.name] = build_transport(spec.transport_cfg, session)
        return out

    def _first_openrouter_cfg(self) -> TransportConfig | None:
        for spec in self.inventory.providers:
            if spec.transport_cfg.transport == "openrouter":
                return spec.transport_cfg
        return None

    def _send_via_transport(
        self,
        provider: str,
        prompt: str,
        max_tokens: int,
        timeout: int,
        ttft_event: threading.Event | None,
        ttft_info: dict[str, Any] | None,
    ) -> SingleRequestResult:
        """Dispatch one request to a real or sentinel provider."""
        if provider == OR_AUTO_SENTINEL or provider in OR_SORT_SENTINEL_TO_MODE:
            return self._send_or_sentinel(
                provider, prompt, max_tokens, timeout, ttft_event, ttft_info
            )

        transport = self._transports.get(provider)
        if transport is None:
            return SingleRequestResult(
                ttft_ms=-1.0,
                e2e_ms=-1.0,
                status="error",
                provider=provider,
                error_message=f"unknown_provider:{provider}",
            )
        transport.session = self._session()
        return transport.send(
            prompt=prompt,
            max_tokens=max_tokens,
            timeout=timeout,
            ttft_event=ttft_event,
            ttft_info=ttft_info,
        )

    def _send_or_sentinel(
        self,
        sentinel: str,
        prompt: str,
        max_tokens: int,
        timeout: int,
        ttft_event: threading.Event | None,
        ttft_info: dict[str, Any] | None,
    ) -> SingleRequestResult:
        """Dispatch a sentinel via OpenRouter (auto or any sort= mode)."""
        base = self._or_base_cfg
        if base is None:
            return SingleRequestResult(
                ttft_ms=-1.0,
                e2e_ms=-1.0,
                status="error",
                provider=sentinel,
                error_message="no_openrouter_in_inventory",
            )
        sort_mode = OR_SORT_SENTINEL_TO_MODE.get(sentinel)
        cfg = dataclasses.replace(
            base,
            name=sentinel,
            provider_hint=None,
            sort_mode=sort_mode,
            extra_headers=dict(base.extra_headers),
        )
        from experiments.real_evaluation.transports import OpenAICompatStreamingTransport

        transport = OpenAICompatStreamingTransport(cfg, self._session())
        return transport.send(
            prompt=prompt,
            max_tokens=max_tokens,
            timeout=timeout,
            ttft_event=ttft_event,
            ttft_info=ttft_info,
        )

    # ------------------------------------------------------------------
    # Warmup + replay.
    # ------------------------------------------------------------------

    def warmup(self, probes_per_provider: int = 3, sleep_sec: float = 0.5) -> None:
        """Send a few probes to each provider; broadcast samples to all policies."""
        for spec in self.inventory.providers:
            for i in range(probes_per_provider):
                if self._stop_event.is_set():
                    return
                ts = time.time()
                result = self._send_via_transport(
                    provider=spec.name,
                    prompt=DEFAULT_PROBE_PROMPT,
                    max_tokens=8,
                    timeout=self.timeout_sec,
                    ttft_event=None,
                    ttft_info=None,
                )
                self._broadcast_sample(spec.name, ts, result)
                with self._cost_lock:
                    self._total_cost_usd += result.billed_cost_usd
                logger.info(
                    "warmup %s/%d %s: %s ttft=%.1fms cost=$%.5f",
                    spec.name,
                    i + 1,
                    probes_per_provider,
                    result.status,
                    result.ttft_ms,
                    result.billed_cost_usd,
                )
                time.sleep(sleep_sec)

    def replay(
        self,
        trace: list[TraceRequest],
        *,
        speedup: float = 1.0,
        duration_sec: float = float("inf"),
    ) -> None:
        if not trace:
            logger.warning("replay called with empty trace")
            return
        run_start = time.time()
        threads: list[threading.Thread] = []
        n_policies = len(self.policies)
        policy_names = list(self.policies.keys())

        timed_out = False
        for i, req in enumerate(trace):
            # Wait until this request's arrival time. Must be a `while` loop:
            # a previous bug used `continue` in a for-loop, which advanced
            # to the next request and silently dropped any whose
            # inter-arrival gap exceeded the sleep cap.
            while True:
                if self._stop_event.is_set():
                    logger.info("stop event set; halting trace dispatch")
                    return
                if (time.time() - run_start) > duration_sec:
                    logger.info("duration cap %.0fs reached", duration_sec)
                    timed_out = True
                    break
                now_relative = (time.time() - run_start) * speedup
                wait = req.arrival_time_sec - now_relative
                if wait <= 0:
                    break
                time.sleep(min(wait / max(speedup, 1e-6), 5.0))
            if timed_out:
                break

            policy_name = policy_names[i % n_policies]
            policy = self.policies[policy_name]
            t = threading.Thread(
                target=self._dispatch_one,
                args=(policy, req, i),
                daemon=True,
            )
            t.start()
            threads.append(t)

            if (i + 1) % 25 == 0:
                threads = [t for t in threads if t.is_alive()]

        for t in threads:
            t.join(timeout=self.timeout_sec + 5)

    # ------------------------------------------------------------------
    # Per-request dispatch.
    # ------------------------------------------------------------------

    def _dispatch_one(
        self, policy: BasePolicy, req: TraceRequest, req_index: int
    ) -> None:
        if self._cost_exhausted(policy.name):
            return

        ctx = RequestContext(
            prompt_tokens=max(1, req.prompt_tokens or 1),
            completion_tokens_budget=max(1, req.max_tokens),
        )
        now = time.time()
        decision = policy.route(now, ctx)
        if decision.primary is None:
            self._record_no_route(policy, req, req_index, decision, now)
            return

        prompt = req.prompt if req.use_real_prompt else DEFAULT_PROBE_PROMPT
        expected_service_sec = max(0.5, req.max_tokens / 40.0 if req.max_tokens else 5.0)

        if policy.use_hedge:
            backup = select_safe_cheapest_backup(
                primary=decision.primary,
                states=policy.states,
                ctx=ctx,
                slo_sec=self.slo_sec,
                now=now,
            )
            hedge_delay_sec = float("inf")
            if (
                backup is not None
                and backup != decision.primary
                and decision.primary in policy.states
                and backup in policy.states
            ):
                hedge_delay_sec = compute_hedge_time_sec(
                    primary_state=policy.states[decision.primary],
                    backup_state=policy.states[backup],
                    slo_sec=self.slo_sec,
                    now=now,
                )
            decision.hedge = backup
            decision.hedge_delay_sec = hedge_delay_sec

            policy.charge_capacity(decision.primary, now, expected_service_sec)
            if backup is not None and math.isfinite(hedge_delay_sec):
                # Charge backup capacity at the *moment the backup thread
                # starts*, not after the hedged request returns. Without
                # this, concurrent route() calls during the backup's
                # lifetime would still see the slot as free.
                def _charge_backup(dispatch_ts: float, _b=backup) -> None:
                    policy.charge_capacity(_b, dispatch_ts, expected_service_sec)

                hedged = send_hedged_request(
                    send_fn=self._send_via_transport,
                    primary_provider=decision.primary,
                    backup_provider=backup,
                    hedge_delay_sec=hedge_delay_sec,
                    prompt=prompt,
                    max_tokens=req.max_tokens,
                    timeout=self.timeout_sec,
                    dispatch_overhead_sec=HEDGE_DISPATCH_OVERHEAD_SEC,
                    on_backup_dispatch=_charge_backup,
                )
                self._feed_back_hedged(policy, hedged)
                self._account_cost(policy, hedged)
                self._record_hedged(
                    policy, req, req_index, decision, hedged, hedge_delay_sec
                )
                return

            # Hedge disabled (no backup or hedge_time = inf): single send.
            result = send_request(
                send_fn=self._send_via_transport,
                provider=decision.primary,
                prompt=prompt,
                max_tokens=req.max_tokens,
                timeout=self.timeout_sec,
            )
            self._feed_back_single(policy, decision.primary, result)
            self._account_single(policy, result)
            self._record_single(policy, req, req_index, decision, result)
            return

        # Non-hedging policy.
        policy.charge_capacity(decision.primary, now, expected_service_sec)
        result = send_request(
            send_fn=self._send_via_transport,
            provider=decision.primary,
            prompt=prompt,
            max_tokens=req.max_tokens,
            timeout=self.timeout_sec,
        )
        self._feed_back_single(policy, decision.primary, result)
        self._account_single(policy, result)
        self._record_single(policy, req, req_index, decision, result)

    # ------------------------------------------------------------------
    # Profile + capacity feedback.
    # ------------------------------------------------------------------

    def _broadcast_sample(
        self, provider: str, ts: float, result: SingleRequestResult
    ) -> None:
        error_type = None if result.status == "success" else result.status
        ttft_ms = result.ttft_ms if result.status == "success" else -1.0
        for policy in self.policies.values():
            policy.add_sample(provider, ts, ttft_ms, error_type)

    def _feed_back_single(
        self, policy: BasePolicy, provider: str, result: SingleRequestResult
    ) -> None:
        error_type = None if result.status == "success" else result.status
        ttft_ms = result.ttft_ms if result.status == "success" else -1.0
        policy.add_sample(provider, result.start_ts or time.time(), ttft_ms, error_type)

    def _feed_back_hedged(self, policy: BasePolicy, hedged: HedgedResult) -> None:
        primary_provider = hedged.primary_result.provider.split("@")[0]
        self._feed_back_single(policy, primary_provider, hedged.primary_result)
        if hedged.backup_result is not None:
            backup_provider = hedged.backup_result.provider.split("@")[0]
            self._feed_back_single(policy, backup_provider, hedged.backup_result)

    # ------------------------------------------------------------------
    # Cost accounting + recording.
    # ------------------------------------------------------------------

    def _account_single(self, policy: BasePolicy, result: SingleRequestResult) -> None:
        with self._cost_lock:
            self._cost_per_policy[policy.name] += result.billed_cost_usd
            self._total_cost_usd += result.billed_cost_usd

    def _account_cost(self, policy: BasePolicy, hedged: HedgedResult) -> None:
        cost = hedged.primary_result.billed_cost_usd
        if hedged.backup_result is not None:
            cost += hedged.backup_result.billed_cost_usd
        with self._cost_lock:
            self._cost_per_policy[policy.name] += cost
            self._total_cost_usd += cost

    def _cost_exhausted(self, policy_name: str) -> bool:
        with self._cost_lock:
            if self._total_cost_usd >= self.max_cost_usd:
                self._stop_event.set()
                logger.warning(
                    "max_cost_usd $%.2f reached; halting", self.max_cost_usd
                )
                return True
        return False

    def _record_no_route(
        self,
        policy: BasePolicy,
        req: TraceRequest,
        req_index: int,
        decision: RoutingDecision,
        ts: float,
    ) -> None:
        sentinel = SingleRequestResult(
            ttft_ms=-1.0,
            e2e_ms=-1.0,
            status="no_route",
            provider="none",
            error_message=decision.notes or "no_route",
        )
        self.recorder.write_request(
            policy=policy.name,
            req_id=f"{req_index}_{uuid.uuid4().hex[:6]}",
            ctx_prompt_tokens=req.prompt_tokens,
            ctx_max_tokens=req.max_tokens,
            decision=decision,
            primary_result=sentinel,
            ts=ts,
        )

    def _record_single(
        self,
        policy: BasePolicy,
        req: TraceRequest,
        req_index: int,
        decision: RoutingDecision,
        result: SingleRequestResult,
    ) -> None:
        spec = self._spec_by_name.get(decision.primary or "")
        self.recorder.write_request(
            policy=policy.name,
            req_id=f"{req_index}_{uuid.uuid4().hex[:6]}",
            ctx_prompt_tokens=req.prompt_tokens,
            ctx_max_tokens=req.max_tokens,
            decision=decision,
            primary_result=result,
            tier=spec.tier if spec else None,
            transport=spec.transport_cfg.transport if spec else None,
        )

    def _record_hedged(
        self,
        policy: BasePolicy,
        req: TraceRequest,
        req_index: int,
        decision: RoutingDecision,
        hedged: HedgedResult,
        hedge_delay_sec: float,
    ) -> None:
        spec = self._spec_by_name.get(decision.primary or "")
        self.recorder.write_hedged(
            policy=policy.name,
            req_id=f"{req_index}_{uuid.uuid4().hex[:6]}",
            ctx_prompt_tokens=req.prompt_tokens,
            ctx_max_tokens=req.max_tokens,
            decision=decision,
            hedged=hedged,
            hedge_delay_sec=hedge_delay_sec,
            tier=spec.tier if spec else None,
            transport=spec.transport_cfg.transport if spec else None,
        )

    # ------------------------------------------------------------------
    # Cleanup.
    # ------------------------------------------------------------------

    def finalize(self) -> Path:
        """Flush recorder and write summary JSON."""
        path = self.recorder.write_summary(slo_ms=self.slo_ms)
        self.recorder.close()
        return path


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a real online evaluation against live provider APIs."
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        required=True,
        help="Path to the inventory JSON.",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=None,
        help="Path to a JSONL trace. If omitted, a synthetic trace is generated.",
    )
    parser.add_argument(
        "--policy",
        action="append",
        dest="policies",
        default=None,
        help="Repeatable. Policy names to run interleaved on the trace.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Cap the trace to at most this many rows.",
    )
    parser.add_argument(
        "--synthetic-rate",
        type=float,
        default=1.0,
        help="If using a synthetic trace, requests per second.",
    )
    parser.add_argument(
        "--synthetic-n",
        type=int,
        default=10,
        help="If using a synthetic trace, total request count.",
    )
    parser.add_argument(
        "--speedup", type=float, default=1.0, help="Replay speedup factor."
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=float("inf"),
        help="Wall-clock cap for the replay phase.",
    )
    parser.add_argument(
        "--warmup-probes",
        type=int,
        default=3,
        help="Number of warmup probes per provider before replay.",
    )
    parser.add_argument(
        "--slo-ms",
        type=float,
        default=None,
        help="Override the inventory's primary_slo_ms.",
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=5.0,
        help="Global cost cap; replay aborts when exceeded.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for CSV + summary JSON.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=DEFAULT_TIMEOUT_SEC,
        help="Per-request timeout.",
    )
    parser.add_argument(
        "--profile-window-sec",
        type=float,
        default=15 * 60.0,
        help="Rolling latency profile window length.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    inventory = load_inventory(args.inventory)
    if not args.policies:
        logger.error("at least one --policy is required")
        return 2

    trace = (
        load_trace_jsonl(args.trace, max_requests=args.max_requests)
        if args.trace is not None
        else make_synthetic_trace(
            n_requests=args.synthetic_n,
            rate_per_sec=args.synthetic_rate,
        )
    )
    if not trace:
        logger.error("trace is empty; nothing to replay")
        return 2

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2)
    )

    recorder = Recorder(output_dir)
    runner = RealExperimentRunner(
        inventory=inventory,
        policy_names=args.policies,
        recorder=recorder,
        slo_ms=args.slo_ms,
        max_cost_usd=args.max_cost_usd,
        timeout_sec=args.timeout_sec,
        profile_window_sec=args.profile_window_sec,
    )

    if args.warmup_probes > 0:
        runner.warmup(probes_per_provider=args.warmup_probes)

    runner.replay(
        trace=trace,
        speedup=args.speedup,
        duration_sec=args.duration_sec,
    )

    summary_path = runner.finalize()
    logger.info("wrote summary to %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
