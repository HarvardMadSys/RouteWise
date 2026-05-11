"""Trace-replay runner for real online evaluation.

Drives one or more policies through a real trace against live provider APIs.
By default every policy sees the full trace at the same wall-clock arrivals
with isolated state. Each request/policy decision is dispatched in a daemon
thread; per-policy profiles and capacity state are updated by feedback after
each completion.

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
DEFAULT_WARMUP_PROBES_PER_PROVIDER: int = 5
DEFAULT_WARMUP_PROBE_INTERVAL_SEC: float = 180.0
DEFAULT_PROFILE_PROBE_SLEEP_SEC: float = 0.5
DEFAULT_PERIODIC_PROBE_INTERVAL_SEC: float = 180.0
DEFAULT_MIN_PROFILE_SUCCESS_SAMPLES: int = 5
WARMUP_PROBE_PROMPT: str = "Write a one-sentence greeting."

logger = logging.getLogger(__name__)


@dataclass
class TraceRequest:
    """One row from a trace JSONL."""

    arrival_time_sec: float
    prompt: str
    prompt_tokens: int
    max_tokens: int
    prefix_id: str | None = None


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
        prefix_id        : ``prefix_id`` | ``sharegpt_conversation_id`` |
                           ``session_id`` (optional)

    ``time_compression`` divides arrival times so a long trace can be
    replayed in less wall-clock time.
    """
    out: list[TraceRequest] = []
    first_ts: float | None = None
    skipped_nonpositive_output_cap = 0
    trace_path = Path(path)
    with trace_path.open() as handle:
        for line_num, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{trace_path}:{line_num}: invalid JSON: {exc.msg}") from exc
            arrived_raw = _first_present(rec, ("arrived_at", "arrival_time_sec"))
            if arrived_raw is None:
                raise ValueError(
                    f"{trace_path}:{line_num}: missing arrival timestamp "
                    "(expected arrived_at or arrival_time_sec)"
                )
            arrived = _coerce_float(trace_path, line_num, "arrival timestamp", arrived_raw)
            if first_ts is None:
                first_ts = arrived
            relative = (arrived - first_ts) / max(time_compression, 1e-6)
            if relative < trace_start_sec:
                continue
            if relative > trace_end_sec:
                break
            max_tokens_raw = _first_present(rec, ("num_decode_tokens", "max_tokens"))
            if max_tokens_raw is None:
                raise ValueError(
                    f"{trace_path}:{line_num}: missing output token cap "
                    "(expected num_decode_tokens or max_tokens)"
                )
            max_tokens = _coerce_int(trace_path, line_num, "output token cap", max_tokens_raw)
            if max_tokens <= 0:
                skipped_nonpositive_output_cap += 1
                continue
            prompt = _first_present(rec, ("prompt_text", "prompt"))
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"{trace_path}:{line_num}: missing non-empty prompt "
                    "(expected prompt_text or prompt)"
                )
            prompt_tokens_raw = _first_present(rec, ("num_prefill_tokens", "prompt_tokens"))
            if prompt_tokens_raw is None:
                raise ValueError(
                    f"{trace_path}:{line_num}: missing prompt token count "
                    "(expected num_prefill_tokens or prompt_tokens)"
                )
            prompt_tokens = _coerce_int(
                trace_path, line_num, "prompt token count", prompt_tokens_raw
            )
            if prompt_tokens < 0:
                raise ValueError(f"{trace_path}:{line_num}: prompt token count must be >= 0")
            out.append(
                TraceRequest(
                    arrival_time_sec=relative,
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                    max_tokens=max_tokens,
                    prefix_id=_prefix_id_from_record(rec),
                )
            )
            if max_requests is not None and len(out) >= max_requests:
                break
    if skipped_nonpositive_output_cap:
        logger.warning(
            "Skipped %d trace rows from %s because output token cap was <= 0",
            skipped_nonpositive_output_cap,
            trace_path,
        )
    return out


def _first_present(rec: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the first key that exists and is not None.

    Do not use ``or`` here: valid trace values such as ``0`` must not be
    treated as missing.
    """
    for key in keys:
        if key in rec and rec[key] is not None:
            return rec[key]
    return None


def _prefix_id_from_record(rec: dict[str, Any]) -> str | None:
    value = _first_present(rec, ("prefix_id", "sharegpt_conversation_id", "session_id"))
    if value is None:
        return None
    prefix_id = str(value).strip()
    return prefix_id or None


def _coerce_float(path: Path, line_num: int, field: str, value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{line_num}: {field} must be numeric, got {value!r}") from exc


def _coerce_int(path: Path, line_num: int, field: str, value: Any) -> int:
    try:
        as_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{line_num}: {field} must be an integer, got {value!r}") from exc
    if not math.isfinite(as_float) or not as_float.is_integer():
        raise ValueError(f"{path}:{line_num}: {field} must be an integer, got {value!r}")
    return int(as_float)


@dataclass
class _PreparedDispatch:
    """A routed request ready for physical execution."""

    policy: BasePolicy
    req: TraceRequest
    req_index: int
    decision: RoutingDecision
    ctx: RequestContext
    prompt: str
    expected_service_sec: float
    backup: str | None
    hedge_delay_sec: float
    primary_capacity_id: int | None = None
    primary_cached_input_tokens: int = 0
    backup_cached_input_tokens: int = 0
    primary_routing_estimated_cost_usd: float | None = None
    backup_routing_estimated_cost_usd: float | None = None


@dataclass
class _PeriodicProbeHandle:
    """Background profile-maintenance probe loop."""

    thread: threading.Thread
    stop_event: threading.Event


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
        prefix_cache_routing: bool = False,
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
                prefix_cache_routing=prefix_cache_routing,
            )
            for name in policy_names
        }
        self.prefix_cache_routing = bool(prefix_cache_routing)

        self._spec_by_name: dict[str, ProviderSpec] = {
            spec.name: spec for spec in inventory.providers
        }
        self._transports: dict[str, BaseTransport] = self._build_transports()
        self._or_base_cfg: TransportConfig | None = self._first_openrouter_cfg()

        self._cost_lock = threading.Lock()
        self._cost_per_policy: dict[str, float] = dict.fromkeys(policy_names, 0.0)
        self._total_cost_usd: float = 0.0
        self._profile_probe_cost_usd: float = 0.0
        self._profile_probe_counts: dict[str, int] = {"warmup": 0, "periodic": 0}
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
            if (
                spec.transport_cfg.transport == "openrouter"
                and spec.tier == "api"
                and spec.transport_cfg.billing_mode == "metered"
            ):
                return spec.transport_cfg
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
        cancel_event: threading.Event | None = None,
    ) -> SingleRequestResult:
        """Dispatch one request to a real or sentinel provider."""
        if provider == OR_AUTO_SENTINEL or provider in OR_SORT_SENTINEL_TO_MODE:
            return self._send_or_sentinel(
                provider,
                prompt,
                max_tokens,
                timeout,
                ttft_event,
                ttft_info,
                cancel_event,
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
            cancel_event=cancel_event,
        )

    def _send_or_sentinel(
        self,
        sentinel: str,
        prompt: str,
        max_tokens: int,
        timeout: int,
        ttft_event: threading.Event | None,
        ttft_info: dict[str, Any] | None,
        cancel_event: threading.Event | None = None,
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
            provider_only=(self.inventory.openrouter_provider_only or base.provider_only),
            provider_ignore=(self.inventory.openrouter_provider_ignore or base.provider_ignore),
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
            cancel_event=cancel_event,
        )

    # ------------------------------------------------------------------
    # Warmup + replay.
    # ------------------------------------------------------------------

    def probe_profiles(
        self,
        *,
        probes_per_provider: int = 1,
        sleep_sec: float = DEFAULT_PROFILE_PROBE_SLEEP_SEC,
        round_interval_sec: float = 0.0,
        phase: str = "periodic",
        stop_event: threading.Event | None = None,
    ) -> None:
        """Probe every provider and broadcast observations to all policies.

        This is profile-maintenance infrastructure. It is intentionally shared
        across policies so probe traffic does not multiply by policy count.
        Transient provider failures are retried until the provider yields one
        successful TTFT sample; configuration failures such as missing API keys
        still fail fast.
        """
        if probes_per_provider <= 0:
            return
        for round_idx in range(probes_per_provider):
            for spec in self.inventory.providers:
                if self._probe_stop_requested(stop_event):
                    return
                if self._cost_exhausted("profile_maintenance"):
                    return
                result = self._probe_provider_until_success(
                    spec,
                    phase=phase,
                    round_idx=round_idx,
                    probes_per_provider=probes_per_provider,
                    retry_sleep_sec=sleep_sec,
                    stop_event=stop_event,
                )
                if result is None:
                    return
                if sleep_sec > 0 and self._probe_sleep(sleep_sec, stop_event):
                    return
            if round_idx + 1 < probes_per_provider and round_interval_sec > 0:
                logger.info(
                    "%s probe round %d/%d complete; sleeping %.1fs",
                    phase,
                    round_idx + 1,
                    probes_per_provider,
                    round_interval_sec,
                )
                if self._stop_event.wait(round_interval_sec):
                    return

    def _probe_provider_until_success(
        self,
        spec: ProviderSpec,
        *,
        phase: str,
        round_idx: int,
        probes_per_provider: int,
        retry_sleep_sec: float,
        stop_event: threading.Event | None,
    ) -> SingleRequestResult | None:
        attempt = 0
        while True:
            if self._probe_stop_requested(stop_event):
                return None
            if self._cost_exhausted("profile_maintenance"):
                return None

            attempt += 1
            ts = time.time()
            try:
                result = self._send_via_transport(
                    provider=spec.name,
                    prompt=WARMUP_PROBE_PROMPT,
                    max_tokens=8,
                    timeout=self.timeout_sec,
                    ttft_event=None,
                    ttft_info=None,
                )
            except RuntimeError as exc:
                if "Missing API key" in str(exc):
                    raise
                logger.warning(
                    "%s probe round %d/%d %s attempt %d raised %s; retrying",
                    phase,
                    round_idx + 1,
                    probes_per_provider,
                    spec.name,
                    attempt,
                    exc,
                )
                if retry_sleep_sec > 0 and self._probe_sleep(retry_sleep_sec, stop_event):
                    return None
                continue

            self._account_profile_probe_attempt(phase, result)
            if result.status == "success" and result.ttft_ms > 0:
                self._broadcast_sample(spec.name, ts, result)
                logger.info(
                    (
                        "%s probe round %d/%d %s: success ttft=%.1fms "
                        "billed=$%.5f physical=$%.5f attempts=%d"
                    ),
                    phase,
                    round_idx + 1,
                    probes_per_provider,
                    spec.name,
                    result.ttft_ms,
                    result.billed_cost_usd,
                    self._single_physical_cost(result),
                    attempt,
                )
                return result

            logger.warning(
                (
                    "%s probe round %d/%d %s attempt %d failed: %s "
                    "ttft=%.1fms billed=$%.5f physical=$%.5f; retrying"
                ),
                phase,
                round_idx + 1,
                probes_per_provider,
                spec.name,
                attempt,
                result.status,
                result.ttft_ms,
                result.billed_cost_usd,
                self._single_physical_cost(result),
            )
            if retry_sleep_sec > 0 and self._probe_sleep(retry_sleep_sec, stop_event):
                return None

    def _account_profile_probe_attempt(
        self,
        phase: str,
        result: SingleRequestResult,
    ) -> None:
        physical_cost = self._single_physical_cost(result)
        with self._cost_lock:
            self._total_cost_usd += physical_cost
            self._profile_probe_cost_usd += physical_cost
            self._profile_probe_counts[phase] = self._profile_probe_counts.get(phase, 0) + 1

    def _probe_stop_requested(self, stop_event: threading.Event | None = None) -> bool:
        return self._stop_event.is_set() or bool(stop_event is not None and stop_event.is_set())

    def _probe_sleep(
        self,
        sleep_sec: float,
        stop_event: threading.Event | None = None,
    ) -> bool:
        deadline = time.time() + max(0.0, sleep_sec)
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return self._probe_stop_requested(stop_event)
            if self._stop_event.wait(min(remaining, 0.1)):
                return True
            if stop_event is not None and stop_event.wait(0):
                return True

    def warmup(
        self,
        probes_per_provider: int = DEFAULT_WARMUP_PROBES_PER_PROVIDER,
        sleep_sec: float = DEFAULT_PROFILE_PROBE_SLEEP_SEC,
        round_interval_sec: float = DEFAULT_WARMUP_PROBE_INTERVAL_SEC,
    ) -> None:
        """Seed all policy-local latency profiles before replay."""
        self.probe_profiles(
            probes_per_provider=probes_per_provider,
            sleep_sec=sleep_sec,
            round_interval_sec=round_interval_sec,
            phase="warmup",
        )

    def export_initial_profile(self) -> dict[str, Any]:
        """Export one policy's current latency profiles for later reuse."""
        policy = next(iter(self.policies.values()), None)
        now = time.time()
        providers: dict[str, Any] = {}
        if policy is None:
            return {"version": 1, "created_ts": now, "providers": providers}
        for spec in self.inventory.providers:
            state = policy.states.get(spec.name)
            if state is None:
                continue
            state.profile._active_samples(now)
            providers[spec.name] = {
                "samples": [
                    {"ts": ts, "ttft_ms": ttft_ms} for ts, ttft_ms in state.profile.samples
                ],
                "errors": [
                    {"ts": ts, "error_type": error_type}
                    for ts, error_type in state.profile.error_samples
                ],
            }
        return {
            "version": 1,
            "created_ts": now,
            "providers": providers,
        }

    def load_initial_profile(
        self,
        path: Path | str,
        *,
        rebase_to_now: bool = True,
    ) -> None:
        """Load prebuilt latency profile samples into every policy."""
        raw = json.loads(Path(path).read_text())
        provider_entries = raw.get("providers", {})
        if not isinstance(provider_entries, dict):
            raise ValueError(f"invalid initial profile {path}: providers must be an object")

        timestamps: list[float] = []
        for entry in provider_entries.values():
            if not isinstance(entry, dict):
                continue
            for sample in entry.get("samples", []):
                if isinstance(sample, dict) and sample.get("ts") is not None:
                    timestamps.append(float(sample["ts"]))
            for error in entry.get("errors", []):
                if isinstance(error, dict) and error.get("ts") is not None:
                    timestamps.append(float(error["ts"]))
        offset = time.time() - max(timestamps) if rebase_to_now and timestamps else 0.0

        loaded_success = 0
        loaded_errors = 0
        for policy in self.policies.values():
            for provider, entry in provider_entries.items():
                if not isinstance(entry, dict):
                    continue
                state = policy.states.get(str(provider))
                if state is None:
                    continue
                for sample in entry.get("samples", []):
                    if not isinstance(sample, dict):
                        continue
                    state.profile.add_sample(
                        float(sample["ts"]) + offset,
                        float(sample["ttft_ms"]),
                    )
                    loaded_success += 1
                for error in entry.get("errors", []):
                    if not isinstance(error, dict):
                        continue
                    state.profile.add_sample(
                        float(error["ts"]) + offset,
                        -1.0,
                        str(error.get("error_type") or "error"),
                    )
                    loaded_errors += 1

        logger.info(
            "loaded initial profile from %s: success_samples=%d error_samples=%d",
            path,
            loaded_success,
            loaded_errors,
        )

    def validate_profile_bootstrap(
        self,
        *,
        min_success_samples: int = DEFAULT_MIN_PROFILE_SUCCESS_SAMPLES,
    ) -> None:
        """Fail before replay if profile-dependent policies lack samples.

        Call this only before replay starts. It intentionally reads profile
        counts without taking every policy lock because no replay/probe worker
        should be mutating profiles yet.
        """
        if min_success_samples <= 0:
            return
        policies = [
            policy for policy in self.policies.values() if policy.requires_latency_profile_bootstrap
        ]
        if not policies:
            return

        now = time.time()
        missing: list[str] = []
        for policy in policies:
            for spec in self.inventory.providers:
                state = policy.states.get(spec.name)
                count = state.profile.sample_count(now) if state is not None else 0
                if count < min_success_samples:
                    missing.append(f"{policy.name}:{spec.name}={count}")

        if missing:
            preview = ", ".join(missing[:12])
            suffix = "" if len(missing) <= 12 else f", ... (+{len(missing) - 12})"
            raise RuntimeError(
                "profile bootstrap failed: expected at least "
                f"{min_success_samples} successful samples per provider for "
                f"profile-dependent policies; got {preview}{suffix}"
            )

    def replay(
        self,
        trace: list[TraceRequest],
        *,
        speedup: float = 1.0,
        duration_sec: float = float("inf"),
        coalesce_identical_actions: bool = False,
        periodic_probe_interval_sec: float = 0.0,
        periodic_probe_sleep_sec: float = DEFAULT_PROFILE_PROBE_SLEEP_SEC,
    ) -> None:
        """Replay a trace against the configured policies.

        Default semantics are paper-safe: every policy gets the full trace at
        the same wall-clock request arrivals with isolated policy state.
        ``coalesce_identical_actions=True`` keeps those full-trace semantics
        but executes identical physical actions once, then fans the observed
        result out to each policy's virtual accounting.
        """
        if not trace:
            logger.warning("replay called with empty trace")
            return
        probe_handle = self._start_periodic_profile_probe_thread(
            interval_sec=periodic_probe_interval_sec,
            sleep_sec=periodic_probe_sleep_sec,
        )
        try:
            if coalesce_identical_actions:
                self._replay_coalesced_full_trace(trace, speedup=speedup, duration_sec=duration_sec)
                return

            self._replay_parallel_full_trace(trace, speedup=speedup, duration_sec=duration_sec)
        finally:
            self._stop_periodic_profile_probe_thread(probe_handle)

    def _start_periodic_profile_probe_thread(
        self,
        *,
        interval_sec: float,
        sleep_sec: float,
    ) -> _PeriodicProbeHandle | None:
        if interval_sec <= 0:
            return None

        stop_event = threading.Event()

        def _loop() -> None:
            logger.info("periodic profile probing enabled: interval=%.1fs", interval_sec)
            while not stop_event.wait(interval_sec):
                if self._stop_event.is_set():
                    return
                logger.info("periodic profile probe round starting")
                self.probe_profiles(
                    probes_per_provider=1,
                    sleep_sec=sleep_sec,
                    phase="periodic",
                    stop_event=stop_event,
                )

        thread = threading.Thread(
            target=_loop,
            name="real-eval-profile-prober",
            daemon=True,
        )
        thread.start()
        return _PeriodicProbeHandle(thread=thread, stop_event=stop_event)

    @staticmethod
    def _stop_periodic_profile_probe_thread(
        handle: _PeriodicProbeHandle | None,
    ) -> None:
        if handle is None:
            return
        handle.stop_event.set()
        handle.thread.join(timeout=5.0)

    def _wait_for_request_arrival(
        self,
        run_start: float,
        req: TraceRequest,
        *,
        speedup: float,
        duration_sec: float,
    ) -> bool:
        """Return True when ``req`` is due; False on stop/duration cap."""
        while True:
            if self._stop_event.is_set():
                logger.info("stop event set; halting trace dispatch")
                return False
            if (time.time() - run_start) > duration_sec:
                logger.info("duration cap %.0fs reached", duration_sec)
                return False
            now_relative = (time.time() - run_start) * speedup
            wait = req.arrival_time_sec - now_relative
            if wait <= 0:
                return True
            time.sleep(min(wait / max(speedup, 1e-6), 5.0))

    def _join_threads(self, threads: list[threading.Thread]) -> None:
        for t in threads:
            t.join(timeout=self.timeout_sec + 5)

    def _replay_parallel_full_trace(
        self,
        trace: list[TraceRequest],
        *,
        speedup: float,
        duration_sec: float,
    ) -> None:
        run_start = time.time()
        threads: list[threading.Thread] = []
        policies = list(self.policies.values())
        for i, req in enumerate(trace):
            if not self._wait_for_request_arrival(
                run_start, req, speedup=speedup, duration_sec=duration_sec
            ):
                break
            for policy in policies:
                t = threading.Thread(
                    target=self._dispatch_one,
                    args=(policy, req, i),
                    daemon=True,
                )
                t.start()
                threads.append(t)

            if (i + 1) % 25 == 0:
                threads = [t for t in threads if t.is_alive()]

        self._join_threads(threads)

    def _replay_coalesced_full_trace(
        self,
        trace: list[TraceRequest],
        *,
        speedup: float,
        duration_sec: float,
    ) -> None:
        run_start = time.time()
        threads: list[threading.Thread] = []
        policies = list(self.policies.values())
        for i, req in enumerate(trace):
            if not self._wait_for_request_arrival(
                run_start, req, speedup=speedup, duration_sec=duration_sec
            ):
                break

            groups: dict[tuple[Any, ...], list[_PreparedDispatch]] = {}
            for policy in policies:
                prepared = self._prepare_dispatch(policy, req, i)
                if prepared is None:
                    continue
                groups.setdefault(self._action_key(prepared), []).append(prepared)

            for prepareds in groups.values():
                t = threading.Thread(
                    target=self._execute_coalesced_group,
                    args=(prepareds,),
                    daemon=True,
                )
                t.start()
                threads.append(t)

            if (i + 1) % 25 == 0:
                threads = [t for t in threads if t.is_alive()]

        self._join_threads(threads)

    # ------------------------------------------------------------------
    # Per-request dispatch.
    # ------------------------------------------------------------------

    def _prepare_dispatch(
        self, policy: BasePolicy, req: TraceRequest, req_index: int
    ) -> _PreparedDispatch | None:
        if self._cost_exhausted(policy.name):
            return None

        ctx = RequestContext(
            prompt_tokens=max(1, req.prompt_tokens or 1),
            completion_tokens_budget=max(1, req.max_tokens),
            prefix_id=req.prefix_id,
        )
        now = time.time()
        decision = policy.route(now, ctx)
        if decision.primary is None:
            self._record_no_route(policy, req, req_index, decision, now)
            return None

        prompt = req.prompt
        expected_service_sec = max(0.5, req.max_tokens / 40.0 if req.max_tokens else 5.0)

        backup: str | None = None
        hedge_delay_sec = float("inf")
        if policy.use_hedge:
            backup = select_safe_cheapest_backup(
                primary=decision.primary,
                states=policy.states,
                ctx=ctx,
                slo_sec=self.slo_sec,
                now=now,
                cost_fn=lambda state: policy.request_cost_for_state(state, ctx),
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

        primary_capacity_id = policy.charge_capacity(decision.primary, now, expected_service_sec)
        primary_cached, primary_estimated_cost = policy.routing_cache_diagnostics(
            decision.primary, ctx
        )
        backup_cached = 0
        backup_estimated_cost: float | None = None
        if backup is not None:
            backup_cached, backup_estimated_cost = policy.routing_cache_diagnostics(backup, ctx)
        return _PreparedDispatch(
            policy=policy,
            req=req,
            req_index=req_index,
            decision=decision,
            ctx=ctx,
            prompt=prompt,
            expected_service_sec=expected_service_sec,
            backup=backup,
            hedge_delay_sec=hedge_delay_sec,
            primary_capacity_id=primary_capacity_id,
            primary_cached_input_tokens=primary_cached,
            backup_cached_input_tokens=backup_cached,
            primary_routing_estimated_cost_usd=primary_estimated_cost,
            backup_routing_estimated_cost_usd=backup_estimated_cost,
        )

    @staticmethod
    def _is_hedged_action(prepared: _PreparedDispatch) -> bool:
        return prepared.backup is not None and math.isfinite(prepared.hedge_delay_sec)

    def _action_key(self, prepared: _PreparedDispatch) -> tuple[Any, ...]:
        """Physical action identity for safe coalescing."""
        hedged = self._is_hedged_action(prepared)
        return (
            "hedged" if hedged else "single",
            prepared.decision.primary,
            prepared.backup if hedged else None,
            round(prepared.hedge_delay_sec, 6) if hedged else None,
            prepared.prompt,
            prepared.req.max_tokens,
        )

    def _send_single_with_rate_limit_fallback(
        self,
        prepared: _PreparedDispatch,
        *,
        initial_result: SingleRequestResult | None = None,
    ) -> tuple[str, SingleRequestResult, list[tuple[str, SingleRequestResult]]]:
        """Send a non-hedged request, retrying 429s on alternate providers."""
        provider = prepared.decision.primary or ""
        capacity_id = prepared.primary_capacity_id
        attempts: list[tuple[str, SingleRequestResult]] = []
        excluded: set[str] = set()

        try:
            if initial_result is not None:
                result = initial_result
            else:
                result = send_request(
                    send_fn=self._send_via_transport,
                    provider=provider,
                    prompt=prepared.prompt,
                    max_tokens=prepared.req.max_tokens,
                    timeout=self.timeout_sec,
                )
        finally:
            prepared.policy.release_capacity(provider, capacity_id, time.time())
        attempts.append((provider, result))
        if result.rate_limited:
            excluded.add(provider)
            return self._continue_after_rate_limited_attempts(prepared, attempts, excluded)

        final_provider, final_result = attempts[-1]
        self._annotate_rate_limit_fallback(final_result, attempts)
        return final_provider, final_result, attempts

    def _continue_after_rate_limited_attempts(
        self,
        prepared: _PreparedDispatch,
        attempts: list[tuple[str, SingleRequestResult]],
        excluded: set[str],
    ) -> tuple[str, SingleRequestResult, list[tuple[str, SingleRequestResult]]]:
        """Continue a request after one or more provider-local HTTP 429s."""
        while attempts and attempts[-1][1].rate_limited:
            fallback_candidates = prepared.policy.rate_limit_fallback_candidates(
                time.time(),
                prepared.ctx,
                excluded=excluded,
            )
            provider = next(
                (candidate for candidate in fallback_candidates if candidate not in excluded),
                None,
            )
            if provider is None:
                break

            capacity_id = prepared.policy.charge_capacity(
                provider,
                time.time(),
                prepared.expected_service_sec,
            )
            try:
                result = send_request(
                    send_fn=self._send_via_transport,
                    provider=provider,
                    prompt=prepared.prompt,
                    max_tokens=prepared.req.max_tokens,
                    timeout=self.timeout_sec,
                )
            finally:
                prepared.policy.release_capacity(provider, capacity_id, time.time())
            attempts.append((provider, result))
            if result.rate_limited:
                excluded.add(provider)

        final_provider, final_result = attempts[-1]
        self._annotate_rate_limit_fallback(final_result, attempts)
        return final_provider, final_result, attempts

    @staticmethod
    def _annotate_rate_limit_fallback(
        final_result: SingleRequestResult,
        attempts: list[tuple[str, SingleRequestResult]],
    ) -> None:
        prior = attempts[:-1]
        rate_limited_before_final = sum(1 for _, result in prior if result.rate_limited)
        if rate_limited_before_final <= 0:
            return
        final_result.rate_limited = True
        # CSV compatibility: after 429 fallback, retry_count means provider
        # switches caused by 429s, not same-provider transport retries.
        final_result.retry_count += rate_limited_before_final
        final_result.retry_sleep_ms += sum(result.retry_sleep_ms for _, result in prior)

    def _feed_back_single_attempts(
        self,
        policy: BasePolicy,
        attempts: list[tuple[str, SingleRequestResult]],
    ) -> None:
        for provider, result in attempts:
            self._feed_back_single(policy, provider, result)

    def _account_single_attempts(
        self,
        policy: BasePolicy,
        attempts: list[tuple[str, SingleRequestResult]],
        *,
        physical: bool = True,
    ) -> None:
        with self._cost_lock:
            self._cost_per_policy[policy.name] += sum(
                result.billed_cost_usd for _, result in attempts
            )
            if physical:
                self._total_cost_usd += sum(
                    self._single_physical_cost(result) for _, result in attempts
                )

    def _dispatch_one(self, policy: BasePolicy, req: TraceRequest, req_index: int) -> None:
        prepared = self._prepare_dispatch(policy, req, req_index)
        if prepared is None:
            return
        self._execute_prepared_dispatch(prepared)

    def _execute_prepared_dispatch(self, prepared: _PreparedDispatch) -> None:
        policy = prepared.policy
        req = prepared.req
        decision = prepared.decision

        if self._is_hedged_action(prepared):
            assert prepared.backup is not None

            # Charge backup capacity at the *moment the backup thread
            # starts*, not after the hedged request returns. Without
            # this, concurrent route() calls during the backup's
            # lifetime would still see the slot as free.
            backup_capacity_id: int | None = None
            backup_capacity_lock = threading.Lock()

            def _charge_backup(dispatch_ts: float, _b=prepared.backup) -> None:
                nonlocal backup_capacity_id
                capacity_id = policy.charge_capacity(_b, dispatch_ts, prepared.expected_service_sec)
                with backup_capacity_lock:
                    backup_capacity_id = capacity_id

            try:
                hedged = send_hedged_request(
                    send_fn=self._send_via_transport,
                    primary_provider=decision.primary or "",
                    backup_provider=prepared.backup,
                    hedge_delay_sec=prepared.hedge_delay_sec,
                    prompt=prepared.prompt,
                    max_tokens=req.max_tokens,
                    timeout=self.timeout_sec,
                    dispatch_overhead_sec=HEDGE_DISPATCH_OVERHEAD_SEC,
                    on_backup_dispatch=_charge_backup,
                )
            finally:
                policy.release_capacity(
                    decision.primary,
                    prepared.primary_capacity_id,
                    time.time(),
                )
                with backup_capacity_lock:
                    capacity_id = backup_capacity_id
                policy.release_capacity(prepared.backup, capacity_id, time.time())
            hedged_attempts = [(decision.primary or "", hedged.primary_result)]
            if hedged.backup_result is not None:
                hedged_attempts.append((prepared.backup, hedged.backup_result))
            if hedged_attempts and all(result.rate_limited for _, result in hedged_attempts):
                excluded = {provider for provider, _ in hedged_attempts if provider}
                initial_attempt_count = len(hedged_attempts)
                final_provider, result, attempts = self._continue_after_rate_limited_attempts(
                    prepared,
                    hedged_attempts,
                    excluded,
                )
                if len(attempts) > initial_attempt_count:
                    self._feed_back_single_attempts(policy, attempts)
                    self._account_single_attempts(policy, attempts)
                    primary_cached, primary_estimated_cost = (
                        prepared.primary_cached_input_tokens,
                        prepared.primary_routing_estimated_cost_usd,
                    )
                    if final_provider != decision.primary:
                        primary_cached, primary_estimated_cost = policy.routing_cache_diagnostics(
                            final_provider,
                            prepared.ctx,
                        )
                    self._record_prefix_cache_dispatch(
                        policy,
                        final_provider,
                        prepared.ctx,
                        result,
                    )
                    self._record_single(
                        policy,
                        req,
                        prepared.req_index,
                        decision,
                        result,
                        primary_cached_input_tokens=primary_cached,
                        primary_routing_estimated_cost_usd=primary_estimated_cost,
                    )
                    return
            self._feed_back_hedged(policy, hedged)
            self._account_cost(policy, hedged)
            self._record_prefix_cache_dispatches(prepared, hedged)
            self._record_hedged(
                policy,
                req,
                prepared.req_index,
                decision,
                hedged,
                prepared.hedge_delay_sec,
                primary_cached_input_tokens=prepared.primary_cached_input_tokens,
                backup_cached_input_tokens=prepared.backup_cached_input_tokens,
                primary_routing_estimated_cost_usd=(prepared.primary_routing_estimated_cost_usd),
                backup_routing_estimated_cost_usd=(prepared.backup_routing_estimated_cost_usd),
            )
            return

        final_provider, result, attempts = self._send_single_with_rate_limit_fallback(prepared)
        self._feed_back_single_attempts(policy, attempts)
        self._account_single_attempts(policy, attempts)
        primary_cached, primary_estimated_cost = (
            prepared.primary_cached_input_tokens,
            (prepared.primary_routing_estimated_cost_usd),
        )
        if final_provider != decision.primary:
            primary_cached, primary_estimated_cost = policy.routing_cache_diagnostics(
                final_provider,
                prepared.ctx,
            )
        self._record_prefix_cache_dispatch(
            policy,
            final_provider,
            prepared.ctx,
            result,
        )
        self._record_single(
            policy,
            req,
            prepared.req_index,
            decision,
            result,
            primary_cached_input_tokens=primary_cached,
            primary_routing_estimated_cost_usd=primary_estimated_cost,
        )

    def _execute_coalesced_group(self, prepareds: list[_PreparedDispatch]) -> None:
        """Execute one physical action and fan out result to virtual policies."""
        if not prepareds:
            return
        if len(prepareds) == 1:
            self._execute_prepared_dispatch(prepareds[0])
            return

        first = prepareds[0]
        if self._is_hedged_action(first):
            assert first.backup is not None
            backup_capacity_ids: dict[int, int | None] = {}
            backup_capacity_lock = threading.Lock()

            def _charge_all_backups(dispatch_ts: float) -> None:
                for prepared in prepareds:
                    assert prepared.backup is not None
                    capacity_id = prepared.policy.charge_capacity(
                        prepared.backup,
                        dispatch_ts,
                        prepared.expected_service_sec,
                    )
                    with backup_capacity_lock:
                        backup_capacity_ids[id(prepared)] = capacity_id

            try:
                hedged = send_hedged_request(
                    send_fn=self._send_via_transport,
                    primary_provider=first.decision.primary or "",
                    backup_provider=first.backup,
                    hedge_delay_sec=first.hedge_delay_sec,
                    prompt=first.prompt,
                    max_tokens=first.req.max_tokens,
                    timeout=self.timeout_sec,
                    dispatch_overhead_sec=HEDGE_DISPATCH_OVERHEAD_SEC,
                    on_backup_dispatch=_charge_all_backups,
                )
            finally:
                with backup_capacity_lock:
                    backup_ids = dict(backup_capacity_ids)
                for prepared in prepareds:
                    prepared.policy.release_capacity(
                        prepared.decision.primary,
                        prepared.primary_capacity_id,
                        time.time(),
                    )
                    prepared.policy.release_capacity(
                        prepared.backup,
                        backup_ids.get(id(prepared)),
                        time.time(),
                    )
            physical_cost = self._hedged_physical_cost(hedged)
            for prepared in prepareds:
                self._feed_back_hedged(prepared.policy, hedged)
                self._account_cost(prepared.policy, hedged, physical=False)
                self._record_prefix_cache_dispatches(prepared, hedged)
                self._record_hedged(
                    prepared.policy,
                    prepared.req,
                    prepared.req_index,
                    prepared.decision,
                    hedged,
                    prepared.hedge_delay_sec,
                    primary_cached_input_tokens=prepared.primary_cached_input_tokens,
                    backup_cached_input_tokens=prepared.backup_cached_input_tokens,
                    primary_routing_estimated_cost_usd=(
                        prepared.primary_routing_estimated_cost_usd
                    ),
                    backup_routing_estimated_cost_usd=(prepared.backup_routing_estimated_cost_usd),
                )
            self._account_physical_cost(physical_cost)
            return

        try:
            result = send_request(
                send_fn=self._send_via_transport,
                provider=first.decision.primary or "",
                prompt=first.prompt,
                max_tokens=first.req.max_tokens,
                timeout=self.timeout_sec,
            )
        except Exception:
            for prepared in prepareds:
                prepared.policy.release_capacity(
                    prepared.decision.primary,
                    prepared.primary_capacity_id,
                    time.time(),
                )
            raise
        if result.rate_limited:
            for prepared in prepareds:
                final_provider, final_result, attempts = self._send_single_with_rate_limit_fallback(
                    prepared,
                    initial_result=result,
                )
                self._feed_back_single_attempts(prepared.policy, attempts)
                self._account_single_attempts(prepared.policy, attempts)
                primary_cached, primary_estimated_cost = (
                    prepared.primary_cached_input_tokens,
                    (prepared.primary_routing_estimated_cost_usd),
                )
                if final_provider != prepared.decision.primary:
                    primary_cached, primary_estimated_cost = (
                        prepared.policy.routing_cache_diagnostics(
                            final_provider,
                            prepared.ctx,
                        )
                    )
                self._record_prefix_cache_dispatch(
                    prepared.policy,
                    final_provider,
                    prepared.ctx,
                    final_result,
                )
                self._record_single(
                    prepared.policy,
                    prepared.req,
                    prepared.req_index,
                    prepared.decision,
                    final_result,
                    primary_cached_input_tokens=primary_cached,
                    primary_routing_estimated_cost_usd=primary_estimated_cost,
                )
            return

        physical_cost = self._single_physical_cost(result)
        for prepared in prepareds:
            prepared.policy.release_capacity(
                prepared.decision.primary,
                prepared.primary_capacity_id,
                time.time(),
            )
        for prepared in prepareds:
            self._feed_back_single(prepared.policy, prepared.decision.primary or "", result)
            self._account_single(prepared.policy, result, physical=False)
            self._record_prefix_cache_dispatch(
                prepared.policy,
                prepared.decision.primary,
                prepared.ctx,
                result,
            )
            self._record_single(
                prepared.policy,
                prepared.req,
                prepared.req_index,
                prepared.decision,
                result,
                primary_cached_input_tokens=prepared.primary_cached_input_tokens,
                primary_routing_estimated_cost_usd=(prepared.primary_routing_estimated_cost_usd),
            )
        self._account_physical_cost(physical_cost)

    @staticmethod
    def _record_prefix_cache_dispatch(
        policy: BasePolicy,
        provider: str | None,
        ctx: RequestContext,
        result: SingleRequestResult,
    ) -> None:
        policy.record_prefix_cache_dispatch(
            provider,
            ctx,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )

    def _record_prefix_cache_dispatches(
        self,
        prepared: _PreparedDispatch,
        hedged: HedgedResult,
    ) -> None:
        self._record_prefix_cache_dispatch(
            prepared.policy,
            prepared.decision.primary,
            prepared.ctx,
            hedged.primary_result,
        )
        if hedged.backup_result is not None:
            self._record_prefix_cache_dispatch(
                prepared.policy,
                prepared.backup,
                prepared.ctx,
                hedged.backup_result,
            )

    # ------------------------------------------------------------------
    # Profile + capacity feedback.
    # ------------------------------------------------------------------

    def _broadcast_sample(self, provider: str, ts: float, result: SingleRequestResult) -> None:
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

    def _account_physical_cost(self, cost: float) -> None:
        with self._cost_lock:
            self._total_cost_usd += cost

    @staticmethod
    def _hedged_cost(hedged: HedgedResult) -> float:
        cost = hedged.primary_result.billed_cost_usd
        if hedged.backup_result is not None:
            cost += hedged.backup_result.billed_cost_usd
        return cost

    @staticmethod
    def _single_physical_cost(result: SingleRequestResult) -> float:
        value = result.physical_cost_usd
        return result.billed_cost_usd if value is None else float(value)

    @classmethod
    def _hedged_physical_cost(cls, hedged: HedgedResult) -> float:
        cost = cls._single_physical_cost(hedged.primary_result)
        if hedged.backup_result is not None:
            cost += cls._single_physical_cost(hedged.backup_result)
        return cost

    def _account_single(
        self,
        policy: BasePolicy,
        result: SingleRequestResult,
        *,
        physical: bool = True,
    ) -> None:
        with self._cost_lock:
            self._cost_per_policy[policy.name] += result.billed_cost_usd
            if physical:
                self._total_cost_usd += self._single_physical_cost(result)

    def _account_cost(
        self,
        policy: BasePolicy,
        hedged: HedgedResult,
        *,
        physical: bool = True,
    ) -> None:
        cost = self._hedged_cost(hedged)
        with self._cost_lock:
            self._cost_per_policy[policy.name] += cost
            if physical:
                self._total_cost_usd += self._hedged_physical_cost(hedged)

    def _cost_exhausted(self, policy_name: str) -> bool:
        with self._cost_lock:
            if self._total_cost_usd >= self.max_cost_usd:
                self._stop_event.set()
                logger.warning("max_cost_usd $%.2f reached; halting", self.max_cost_usd)
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
            slo_ms=self.slo_sec * 1000.0,
            ts=ts,
        )

    def _record_single(
        self,
        policy: BasePolicy,
        req: TraceRequest,
        req_index: int,
        decision: RoutingDecision,
        result: SingleRequestResult,
        *,
        primary_cached_input_tokens: int = 0,
        primary_routing_estimated_cost_usd: float | None = None,
    ) -> None:
        spec = self._spec_by_name.get(decision.primary or "")
        self.recorder.write_request(
            policy=policy.name,
            req_id=f"{req_index}_{uuid.uuid4().hex[:6]}",
            ctx_prompt_tokens=req.prompt_tokens,
            ctx_max_tokens=req.max_tokens,
            decision=decision,
            primary_result=result,
            primary_tier=spec.tier if spec else None,
            final_tier=spec.tier if spec else None,
            slo_ms=self.slo_sec * 1000.0,
            transport=spec.transport_cfg.transport if spec else None,
            primary_cached_input_tokens=primary_cached_input_tokens,
            primary_routing_estimated_cost_usd=primary_routing_estimated_cost_usd,
        )

    def _record_hedged(
        self,
        policy: BasePolicy,
        req: TraceRequest,
        req_index: int,
        decision: RoutingDecision,
        hedged: HedgedResult,
        hedge_delay_sec: float,
        *,
        primary_cached_input_tokens: int = 0,
        backup_cached_input_tokens: int = 0,
        primary_routing_estimated_cost_usd: float | None = None,
        backup_routing_estimated_cost_usd: float | None = None,
    ) -> None:
        primary_spec = self._spec_by_name.get(decision.primary or "")
        backup_spec = self._spec_by_name.get(decision.hedge or "")
        final_spec = backup_spec if hedged.winner == "backup" else primary_spec
        self.recorder.write_hedged(
            policy=policy.name,
            req_id=f"{req_index}_{uuid.uuid4().hex[:6]}",
            ctx_prompt_tokens=req.prompt_tokens,
            ctx_max_tokens=req.max_tokens,
            decision=decision,
            hedged=hedged,
            hedge_delay_sec=hedge_delay_sec,
            primary_tier=primary_spec.tier if primary_spec else None,
            backup_tier=backup_spec.tier if backup_spec else None,
            final_tier=final_spec.tier if final_spec else None,
            slo_ms=self.slo_sec * 1000.0,
            transport=primary_spec.transport_cfg.transport if primary_spec else None,
            primary_cached_input_tokens=primary_cached_input_tokens,
            backup_cached_input_tokens=backup_cached_input_tokens,
            primary_routing_estimated_cost_usd=primary_routing_estimated_cost_usd,
            backup_routing_estimated_cost_usd=backup_routing_estimated_cost_usd,
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
        required=True,
        help="Path to a JSONL trace.",
    )
    parser.add_argument(
        "--policy",
        action="append",
        dest="policies",
        default=None,
        help=(
            "Repeatable. By default each policy sees the full trace at the "
            "same wall-clock arrivals with isolated state."
        ),
    )
    parser.add_argument(
        "--coalesce-identical-actions",
        action="store_true",
        help=(
            "Run every policy on the full trace, but execute identical "
            "physical provider actions once and fan out the observed result "
            "to each policy's virtual accounting."
        ),
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Cap the trace to at most this many rows.",
    )
    parser.add_argument("--speedup", type=float, default=1.0, help="Replay speedup factor.")
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=float("inf"),
        help="Wall-clock cap for the replay phase.",
    )
    parser.add_argument(
        "--warmup-probes",
        type=int,
        default=DEFAULT_WARMUP_PROBES_PER_PROVIDER,
        help="Number of warmup probe rounds per provider before replay.",
    )
    parser.add_argument(
        "--initial-profile-path",
        type=Path,
        default=None,
        help=(
            "Optional JSON latency profile built by scripts/prebuild_profile.py. "
            "Loaded before local warmup probes."
        ),
    )
    parser.add_argument(
        "--warmup-probe-interval-sec",
        type=float,
        default=DEFAULT_WARMUP_PROBE_INTERVAL_SEC,
        help=("Seconds between warmup probe rounds. Use 0 for a fast smoke run."),
    )
    parser.add_argument(
        "--periodic-probe-interval-sec",
        type=float,
        default=DEFAULT_PERIODIC_PROBE_INTERVAL_SEC,
        help=(
            "Seconds between shared profile-maintenance probe rounds during "
            "replay. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--profile-probe-sleep-sec",
        type=float,
        default=DEFAULT_PROFILE_PROBE_SLEEP_SEC,
        help="Sleep between provider probes inside a warmup/periodic round.",
    )
    parser.add_argument(
        "--min-profile-success-samples",
        type=int,
        default=DEFAULT_MIN_PROFILE_SUCCESS_SAMPLES,
        help=(
            "Fail before replay when profile-dependent policies have fewer "
            "successful warmup samples per provider. Use 0 to disable."
        ),
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
        "--prefix-cache-routing",
        action="store_true",
        help=(
            "Use provider-local prefix-cache state only for RouteWise "
            "route-time API cost estimates. Final billed cost still uses "
            "provider-reported usage.cost."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    # Load .env so OPENROUTER_API_KEY etc. don't have to be exported by hand
    # before each run. The transport reads from os.environ, so this must
    # happen before any provider config is built.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

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

    trace_end_sec = (
        args.duration_sec * args.speedup if math.isfinite(args.duration_sec) else float("inf")
    )
    trace = load_trace_jsonl(
        args.trace,
        max_requests=args.max_requests,
        trace_end_sec=trace_end_sec,
    )
    if not trace:
        logger.error("trace is empty; nothing to replay")
        return 2

    trace_span_sec = trace[-1].arrival_time_sec if trace else 0.0
    max_policy_dispatches = len(trace) * len(args.policies)
    logger.info(
        "run plan: %d trace requests over %.1fs trace time; %d policies -> "
        "<= %d policy dispatches; speedup=%.2gx duration_cap=%s",
        len(trace),
        trace_span_sec,
        len(args.policies),
        max_policy_dispatches,
        args.speedup,
        "inf" if not math.isfinite(args.duration_sec) else f"{args.duration_sec:.0f}s",
    )
    logger.info("policies: %s", ", ".join(args.policies))
    logger.info(
        "inventory: %d providers from %s; warmup_probes=%d; "
        "warmup_probe_interval_sec=%.1f; periodic_probe_interval_sec=%.1f; "
        "min_profile_success_samples=%d; max_cost_usd=$%.2f",
        len(inventory.providers),
        args.inventory,
        args.warmup_probes,
        args.warmup_probe_interval_sec,
        args.periodic_probe_interval_sec,
        args.min_profile_success_samples,
        args.max_cost_usd,
    )
    logger.info("output: %s", args.output)
    logger.info("prefix_cache_routing=%s", args.prefix_cache_routing)

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
        prefix_cache_routing=args.prefix_cache_routing,
    )

    if args.initial_profile_path is not None:
        runner.load_initial_profile(args.initial_profile_path)
    if args.warmup_probes > 0:
        runner.warmup(
            probes_per_provider=args.warmup_probes,
            sleep_sec=args.profile_probe_sleep_sec,
            round_interval_sec=args.warmup_probe_interval_sec,
        )
    runner.validate_profile_bootstrap(min_success_samples=args.min_profile_success_samples)

    runner.replay(
        trace=trace,
        speedup=args.speedup,
        duration_sec=args.duration_sec,
        coalesce_identical_actions=args.coalesce_identical_actions,
        periodic_probe_interval_sec=args.periodic_probe_interval_sec,
        periodic_probe_sleep_sec=args.profile_probe_sleep_sec,
    )

    summary_path = runner.finalize()
    logger.info("wrote summary to %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
