"""Trace-replay runner for real online evaluation.

Drives one or more policies through a real trace against live provider APIs.
By default every policy sees the full trace at the same wall-clock arrivals
with isolated state. Each request/policy decision is dispatched in a daemon
thread; per-policy profiles and capacity state are updated by feedback after
each completion.

Implementation notes:

- Per-policy CSV via :class:`Recorder`
- Hedge dispatch uses checkpoint-time probability re-evaluation.
- Sentinel handling for OpenRouter native modes is centralized here
- ``--max-cost-usd`` guardrail per policy and globally
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import logging
import math
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from experiments.real_evaluation.executor import (
    HedgedResult,
    send_checkpoint_hedged_request,
    send_request,
)
from experiments.real_evaluation.inventory import (
    InventoryConfig,
    ProviderSpec,
    load_inventory,
    subscription_fixed_cost_for_inventory,
)
from experiments.real_evaluation.policies import (
    OR_AUTO_SENTINEL,
    OR_SORT_SENTINEL_TO_MODE,
    BasePolicy,
    CapacityUnavailableError,
    RequestContext,
    RoutingDecision,
    build_policy,
    hedge_checkpoints_for_slo,
)
from experiments.real_evaluation.recorder import Recorder
from experiments.real_evaluation.shadow_price import workload_cost_envelope
from experiments.real_evaluation.shared_profile import (
    SharedProfileEvent,
    SharedProfileEventLog,
)
from experiments.real_evaluation.transports import (
    BaseTransport,
    SingleRequestResult,
    TransportConfig,
    build_transport,
)

DEFAULT_TIMEOUT_SEC: int = 60
# Warmup default: 24 rounds at 5s start-to-start cadence ≈ 2 minutes total.
# Each round fires probes against every provider concurrently. Slow providers
# may accumulate more than one in-flight probe across rounds; each transport
# call is still bounded by the request timeout.
DEFAULT_WARMUP_PROBES_PER_PROVIDER: int = 24
DEFAULT_WARMUP_PROBE_INTERVAL_SEC: float = 5.0
DEFAULT_PROFILE_PROBE_SLEEP_SEC: float = 0.5
DEFAULT_PROFILE_PROBE_PARALLELISM: int = 0
DEFAULT_PERIODIC_PROBE_INTERVAL_SEC: float = 180.0
DEFAULT_MIN_PROFILE_SUCCESS_SAMPLES: int = 5
# Probes are exogenous observations: if a request fails we do not retry and
# do not poison the latency profile. Real-request feedback (which is what the
# routing policy actually cares about) keeps using the existing error-sample
# path with ``RATE_LIMIT_ERROR_PENALTY_MS``.
#
# Warmup is the only exception: if a provider keeps failing during warmup we
# inject one ``PROBE_FAILURE_FALLBACK_TTFT_MS`` sample so the LP can still
# tentatively rank the provider instead of leaving it unprofiled forever.
DEFAULT_PROBE_MAX_ATTEMPTS: int = 1
PROBE_FAILURE_FALLBACK_TTFT_MS: float = 10_000.0
WARMUP_PROBE_PROMPT: str = "Write a one-sentence greeting."
SYNTHETIC_PROMPT_INSTRUCTION_TEMPLATE: str = (
    "FINAL REQUEST: Ignore all previous synthetic padding. Do not include analysis, "
    "reasoning, markdown, labels, or <think> tags. Output only a fictional story "
    "of about {output_tokens} tokens. Start the story with: Once upon a time"
)

logger = logging.getLogger(__name__)


@dataclass
class TraceRequest:
    """One row from a trace JSONL."""

    arrival_time_sec: float
    prompt: str
    prompt_tokens: int
    max_tokens: int
    prefix_id: str | None = None
    trace_cached_input_tokens: int | None = None


def load_trace_jsonl(
    path: Path,
    *,
    max_requests: int | None = None,
    time_compression: float = 1.0,
    trace_start_sec: float = 0.0,
    trace_end_sec: float = float("inf"),
    synthesize_missing_prompts: bool = False,
    synthetic_output_tokens: int | None = None,
) -> list[TraceRequest]:
    """Load an arrival-paced trace from a JSONL file.

    Recognized fields (first non-null wins):
        arrival_time_sec : ``arrived_at`` | ``arrival_time_sec`` | ``timestamp``
        prompt           : ``prompt_text`` | ``prompt``
        prompt_tokens    : ``num_prefill_tokens`` | ``prompt_tokens``
        max_tokens       : ``num_decode_tokens`` | ``max_tokens`` | ``completion_tokens``
        prefix_id        : ``prefix_id`` | ``sharegpt_conversation_id`` |
                           ``session_id`` | ``user_id`` | ``user`` (optional)

    ``time_compression`` divides arrival times so a long trace can be
    replayed in less wall-clock time.

    ``synthesize_missing_prompts`` exists for FreeInference-style traces that
    preserve token counts and cache metadata but not request text. Synthetic
    prompts are whitespace-token approximations; route-time cost still uses
    the trace token counts.
    """
    if synthetic_output_tokens is not None and synthetic_output_tokens <= 0:
        raise ValueError("synthetic_output_tokens must be positive when set")

    parsed: list[tuple[float, int, TraceRequest]] = []
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
            arrived_raw = _first_present(rec, ("arrived_at", "arrival_time_sec", "timestamp"))
            if arrived_raw is None:
                raise ValueError(
                    f"{trace_path}:{line_num}: missing arrival timestamp "
                    "(expected arrived_at, arrival_time_sec, or timestamp)"
                )
            arrived = _coerce_arrival_timestamp(
                trace_path,
                line_num,
                arrived_raw,
            )
            first_ts = arrived if first_ts is None else min(first_ts, arrived)

            status_code = _optional_int(rec.get("status_code"))
            if status_code is not None and status_code >= 400:
                continue

            max_tokens_raw = _first_present(
                rec,
                ("num_decode_tokens", "max_tokens", "completion_tokens"),
            )
            if max_tokens_raw is None:
                raise ValueError(
                    f"{trace_path}:{line_num}: missing output token cap "
                    "(expected num_decode_tokens, max_tokens, or completion_tokens)"
                )
            max_tokens = _coerce_int(trace_path, line_num, "output token cap", max_tokens_raw)
            if max_tokens <= 0:
                skipped_nonpositive_output_cap += 1
                continue
            prompt = _first_present(rec, ("prompt_text", "prompt"))
            if not isinstance(prompt, str) or not prompt.strip():
                if not synthesize_missing_prompts:
                    raise ValueError(
                        f"{trace_path}:{line_num}: missing non-empty prompt "
                        "(expected prompt_text or prompt)"
                    )
                prompt = ""
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
            if not prompt:
                output_tokens = synthetic_output_tokens or max_tokens
                prompt = _synthesize_prompt_text(
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                )
                max_tokens = output_tokens
            parsed.append(
                (
                    arrived,
                    line_num,
                    TraceRequest(
                        arrival_time_sec=0.0,
                        prompt=prompt,
                        prompt_tokens=prompt_tokens,
                        max_tokens=max_tokens,
                        prefix_id=_prefix_id_from_record(rec),
                        trace_cached_input_tokens=_trace_cached_input_tokens_from_record(rec),
                    ),
                )
            )
    if skipped_nonpositive_output_cap:
        logger.warning(
            "Skipped %d trace rows from %s because output token cap was <= 0",
            skipped_nonpositive_output_cap,
            trace_path,
        )

    out: list[TraceRequest] = []
    if first_ts is None:
        return out
    for arrived, _, request in sorted(parsed, key=lambda item: (item[0], item[1])):
        relative = (arrived - first_ts) / max(time_compression, 1e-6)
        if relative < trace_start_sec:
            continue
        if relative > trace_end_sec:
            break
        out.append(dataclasses.replace(request, arrival_time_sec=relative))
        if max_requests is not None and len(out) >= max_requests:
            break
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
    value = _first_present(
        rec,
        ("prefix_id", "sharegpt_conversation_id", "session_id", "user_id", "user"),
    )
    if value is None:
        return None
    prefix_id = str(value).strip()
    return prefix_id or None


def _synthesize_prompt_text(*, prompt_tokens: int, output_tokens: int) -> str:
    """Build a length-matched synthetic prompt for traces without text.

    This is intentionally simple and dependency-free: one repeated whitespace
    word roughly maps to one BPE-ish token for the providers we probe. The
    final instruction is last so it dominates the synthetic padding. Keep the
    filler to one-token-ish ASCII words; multi-piece words caused large
    FreeInference rows to overflow provider context limits.
    """
    instruction = SYNTHETIC_PROMPT_INSTRUCTION_TEMPLATE.format(output_tokens=output_tokens)
    instruction_words = instruction.split()
    target_filler_words = max(int(prompt_tokens) - len(instruction_words), 0)
    filler_words = ["a"] * target_filler_words
    return " ".join([*filler_words, *instruction_words]).strip()


def _trace_cached_input_tokens_from_record(rec: dict[str, Any]) -> int | None:
    """Return trace-reported cached-input token count, or ``None`` if absent.

    Accepts ``cache_read_tokens`` (the freeinference field name) and
    ``cached_input_tokens`` for parity with the simulator. Missing or empty
    values yield ``None`` so the caller treats the request as a cold miss.
    """
    value = _first_present(rec, ("cache_read_tokens", "cached_input_tokens"))
    if value is None or value == "":
        return None
    try:
        return max(int(float(value)), 0)
    except (TypeError, ValueError):
        return None


def _coerce_arrival_timestamp(path: Path, line_num: int, value: Any) -> float:
    if isinstance(value, int | float):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}:{line_num}: arrival timestamp must be non-empty")
    timestamp = value.strip()
    try:
        return float(timestamp)
    except ValueError:
        pass
    if timestamp.endswith("Z"):
        timestamp = f"{timestamp[:-1]}+00:00"
    try:
        return datetime.fromisoformat(timestamp).timestamp()
    except ValueError as exc:
        raise ValueError(
            f"{path}:{line_num}: arrival timestamp must be numeric or ISO-8601, "
            f"got {value!r}"
        ) from exc


def _coerce_int(path: Path, line_num: int, field: str, value: Any) -> int:
    try:
        as_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{line_num}: {field} must be an integer, got {value!r}") from exc
    if not math.isfinite(as_float) or not as_float.is_integer():
        raise ValueError(f"{path}:{line_num}: {field} must be an integer, got {value!r}")
    return int(as_float)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not parsed.is_integer():
        return None
    return int(parsed)


def _inventory_has_quota_providers(inventory: InventoryConfig) -> bool:
    """Return True if any provider in ``inventory`` rolls a quota window."""
    for spec in inventory.providers:
        if spec.tier == "quota":
            return True
        if spec.quota_requests is not None and spec.quota_window_sec is not None:
            return True
        if spec.quota_windows:
            return True
    return False


def check_quota_clock_alignment(
    inventory: InventoryConfig,
    *,
    speedup: float,
    allow_mismatch: bool = False,
) -> None:
    """Block non-unit ``--speedup`` on quota-bearing inventories.

    Real-eval quota windows roll on wall-clock (``time.time()``) because the
    real subscription provider does the same. Replaying a trace at
    ``speedup != 1.0`` keeps the wall-clock window length unchanged while
    compressing the trace's logical day, which silently lets the policy
    spend more or fewer free quota requests than the paper's
    ``requests_per_day`` semantics would allow. The mismatch never raises an
    HTTP error, so it is impossible to notice from the live logs alone.

    Pass ``allow_mismatch=True`` (CLI ``--allow-quota-clock-mismatch``) to
    knowingly run an ablation that breaks this alignment.
    """
    if allow_mismatch:
        return
    if not _inventory_has_quota_providers(inventory):
        return
    if math.isclose(speedup, 1.0, rel_tol=0.0, abs_tol=1e-9):
        return
    raise SystemExit(
        "Inventory contains quota-bearing providers but --speedup is "
        f"{speedup}. Quota windows roll on wall-clock; running real-eval "
        "at a non-unit speedup silently breaks the paper's "
        "requests-per-day subscription semantics. Re-run with "
        "--speedup=1.0, or pass --allow-quota-clock-mismatch if you "
        "really want this."
    )


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
    hedge_checkpoints_sec: tuple[float, ...] = ()
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
        shared_profile_events_path: Path | None = None,
        shared_profile_poll_sec: float = 1.0,
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
        self._shared_profile_log = (
            SharedProfileEventLog(shared_profile_events_path)
            if shared_profile_events_path is not None
            else None
        )
        self._shared_profile_poll_sec = max(0.1, float(shared_profile_poll_sec))

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
        # Tracks providers whose API key env var is unset so the periodic
        # prober can skip them after warning once. Populated lazily by
        # ``_provider_api_key_missing``.
        self._providers_missing_api_key: set[str] = set()
        # ``threading.local`` must live on the runner instance so each
        # worker thread sees a stable storage; a fresh ``threading.local()``
        # in ``_session()`` would defeat the per-thread reuse contract.
        self._thread_local = threading.local()

    # ------------------------------------------------------------------
    # Cost envelope plumbing.
    # ------------------------------------------------------------------

    def apply_cost_envelope(self, envelope: tuple[float, float] | None) -> None:
        """Push a workload-level ``(L, U)`` envelope to all policies.

        Mirrors the simulator path where the envelope is calibrated once from
        the workload's cheapest-API cost distribution and shared across
        policies for the whole run. Real-eval policies fail fast if this is not
        installed; there is intentionally no single-request fallback.
        """
        for policy in self.policies.values():
            policy.set_cost_envelope(envelope)

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
        # OR auto/sort sentinels are API-only baselines. The inventory's
        # ``openrouter_provider_only`` field also has to include subscription
        # endpoints (e.g. Chutes via OpenRouter) so the joint pool can survive
        # ``_skip_openrouter_provider_entry``; if we forwarded that full list
        # to the sentinel here, OR's server-side routing would pick subscription
        # tiers as candidates and pollute the API-only baseline. Derive the
        # sentinel allowlist from API-tier OR providers in the inventory
        # instead, so the baseline always faces exactly the eight (or N)
        # metered API providers the experiment is comparing against.
        sentinel_allowlist = tuple(
            sorted(
                {
                    spec.transport_cfg.provider_hint
                    for spec in self.inventory.providers
                    if spec.tier == "api"
                    and spec.transport_cfg.transport == "openrouter"
                    and spec.transport_cfg.provider_hint
                }
            )
        )
        provider_only = sentinel_allowlist or base.provider_only
        cfg = dataclasses.replace(
            base,
            name=sentinel,
            provider_hint=None,
            sort_mode=sort_mode,
            provider_only=provider_only,
            provider_ignore=(self.inventory.openrouter_provider_ignore or base.provider_ignore),
            stream_cancel_billing="continues",
            stream_cancel_billing_by_provider=dict(
                self.inventory.openrouter_stream_cancel_billing_by_provider
            ),
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
        parallelism: int = DEFAULT_PROFILE_PROBE_PARALLELISM,
    ) -> None:
        """Probe every provider and broadcast observations to all policies.

        ``round_interval_sec`` is **start-to-start** cadence: round ``N+1``
        begins ``round_interval_sec`` after round ``N`` started, regardless of
        how long round ``N``'s probes take. The cadenced path fires a probe at
        every provider each round, so slow providers may have multiple
        in-flight probes.

        When ``round_interval_sec <= 0`` or ``probes_per_provider == 1`` the
        loop falls back to the simple synchronous path used by periodic /
        single-shot calls.

        Providers whose API key env var is unset in this process get skipped
        on first encounter (with a one-time warning) so e.g. OR-only
        baselines launched without CHUTES_API_KEY / FEATHERLESS_API_KEY don't
        crash the prober — those policies never dispatch real traffic to the
        skipped providers anyway.
        """
        if probes_per_provider <= 0:
            return
        if probes_per_provider == 1 or round_interval_sec <= 0:
            self._probe_profiles_synchronous(
                probes_per_provider=probes_per_provider,
                sleep_sec=sleep_sec,
                round_interval_sec=round_interval_sec,
                phase=phase,
                stop_event=stop_event,
                parallelism=parallelism,
            )
            return
        self._probe_profiles_cadenced(
            probes_per_provider=probes_per_provider,
            sleep_sec=sleep_sec,
            round_interval_sec=round_interval_sec,
            phase=phase,
            stop_event=stop_event,
        )

    def _probe_profiles_synchronous(
        self,
        *,
        probes_per_provider: int,
        sleep_sec: float,
        round_interval_sec: float,
        phase: str,
        stop_event: threading.Event | None,
        parallelism: int,
    ) -> None:
        for round_idx in range(probes_per_provider):
            if self._probe_stop_requested(stop_event):
                return
            if self._cost_exhausted("profile_maintenance"):
                return
            specs = [
                spec
                for spec in self.inventory.providers
                if spec.name not in self._providers_missing_api_key
            ]
            if not specs:
                return
            self._probe_profile_round_parallel(
                specs,
                round_idx=round_idx,
                probes_per_provider=probes_per_provider,
                retry_sleep_sec=sleep_sec,
                phase=phase,
                stop_event=stop_event,
                parallelism=parallelism,
            )
            if self._probe_stop_requested(stop_event) or self._stop_event.is_set():
                return
            if round_idx + 1 < probes_per_provider and round_interval_sec > 0:
                logger.info(
                    "%s probe round %d/%d complete; sleeping %.1fs",
                    phase,
                    round_idx + 1,
                    probes_per_provider,
                    round_interval_sec,
                )
                if self._probe_sleep(round_interval_sec, stop_event):
                    return

    def _probe_profiles_cadenced(
        self,
        *,
        probes_per_provider: int,
        sleep_sec: float,
        round_interval_sec: float,
        phase: str,
        stop_event: threading.Event | None,
    ) -> None:
        """Start-to-start cadence: every ``round_interval_sec`` fire one
        probe at every provider.

        A provider with ``concurrency_limit`` set (e.g. Featherless's
        subscription tier, where the account holds only N concurrent slots)
        gets skip-in-flight: if its previous probe is still cooking when
        the next tick arrives, this round skips it. Otherwise our probe
        would occupy the only slot and force real production traffic to
        429 — which *does* poison the profile via ``error_samples``.

        Providers without a concurrency cap (OpenRouter pinned endpoints,
        native quota tiers) get a probe on every tick regardless of
        in-flight state. Probe failures are silent in non-warmup phases, so
        accumulating in-flight probes against a slow provider has no
        profile-pollution downside and a late-returning probe is still a
        useful TTFT observation.
        """
        all_specs = list(self.inventory.providers)
        if not all_specs:
            return
        in_flight: dict[str, Future[SingleRequestResult | None]] = {}
        outstanding: list[Future[SingleRequestResult | None]] = []
        target_round_start = time.time()
        # Size the pool generously so submits never queue: worst case every
        # round dispatches against every provider while none have returned.
        max_workers = max(1, len(all_specs) * probes_per_provider)
        submitted_total = 0
        skipped_total = 0
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"real-eval-{phase}-cadenced",
        ) as pool:
            for round_idx in range(probes_per_provider):
                if self._probe_stop_requested(stop_event):
                    break
                if self._cost_exhausted("profile_maintenance"):
                    break
                # Reap finished concurrency-limited slots so they're eligible
                # again this round.
                for name in [n for n, fut in in_flight.items() if fut.done()]:
                    in_flight.pop(name, None)
                eligible: list[ProviderSpec] = []
                skipped_this_round = 0
                for spec in all_specs:
                    if spec.name in self._providers_missing_api_key:
                        continue
                    if (
                        spec.concurrency_limit is not None
                        and spec.name in in_flight
                    ):
                        skipped_this_round += 1
                        continue
                    eligible.append(spec)
                skipped_total += skipped_this_round
                if eligible:
                    submitted_total += len(eligible)
                    logger.info(
                        "%s probe round %d/%d submit=%d skip_in_flight=%d",
                        phase,
                        round_idx + 1,
                        probes_per_provider,
                        len(eligible),
                        skipped_this_round,
                    )
                    for spec in eligible:
                        fut = pool.submit(
                            self._probe_provider,
                            spec,
                            phase=phase,
                            round_idx=round_idx,
                            probes_per_provider=probes_per_provider,
                            retry_sleep_sec=sleep_sec,
                            stop_event=stop_event,
                            max_attempts=DEFAULT_PROBE_MAX_ATTEMPTS,
                        )
                        outstanding.append(fut)
                        if spec.concurrency_limit is not None:
                            in_flight[spec.name] = fut
                else:
                    logger.info(
                        "%s probe round %d/%d submit=0 skip_in_flight=%d "
                        "(all eligible providers concurrency-locked)",
                        phase,
                        round_idx + 1,
                        probes_per_provider,
                        skipped_this_round,
                    )
                # start-to-start cadence: schedule next round at fixed
                # wall-clock offset, regardless of submit overhead.
                if round_idx + 1 < probes_per_provider:
                    target_round_start += round_interval_sec
                    sleep_for = target_round_start - time.time()
                    if sleep_for > 0 and self._probe_sleep(sleep_for, stop_event):
                        break
            # Wait for outstanding probes so every dispatched sample makes
            # it into the profile before this method returns. Don't honor
            # stop here — daemon worker threads would otherwise never
            # report back.
            for fut in outstanding:
                with contextlib.suppress(Exception):
                    fut.result()
        logger.info(
            "%s cadenced warmup done: submitted=%d skipped_in_flight=%d",
            phase,
            submitted_total,
            skipped_total,
        )

    def _probe_profile_round_parallel(
        self,
        specs: list[ProviderSpec],
        *,
        round_idx: int,
        probes_per_provider: int,
        retry_sleep_sec: float,
        phase: str,
        stop_event: threading.Event | None,
        parallelism: int = DEFAULT_PROFILE_PROBE_PARALLELISM,
    ) -> None:
        max_workers = len(specs) if parallelism <= 0 else min(parallelism, len(specs))
        if max_workers <= 1:
            for spec in specs:
                if self._probe_stop_requested(stop_event):
                    return
                if self._cost_exhausted("profile_maintenance"):
                    return
                if spec.name in self._providers_missing_api_key:
                    continue
                self._probe_provider(
                    spec,
                    phase=phase,
                    round_idx=round_idx,
                    probes_per_provider=probes_per_provider,
                    retry_sleep_sec=retry_sleep_sec,
                    stop_event=stop_event,
                    max_attempts=DEFAULT_PROBE_MAX_ATTEMPTS,
                )
            return

        logger.info(
            "%s probe round %d/%d starting with parallelism=%d providers=%d",
            phase,
            round_idx + 1,
            probes_per_provider,
            max_workers,
            len(specs),
        )
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"real-eval-{phase}-probe",
        ) as pool:
            futures = {
                pool.submit(
                    self._probe_provider,
                    spec,
                    phase=phase,
                    round_idx=round_idx,
                    probes_per_provider=probes_per_provider,
                    retry_sleep_sec=retry_sleep_sec,
                    stop_event=stop_event,
                    max_attempts=DEFAULT_PROBE_MAX_ATTEMPTS,
                ): spec
                for spec in specs
            }
            for future in as_completed(futures):
                if self._probe_stop_requested(stop_event):
                    for pending in futures:
                        pending.cancel()
                    return
                try:
                    future.result()
                except Exception:
                    spec = futures[future]
                    logger.warning(
                        "%s probe round %d/%d %s worker failed",
                        phase,
                        round_idx + 1,
                        probes_per_provider,
                        spec.name,
                        exc_info=True,
                    )

    def _probe_provider(
        self,
        spec: ProviderSpec,
        *,
        phase: str,
        round_idx: int,
        probes_per_provider: int,
        retry_sleep_sec: float,
        stop_event: threading.Event | None,
        max_attempts: int = DEFAULT_PROBE_MAX_ATTEMPTS,
    ) -> SingleRequestResult | None:
        """Probe one provider with bounded retries.

        Returns the successful ``SingleRequestResult`` if the probe succeeded
        within ``max_attempts``. On exhaustion (rate-limit, timeout, etc.) the
        provider is treated as currently degraded: we broadcast a synthetic
        ``PROBE_FAILURE_FALLBACK_TTFT_MS`` sample to every policy profile so
        the LP keeps a bounded, pessimistic estimate (rather than dropping
        the provider entirely as the rolling window expires), and return
        ``None`` for the round bookkeeping.

        ``None`` still means "stop the round" only when paired with a hard
        stop (``stop_event`` / cost-exhausted / missing-API-key). Failure
        exhaustion does not stop the round — callers should keep probing
        the remaining providers.
        """
        attempt = 0
        while attempt < max_attempts:
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
                    # Mark this provider as permanently skipped for the
                    # rest of this process. The launcher only injects
                    # API keys for providers a given policy actually
                    # dispatches to; bailing the whole run for a key
                    # that policy never touches would be wrong.
                    if spec.name not in self._providers_missing_api_key:
                        self._providers_missing_api_key.add(spec.name)
                        logger.warning(
                            "profile prober skipping provider %s: %s",
                            spec.name,
                            exc,
                        )
                    return None
                logger.warning(
                    "%s probe round %d/%d %s attempt %d/%d raised %s",
                    phase,
                    round_idx + 1,
                    probes_per_provider,
                    spec.name,
                    attempt,
                    max_attempts,
                    exc,
                )
                if attempt < max_attempts and retry_sleep_sec > 0 and self._probe_sleep(
                    retry_sleep_sec, stop_event
                ):
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

            detail = (result.error_message or "").strip() or "(no error body)"
            if len(detail) > 240:
                detail = detail[:237] + "..."
            logger.warning(
                (
                    "%s probe round %d/%d %s attempt %d/%d failed: %s "
                    "ttft=%.1fms billed=$%.5f physical=$%.5f detail=%r"
                ),
                phase,
                round_idx + 1,
                probes_per_provider,
                spec.name,
                attempt,
                max_attempts,
                result.status,
                result.ttft_ms,
                result.billed_cost_usd,
                self._single_physical_cost(result),
                detail,
            )
            if attempt < max_attempts and retry_sleep_sec > 0 and self._probe_sleep(
                retry_sleep_sec, stop_event
            ):
                return None

        # All attempts exhausted. Warmup is the only phase that injects a
        # synthetic sample, so the LP can still tentatively rank a provider
        # that failed during startup. Replay-time probes (periodic / shared
        # sidecar) stay silent: probe failures must not poison the routing
        # profile. Real-request 429s use the ``error_samples`` path with
        # ``RATE_LIMIT_ERROR_PENALTY_MS`` instead.
        if phase == "warmup":
            synthetic_ts = time.time()
            self._broadcast_synthetic_probe_sample(
                spec.name,
                synthetic_ts,
                ttft_ms=PROBE_FAILURE_FALLBACK_TTFT_MS,
            )
            logger.warning(
                (
                    "warmup probe round %d/%d %s exhausted %d attempt(s); "
                    "recording synthetic ttft=%.0fms sample"
                ),
                round_idx + 1,
                probes_per_provider,
                spec.name,
                max_attempts,
                PROBE_FAILURE_FALLBACK_TTFT_MS,
            )
        else:
            logger.warning(
                "%s probe round %d/%d %s exhausted %d attempt(s); not "
                "recording a sample",
                phase,
                round_idx + 1,
                probes_per_provider,
                spec.name,
                max_attempts,
            )
        return None

    def _broadcast_synthetic_probe_sample(
        self,
        provider: str,
        ts: float,
        *,
        ttft_ms: float,
    ) -> None:
        """Feed a synthetic positive-TTFT sample into every policy's profile.

        Unlike :meth:`_broadcast_sample`, this does not require a real
        ``SingleRequestResult``; it writes a clamped success-shaped sample
        used when probe retries are exhausted.
        """
        if self._publish_shared_profile_sample(
            provider,
            ts,
            float(ttft_ms),
            None,
            source="probe_synthetic",
        ):
            return
        for policy in self.policies.values():
            policy.add_sample(provider, ts, float(ttft_ms), None)

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
        parallelism: int = DEFAULT_PROFILE_PROBE_PARALLELISM,
    ) -> None:
        """Seed all policy-local latency profiles before replay."""
        self.probe_profiles(
            probes_per_provider=probes_per_provider,
            sleep_sec=sleep_sec,
            round_interval_sec=round_interval_sec,
            phase="warmup",
            parallelism=parallelism,
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
        meta = raw.get("_meta")
        seed_offset = (
            meta.get("shared_profile_seed_offset")
            if isinstance(meta, dict)
            else None
        )
        if self._shared_profile_log is not None and seed_offset is not None:
            self._shared_profile_log.set_read_offset(int(seed_offset))
            logger.info(
                "shared profile event log starts after warmup seed offset=%d",
                int(seed_offset),
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
        periodic_probe_interval_sec: float = 0.0,
        periodic_probe_sleep_sec: float = DEFAULT_PROFILE_PROBE_SLEEP_SEC,
    ) -> None:
        """Replay a trace against the configured policies.

        Every policy gets the full trace at the same wall-clock request
        arrivals with isolated policy state.
        """
        if not trace:
            logger.warning("replay called with empty trace")
            return
        shared_profile_handle = self._start_shared_profile_tailer_thread()
        probe_handle = self._start_periodic_profile_probe_thread(
            interval_sec=periodic_probe_interval_sec,
            sleep_sec=periodic_probe_sleep_sec,
        )
        try:
            self._replay_parallel_full_trace(trace, speedup=speedup, duration_sec=duration_sec)
        finally:
            self._stop_periodic_profile_probe_thread(probe_handle)
            self._stop_periodic_profile_probe_thread(shared_profile_handle)

    def _start_shared_profile_tailer_thread(self) -> _PeriodicProbeHandle | None:
        if self._shared_profile_log is None:
            return None

        stop_event = threading.Event()

        def _drain_once() -> None:
            for event in self._shared_profile_log.read_new():
                self._apply_shared_profile_event(event)

        def _loop() -> None:
            logger.info(
                "shared profile tailing enabled: path=%s poll=%.2fs",
                self._shared_profile_log.path,
                self._shared_profile_poll_sec,
            )
            while not stop_event.is_set():
                _drain_once()
                if stop_event.wait(self._shared_profile_poll_sec):
                    return

        _drain_once()
        thread = threading.Thread(
            target=_loop,
            name="real-eval-shared-profile-tailer",
            daemon=True,
        )
        thread.start()
        return _PeriodicProbeHandle(thread=thread, stop_event=stop_event)

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
            trace_cached_input_tokens=req.trace_cached_input_tokens,
        )
        prompt = req.prompt
        expected_service_sec = max(0.5, req.max_tokens / 40.0 if req.max_tokens else 5.0)

        backup: str | None = None
        hedge_delay_sec = float("inf")
        hedge_checkpoints_sec: tuple[float, ...] = ()
        if policy.use_hedge:
            hedge_checkpoints_sec = hedge_checkpoints_for_slo(self.slo_sec)

        now = time.time()
        try:
            decision, primary_capacity_id = policy.route_and_charge_capacity(
                now,
                ctx,
                expected_service_sec,
            )
        except CapacityUnavailableError as exc:
            self._record_no_route(
                policy,
                req,
                req_index,
                RoutingDecision(
                    primary=None,
                    notes=f"capacity_unavailable:{exc.provider}",
                ),
                time.time(),
            )
            return None

        if decision.primary is None:
            self._record_no_route(policy, req, req_index, decision, now)
            return None

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
            hedge_checkpoints_sec=hedge_checkpoints_sec,
            primary_capacity_id=primary_capacity_id,
            primary_cached_input_tokens=primary_cached,
            backup_cached_input_tokens=backup_cached,
            primary_routing_estimated_cost_usd=primary_estimated_cost,
            backup_routing_estimated_cost_usd=backup_estimated_cost,
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
            provider, capacity_id = prepared.policy.fallback_candidate_and_charge_capacity(
                now=time.time(),
                ctx=prepared.ctx,
                excluded=excluded,
                expected_service_sec=prepared.expected_service_sec,
            )
            if provider is None:
                break

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

    def _observe_completion(
        self,
        policy: BasePolicy,
        prepared: _PreparedDispatch,
        result: SingleRequestResult,
    ) -> None:
        """Update the policy's output-token predictor with a realized response."""
        if result.status != "success":
            return
        policy.observe_response(prepared.ctx, result.completion_tokens)

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

        if prepared.hedge_checkpoints_sec:
            backup_capacity_id: int | None = None
            backup_capacity_lock = threading.Lock()

            def _checkpoint_backup(elapsed_sec: float, checkpoint_ts: float) -> str | None:
                nonlocal backup_capacity_id
                future_checkpoints = tuple(
                    checkpoint
                    for checkpoint in prepared.hedge_checkpoints_sec
                    if checkpoint > elapsed_sec + 1e-9
                )
                try:
                    (
                        checkpoint_decision,
                        capacity_id,
                    ) = policy.checkpoint_backup_and_charge_capacity(
                        primary=decision.primary or "",
                        ctx=prepared.ctx,
                        slo_sec=self.slo_sec,
                        now=checkpoint_ts,
                        elapsed_sec=elapsed_sec,
                        future_checkpoints_sec=future_checkpoints,
                        expected_service_sec=prepared.expected_service_sec,
                        cost_fn=lambda state: policy.request_cost_for_state(
                            state,
                            prepared.ctx,
                        ),
                    )
                except CapacityUnavailableError:
                    return None
                if checkpoint_decision.backup is None:
                    return None

                backup = checkpoint_decision.backup
                prepared.backup = backup
                prepared.hedge_delay_sec = elapsed_sec
                decision.hedge = backup
                decision.hedge_delay_sec = elapsed_sec
                (
                    prepared.backup_cached_input_tokens,
                    prepared.backup_routing_estimated_cost_usd,
                ) = policy.routing_cache_diagnostics(backup, prepared.ctx)
                with backup_capacity_lock:
                    backup_capacity_id = capacity_id
                return backup

            try:
                hedged = send_checkpoint_hedged_request(
                    send_fn=self._send_via_transport,
                    primary_provider=decision.primary or "",
                    hedge_checkpoints_sec=prepared.hedge_checkpoints_sec,
                    checkpoint_fn=_checkpoint_backup,
                    prompt=prepared.prompt,
                    max_tokens=req.max_tokens,
                    timeout=self.timeout_sec,
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

            if hedged.backup_provider is not None and prepared.backup is None:
                prepared.backup = hedged.backup_provider
                decision.hedge = hedged.backup_provider
            if hedged.hedge_delay_sec is not None:
                prepared.hedge_delay_sec = hedged.hedge_delay_sec
                decision.hedge_delay_sec = hedged.hedge_delay_sec

            hedged_attempts = [(decision.primary or "", hedged.primary_result)]
            if hedged.backup_result is not None and prepared.backup is not None:
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
                    self._observe_completion(policy, prepared, result)
                    primary_cached, primary_estimated_cost = (
                        prepared.primary_cached_input_tokens,
                        prepared.primary_routing_estimated_cost_usd,
                    )
                    if final_provider != decision.primary:
                        primary_cached, primary_estimated_cost = policy.routing_cache_diagnostics(
                            final_provider,
                            prepared.ctx,
                        )
                    self._record_single(
                        policy,
                        req,
                        prepared.req_index,
                        decision,
                        result,
                        final_provider=final_provider,
                        primary_cached_input_tokens=primary_cached,
                        primary_routing_estimated_cost_usd=primary_estimated_cost,
                    )
                    return
            self._feed_back_hedged(policy, hedged)
            self._account_cost(policy, hedged)
            self._observe_completion(policy, prepared, hedged.chosen_result)
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
        self._observe_completion(policy, prepared, result)
        primary_cached, primary_estimated_cost = (
            prepared.primary_cached_input_tokens,
            (prepared.primary_routing_estimated_cost_usd),
        )
        if final_provider != decision.primary:
            primary_cached, primary_estimated_cost = policy.routing_cache_diagnostics(
                final_provider,
                prepared.ctx,
            )
        self._record_single(
            policy,
            req,
            prepared.req_index,
            decision,
            result,
            final_provider=final_provider,
            primary_cached_input_tokens=primary_cached,
            primary_routing_estimated_cost_usd=primary_estimated_cost,
        )

    # ------------------------------------------------------------------
    # Profile + capacity feedback.
    # ------------------------------------------------------------------

    @staticmethod
    def _profile_sample_ts(
        result: SingleRequestResult,
        *,
        fallback_ts: float | None = None,
    ) -> float:
        if result.status == "success" and result.first_token_ts is not None:
            return float(result.first_token_ts)
        if result.start_ts:
            return float(result.start_ts)
        if fallback_ts is not None:
            return float(fallback_ts)
        return time.time()

    def _apply_shared_profile_event(self, event: SharedProfileEvent) -> None:
        for policy in self.policies.values():
            policy.add_sample(
                event.provider,
                event.ts,
                event.ttft_ms,
                event.error_type,
            )

    def _publish_shared_profile_sample(
        self,
        provider: str,
        ts: float,
        ttft_ms: float,
        error_type: str | None,
        *,
        source: str,
        policy: str | None = None,
    ) -> bool:
        if self._shared_profile_log is None:
            return False
        try:
            event = self._shared_profile_log.append(
                provider=provider,
                ts=ts,
                ttft_ms=ttft_ms,
                error_type=error_type,
                source=source,
                policy=policy,
            )
        except Exception as exc:
            logger.warning("failed to publish shared profile sample: %s", exc)
            return False
        # Apply immediately in this process and mark the event as already
        # consumed by ``append`` so the tailer does not duplicate the local
        # write.
        self._apply_shared_profile_event(event)
        return True

    def _broadcast_sample(self, provider: str, ts: float, result: SingleRequestResult) -> None:
        error_type = None if result.status == "success" else result.status
        ttft_ms = result.ttft_ms if result.status == "success" else -1.0
        sample_ts = self._profile_sample_ts(result, fallback_ts=ts)
        if self._publish_shared_profile_sample(
            provider,
            sample_ts,
            ttft_ms,
            error_type,
            source="probe",
        ):
            return
        for policy in self.policies.values():
            policy.add_sample(provider, sample_ts, ttft_ms, error_type)

    def _feed_back_single(
        self, policy: BasePolicy, provider: str, result: SingleRequestResult
    ) -> None:
        error_type = None if result.status == "success" else result.status
        ttft_ms = result.ttft_ms if result.status == "success" else -1.0
        sample_ts = self._profile_sample_ts(result)
        if self._publish_shared_profile_sample(
            provider,
            sample_ts,
            ttft_ms,
            error_type,
            source="natural",
            policy=policy.name,
        ):
            return
        policy.add_sample(provider, sample_ts, ttft_ms, error_type)

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
        final_provider: str | None = None,
        primary_cached_input_tokens: int = 0,
        primary_routing_estimated_cost_usd: float | None = None,
    ) -> None:
        spec = self._spec_by_name.get(decision.primary or "")
        # ``final_provider`` differs from ``decision.primary`` whenever the
        # rate-limit fallback rerouted the request to a different provider.
        # Recording the originally-decided spec's tier here would mislabel
        # those rows (e.g. a Chutes_SQ -> OR_DeepInfra fallback would be
        # logged as ``tier=quota`` even though OR_DeepInfra is an API
        # provider), which throws off downstream tier-mix analysis.
        final_spec = self._spec_by_name.get(final_provider or "") or spec
        self.recorder.write_request(
            policy=policy.name,
            req_id=f"{req_index}_{uuid.uuid4().hex[:6]}",
            ctx_prompt_tokens=req.prompt_tokens,
            ctx_max_tokens=req.max_tokens,
            decision=decision,
            primary_result=result,
            primary_tier=spec.tier if spec else None,
            final_tier=final_spec.tier if final_spec else None,
            slo_ms=self.slo_sec * 1000.0,
            transport=(final_spec or spec).transport_cfg.transport
            if (final_spec or spec)
            else None,
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

    def finalize(self, *, billing_duration_sec: float | None = None) -> Path:
        """Flush recorder and write summary JSON."""
        path = self.recorder.write_summary(
            slo_ms=self.slo_ms,
            fixed_cost_usd_by_policy=self._summary_fixed_costs_by_policy(
                billing_duration_sec=billing_duration_sec,
            ),
        )
        self.recorder.close()
        return path

    def _summary_fixed_costs_by_policy(
        self,
        *,
        billing_duration_sec: float | None,
    ) -> dict[str, float]:
        if billing_duration_sec is None or billing_duration_sec <= 0.0:
            return {}
        fixed_cost = subscription_fixed_cost_for_inventory(
            self.inventory,
            billing_duration_sec=billing_duration_sec,
        )
        if fixed_cost <= 0.0:
            return {}
        return {
            policy_name: 0.0 if policy_name.startswith("or_") else fixed_cost
            for policy_name in self.policies
        }


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
        "--max-requests",
        type=int,
        default=None,
        help="Cap the trace to at most this many rows.",
    )
    parser.add_argument("--speedup", type=float, default=1.0, help="Replay speedup factor.")
    parser.add_argument(
        "--allow-quota-clock-mismatch",
        action="store_true",
        help=(
            "Skip the startup check that blocks non-unit --speedup with "
            "quota-bearing inventories. Quota windows roll on wall-clock; "
            "running at speedup != 1 silently breaks the paper's "
            "requests-per-day subscription semantics."
        ),
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
        "--shared-profile-events",
        type=Path,
        default=None,
        help=(
            "Append/read latency samples from a cross-process JSONL event log. "
            "When set, natural request feedback and probe samples are shared "
            "with other policy processes using the same path."
        ),
    )
    parser.add_argument(
        "--shared-profile-poll-sec",
        type=float,
        default=1.0,
        help="Polling interval for --shared-profile-events.",
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
        "--synthesize-missing-prompts",
        action="store_true",
        help=(
            "For traces that contain token counts but no request text, build "
            "an approximate-length synthetic prompt ending in a story-writing "
            "instruction. Existing prompt_text/prompt fields are left unchanged."
        ),
    )
    parser.add_argument(
        "--synthetic-output-tokens",
        type=int,
        default=None,
        help=(
            "When --synthesize-missing-prompts is used, override max_tokens "
            "for synthesized rows. For example, use 256 for fixed-length "
            "story-generation probes."
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
    check_quota_clock_alignment(
        inventory,
        speedup=args.speedup,
        allow_mismatch=args.allow_quota_clock_mismatch,
    )
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
        synthesize_missing_prompts=args.synthesize_missing_prompts,
        synthetic_output_tokens=args.synthetic_output_tokens,
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
        shared_profile_events_path=args.shared_profile_events,
        shared_profile_poll_sec=args.shared_profile_poll_sec,
    )

    cost_envelope = workload_cost_envelope(
        inventory.providers,
        [req.prompt_tokens for req in trace],
        [req.max_tokens for req in trace],
    )
    runner.apply_cost_envelope(cost_envelope)
    logger.info(
        "workload cost envelope: L=$%.6g U=$%.6g (P10/P90 of cheapest-API per-request cost over %d trace requests)",
        cost_envelope[0],
        cost_envelope[1],
        len(trace),
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
        periodic_probe_interval_sec=args.periodic_probe_interval_sec,
        periodic_probe_sleep_sec=args.profile_probe_sleep_sec,
    )

    billing_duration_sec = (
        inventory.billing_duration_sec
        if inventory.billing_duration_sec is not None
        else trace_span_sec
    )
    summary_path = runner.finalize(billing_duration_sec=billing_duration_sec)
    logger.info("wrote summary to %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
