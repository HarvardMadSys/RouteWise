"""Runner-level regressions: thread-local session reuse + sentinel
dispatch + dispatch-time backup capacity charge through the runner."""

from __future__ import annotations

import json
import tempfile
import threading
import time
from typing import Any
from unittest.mock import patch

import pytest

from experiments.real_evaluation.inventory import (
    InventoryConfig,
    ProviderSpec,
    ProviderState,
    load_inventory,
    subscription_fixed_cost_for_inventory,
)
from experiments.real_evaluation.policies import (
    OR_AUTO_SENTINEL,
    OR_SORT_SENTINEL_TO_MODE,
    RoutingDecision,
)
from experiments.real_evaluation.recorder import Recorder
from experiments.real_evaluation.runner import (
    PROBE_FAILURE_FALLBACK_TTFT_MS,
    WARMUP_PROBE_PROMPT,
    RealExperimentRunner,
    TraceRequest,
    check_quota_clock_alignment,
    load_trace_jsonl,
)
from experiments.real_evaluation.shared_profile import SharedProfileEventLog
from experiments.real_evaluation.transports import (
    SingleRequestResult,
    TransportConfig as _TransportConfig,
)

_INVENTORY_PATH = "experiments/real_evaluation/data/joint_minimax_m25_online.json"


def _build_runner(
    policy_names: list[str] | None = None,
    *,
    prefix_cache_routing: bool = False,
    shared_profile_events_path=None,
) -> tuple[RealExperimentRunner, Recorder]:
    inventory = load_inventory(_INVENTORY_PATH)
    rec = Recorder(tempfile.mkdtemp())
    runner = RealExperimentRunner(
        inventory=inventory,
        policy_names=policy_names or ["greedy_cost"],
        recorder=rec,
        slo_ms=inventory.primary_slo_ms,
        prefix_cache_routing=prefix_cache_routing,
        shared_profile_events_path=shared_profile_events_path,
        shared_profile_poll_sec=0.01,
    )
    return runner, rec


def _write_trace(tmp_path, rows: list[dict] | list[str]):
    trace_path = tmp_path / "trace.jsonl"
    lines: list[str] = []
    for row in rows:
        if isinstance(row, str):
            lines.append(row)
        else:
            lines.append(json.dumps(row))
    trace_path.write_text("\n".join(lines) + "\n")
    return trace_path


def test_inventory_loads_subscription_plan_facts_from_canonical_yaml() -> None:
    inventory = load_inventory(_INVENTORY_PATH)
    specs = {spec.name: spec for spec in inventory.providers}

    chutes = specs["Chutes_SQ"]
    assert chutes.subscription_plan == "chutes"
    assert chutes.quota_requests == 5000
    assert chutes.quota_window_sec == 86400
    assert [(w.requests, w.window_sec) for w in chutes.quota_windows] == [(5000, 86400.0)]

    minimax = specs["MiniMax_SQ"]
    assert minimax.subscription_plan == "minimax_subscription_plus"
    assert [(w.requests, w.window_sec) for w in minimax.quota_windows] == [
        (4500, 18000.0),
        (45000, 604800.0),
    ]
    state = ProviderState.from_spec(minimax)
    assert state.quota is not None
    assert hasattr(state.quota, "windows")


def test_inventory_allows_scaled_subscription_quota_for_fixed_cost_accounting() -> None:
    inventory = load_inventory(
        "experiments/real_evaluation/data/pilot_or_chutes_subscription_featherless8_rw6_1h.json"
    )
    specs = {spec.name: spec for spec in inventory.providers}

    assert inventory.billing_duration_sec == 3600
    assert specs["Chutes_SQ"].subscription_plan == "chutes"
    assert specs["Chutes_SQ"].quota_requests == 208
    assert specs["Featherless_SC"].subscription_plan == "featherless_premium"
    assert subscription_fixed_cost_for_inventory(
        inventory,
        billing_duration_sec=3600,
    ) == pytest.approx((20.0 + 25.0) / (30 * 24))


def test_or_minimax_subscription_inventory_emulates_real_5h_quota_via_openrouter() -> None:
    inventory = load_inventory(
        "experiments/real_evaluation/data/pilot_or_minimax_subscription_or8_top_24h.json"
    )
    specs = {spec.name: spec for spec in inventory.providers}

    minimax = specs["MiniMax_Plus_SQ"]
    assert inventory.billing_duration_sec == 8 * 3600
    assert minimax.tier == "quota"
    assert minimax.transport_cfg.transport == "openrouter"
    assert minimax.transport_cfg.provider_hint == "Minimax"
    assert minimax.billing_mode == "subscription"
    assert minimax.quota_requests == 2143
    assert minimax.quota_window_sec == 28800
    assert minimax.subscription_plan == "minimax_subscription_plus"
    assert minimax.fixed_cost_usd_override == pytest.approx(0.2381)
    assert "OR_Minimax" not in specs
    assert subscription_fixed_cost_for_inventory(
        inventory,
        billing_duration_sec=inventory.billing_duration_sec or 0,
    ) == pytest.approx(0.2381 + (25.0 / (30.0 * 3.0)))


def test_or_glm_subscription_inventory_emulates_coding_plan_via_openrouter() -> None:
    inventory = load_inventory(
        "experiments/real_evaluation/data/pilot_or_glm51_subscription_or8_top_24h.json"
    )
    specs = {spec.name: spec for spec in inventory.providers}

    glm = specs["GLM_OR_SQ"]
    assert inventory.billing_duration_sec == 8 * 3600
    assert specs["Featherless_SC"].transport_cfg.model == "zai-org/GLM-5.1"
    assert specs["Featherless_SC"].concurrency_limit == 1
    assert glm.tier == "quota"
    assert glm.transport_cfg.transport == "openrouter"
    assert glm.transport_cfg.provider_hint == "Z.AI"
    assert glm.transport_cfg.model == "z-ai/glm-5.1"
    assert glm.billing_mode == "subscription"
    assert glm.quota_requests == 2000
    assert glm.quota_window_sec == 9 * 3600
    assert glm.subscription_plan == "zai_glm_coding_max"
    assert glm.quota_windows == ()
    assert glm.fixed_cost_usd_override == pytest.approx(0.5)
    assert "OR_ZAI" not in specs
    assert subscription_fixed_cost_for_inventory(
        inventory,
        billing_duration_sec=inventory.billing_duration_sec or 0,
    ) == pytest.approx(0.5 + (25.0 / (30.0 * 3.0)))


def test_native_glm_subscription_inventory_uses_zai_key_for_quota() -> None:
    inventory = load_inventory(
        "experiments/real_evaluation/data/pilot_zai_glm51_subscription_or8_top_24h.json"
    )
    specs = {spec.name: spec for spec in inventory.providers}

    glm = specs["GLM_ZAI_SQ"]
    assert inventory.billing_duration_sec == 8 * 3600
    assert specs["Featherless_SC"].transport_cfg.model == "zai-org/GLM-5.1"
    assert specs["Featherless_SC"].concurrency_limit == 1
    assert glm.tier == "quota"
    assert glm.transport_cfg.transport == "zai_native"
    assert glm.transport_cfg.api_key_env == "ZAI_API_KEY"
    assert glm.transport_cfg.base_url == "https://api.z.ai/api/coding/paas/v4"
    assert glm.transport_cfg.model == "glm-5.1"
    assert glm.billing_mode == "subscription"
    assert glm.quota_requests == 2000
    assert glm.quota_window_sec == 9 * 3600
    assert glm.subscription_plan == "zai_glm_coding_max"
    assert glm.fixed_cost_usd_override == pytest.approx(0.5)
    assert "Z.AI" not in inventory.openrouter_provider_only
    assert subscription_fixed_cost_for_inventory(
        inventory,
        billing_duration_sec=inventory.billing_duration_sec or 0,
    ) == pytest.approx(0.5 + (25.0 / (30.0 * 3.0)))


def _api_only_inventory() -> InventoryConfig:
    """Minimal inventory with no quota- or concurrency-bearing providers."""
    spec = ProviderSpec(
        name="OR_x",
        tier="api",
        transport_cfg=_TransportConfig(
            name="OR_x",
            transport="openrouter",
            model="x",
            stream_cancel_billing="stops",
        ),
    )
    return InventoryConfig(
        model_family="test",
        openrouter_model_id="test/model",
        primary_slo_ms=2000.0,
        slo_thresholds_ms=[1000.0, 2000.0],
        providers=[spec],
    )


def _quota_inventory() -> InventoryConfig:
    """Minimal inventory with one quota-bearing provider."""
    spec = ProviderSpec(
        name="Chutes_SQ",
        tier="quota",
        transport_cfg=_TransportConfig(name="Chutes_SQ", transport="chutes", model="x"),
        quota_window_sec=86400.0,
        quota_requests=5000,
    )
    return InventoryConfig(
        model_family="test",
        openrouter_model_id="test/model",
        primary_slo_ms=2000.0,
        slo_thresholds_ms=[1000.0, 2000.0],
        providers=[spec],
    )


def test_check_quota_clock_alignment_passes_for_api_only_inventory_at_any_speedup() -> None:
    inventory = _api_only_inventory()
    # API-only providers do not roll a wall-clock window, so any speedup is
    # safe regardless of the flag.
    check_quota_clock_alignment(inventory, speedup=1.0, allow_mismatch=False)
    check_quota_clock_alignment(inventory, speedup=3.0, allow_mismatch=False)


def test_check_quota_clock_alignment_passes_for_quota_inventory_at_unit_speedup() -> None:
    inventory = _quota_inventory()
    check_quota_clock_alignment(inventory, speedup=1.0, allow_mismatch=False)


def test_check_quota_clock_alignment_blocks_quota_inventory_at_non_unit_speedup() -> None:
    """Quota windows roll on wall-clock. A non-unit speedup keeps the
    wall-clock window length unchanged while compressing the trace's
    logical day, silently breaking the paper's requests-per-day semantics
    without raising any HTTP error."""
    inventory = _quota_inventory()
    with pytest.raises(SystemExit, match=r"--speedup is 3\.0"):
        check_quota_clock_alignment(inventory, speedup=3.0, allow_mismatch=False)


def test_check_quota_clock_alignment_escape_hatch_allows_explicit_mismatch() -> None:
    inventory = _quota_inventory()
    # ``--allow-quota-clock-mismatch`` opts in to the broken alignment for
    # intentional ablations.
    check_quota_clock_alignment(inventory, speedup=3.0, allow_mismatch=True)


def test_inventory_openrouter_filter_limits_loaded_or_provider_pool(tmp_path) -> None:
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "model_family": "test",
                "openrouter_model_id": "test/model",
                "primary_slo_ms": 2000,
                "slo_thresholds_ms": [1000, 2000],
                "openrouter_provider_only": ["Chutes"],
                "providers": [
                    {
                        "name": "Chutes_SQ",
                        "tier": "quota",
                        "transport": "chutes",
                        "model": "provider/model",
                        "input_price_per_m": 0.0,
                        "output_price_per_m": 0.0,
                        "quota_window_sec": 3600,
                        "quota_requests": 100,
                    },
                    {
                        "name": "OR_Chutes",
                        "tier": "api",
                        "transport": "openrouter",
                        "model": "test/model",
                        "provider_hint": "Chutes",
                        "input_price_per_m": 0.1,
                        "output_price_per_m": 1.0,
                        "stream_cancel_billing": "stops",
                    },
                    {
                        "name": "OR_DeepInfra",
                        "tier": "api",
                        "transport": "openrouter",
                        "model": "test/model",
                        "provider_hint": "DeepInfra",
                        "input_price_per_m": 0.2,
                        "output_price_per_m": 1.2,
                        "stream_cancel_billing": "stops",
                    },
                ],
            }
        )
    )

    inventory = load_inventory(inventory_path)

    assert [spec.name for spec in inventory.providers] == ["Chutes_SQ", "OR_Chutes"]
    assert inventory.openrouter_provider_only == ("Chutes",)
    assert inventory.openrouter_stream_cancel_billing_by_provider == {"chutes": "stops"}


def test_or_sentinel_base_prefers_metered_api_openrouter_config(tmp_path) -> None:
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "model_family": "test",
                "openrouter_model_id": "test/model",
                "primary_slo_ms": 2000,
                "slo_thresholds_ms": [1000, 2000],
                "providers": [
                    {
                        "name": "Chutes_SQ",
                        "tier": "quota",
                        "transport": "openrouter",
                        "model": "test/model",
                        "provider_hint": "Chutes",
                        "input_price_per_m": 0.0,
                        "output_price_per_m": 0.0,
                        "quota_window_sec": 3600,
                        "quota_requests": 100,
                        "billing_mode": "subscription",
                        "stream_cancel_billing": "stops",
                    },
                    {
                        "name": "OR_Chutes",
                        "tier": "api",
                        "transport": "openrouter",
                        "model": "test/model",
                        "provider_hint": "Chutes",
                        "input_price_per_m": 0.1,
                        "output_price_per_m": 1.0,
                        "billing_mode": "metered",
                        "stream_cancel_billing": "stops",
                    },
                ],
            }
        )
    )
    inventory = load_inventory(inventory_path)
    recorder = Recorder(tmp_path / "out")

    runner = RealExperimentRunner(
        inventory=inventory,
        policy_names=["or_auto"],
        recorder=recorder,
        slo_ms=inventory.primary_slo_ms,
    )

    assert runner._or_base_cfg is not None
    assert runner._or_base_cfg.name == "OR_Chutes"
    assert runner._or_base_cfg.billing_mode == "metered"
    recorder.close()


def test_load_trace_jsonl_requires_real_prompt_and_token_fields(tmp_path) -> None:
    trace_path = _write_trace(
        tmp_path,
        [
            {
                "arrived_at": 10.0,
                "prompt_text": "Explain TCP briefly.",
                "num_prefill_tokens": 5,
                "num_decode_tokens": 32,
                "sharegpt_conversation_id": "conv-1",
            }
        ],
    )

    trace = load_trace_jsonl(trace_path)

    assert len(trace) == 1
    assert trace[0].arrival_time_sec == 0.0
    assert trace[0].prompt == "Explain TCP briefly."
    assert trace[0].prompt_tokens == 5
    assert trace[0].max_tokens == 32
    assert trace[0].prefix_id == "conv-1"


def test_load_trace_jsonl_prefix_id_prefers_explicit_then_conversation(tmp_path) -> None:
    trace_path = _write_trace(
        tmp_path,
        [
            {
                "arrived_at": 10.0,
                "prompt_text": "explicit",
                "num_prefill_tokens": 5,
                "num_decode_tokens": 32,
                "prefix_id": "explicit-prefix",
                "sharegpt_conversation_id": "conv-1",
                "session_id": "session-1",
            },
            {
                "arrived_at": 11.0,
                "prompt_text": "conversation",
                "num_prefill_tokens": 6,
                "num_decode_tokens": 16,
                "sharegpt_conversation_id": "conv-2",
                "session_id": "session-2",
            },
            {
                "arrived_at": 12.0,
                "prompt_text": "session",
                "num_prefill_tokens": 7,
                "num_decode_tokens": 8,
                "session_id": "session-3",
            },
        ],
    )

    trace = load_trace_jsonl(trace_path)

    assert [row.prefix_id for row in trace] == [
        "explicit-prefix",
        "conv-2",
        "session-3",
    ]


def test_load_trace_jsonl_accepts_freeinference_schema_and_sorts_arrivals(tmp_path) -> None:
    trace_path = _write_trace(
        tmp_path,
        [
            {
                "timestamp": "2026-05-12T00:00:10Z",
                "prompt_text": "Say hello again.",
                "prompt_tokens": "128",
                "completion_tokens": "9",
                "cache_read_tokens": "64",
                "status_code": 200,
                "user_id": "u-1",
            },
            {
                "timestamp": "2026-05-12T00:00:05+00:00",
                "prompt_text": "failed request",
                "prompt_tokens": 64,
                "completion_tokens": 8,
                "status_code": 500,
                "user_id": "skip-me",
            },
            {
                "timestamp": "2026-05-12T00:00:00+00:00",
                "prompt": "Say hello.",
                "prompt_tokens": 64,
                "completion_tokens": 4,
                "cache_read_tokens": 12,
                "status_code": 200,
                "user_id": "u-0",
            },
        ],
    )

    trace = load_trace_jsonl(trace_path)

    assert [row.arrival_time_sec for row in trace] == [0.0, 10.0]
    assert [row.prompt for row in trace] == ["Say hello.", "Say hello again."]
    assert [row.prompt_tokens for row in trace] == [64, 128]
    assert [row.max_tokens for row in trace] == [4, 9]
    assert [row.trace_cached_input_tokens for row in trace] == [12, 64]
    assert [row.prefix_id for row in trace] == ["u-0", "u-1"]


def test_load_trace_jsonl_synthesizes_missing_freeinference_prompt(tmp_path) -> None:
    trace_path = _write_trace(
        tmp_path,
        [
            {
                "timestamp": "2026-05-12T00:00:00Z",
                "prompt_tokens": 64,
                "completion_tokens": 128,
                "cache_read_tokens": 48,
                "status_code": 200,
                "user_id": "u-0",
            },
        ],
    )

    trace = load_trace_jsonl(
        trace_path,
        synthesize_missing_prompts=True,
        synthetic_output_tokens=256,
    )

    assert len(trace) == 1
    assert trace[0].prompt_tokens == 64
    assert len(trace[0].prompt.split()) == 64
    assert trace[0].prompt.startswith("a a a")
    assert trace[0].prompt.endswith(
        "FINAL REQUEST: Ignore all previous synthetic padding. Do not include analysis, "
        "reasoning, markdown, labels, or <think> tags. Output only a fictional story "
        "of about 256 tokens. Start the story with: Once upon a time"
    )
    assert trace[0].max_tokens == 256
    assert trace[0].trace_cached_input_tokens == 48


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            {
                "arrived_at": 10.0,
                "num_prefill_tokens": 5,
                "num_decode_tokens": 32,
            },
            "missing non-empty prompt",
        ),
        (
            {
                "arrived_at": 10.0,
                "prompt_text": "Explain TCP briefly.",
                "num_decode_tokens": 32,
            },
            "missing prompt token count",
        ),
        (
            {
                "arrived_at": 10.0,
                "prompt_text": "Explain TCP briefly.",
                "num_prefill_tokens": 5,
            },
            "missing output token cap",
        ),
        (
            {
                "prompt_text": "Explain TCP briefly.",
                "num_prefill_tokens": 5,
                "num_decode_tokens": 32,
            },
            "missing arrival timestamp",
        ),
    ],
)
def test_load_trace_jsonl_fails_fast_on_missing_or_invalid_fields(
    tmp_path,
    row,
    message,
) -> None:
    trace_path = _write_trace(tmp_path, [row])

    with pytest.raises(ValueError, match=message):
        load_trace_jsonl(trace_path)


def test_load_trace_jsonl_fails_fast_on_bad_json(tmp_path) -> None:
    trace_path = _write_trace(tmp_path, ['{"arrived_at": 1'])

    with pytest.raises(ValueError, match="invalid JSON"):
        load_trace_jsonl(trace_path)


def test_load_trace_jsonl_skips_nonpositive_output_caps(tmp_path, caplog) -> None:
    trace_path = _write_trace(
        tmp_path,
        [
            {
                "arrived_at": 10.0,
                "prompt_text": "skip zero decode",
                "num_prefill_tokens": 5,
                "num_decode_tokens": 0,
            },
            {
                "arrived_at": 12.0,
                "prompt_text": "keep valid decode",
                "num_prefill_tokens": 6,
                "num_decode_tokens": 32,
            },
        ],
    )

    trace = load_trace_jsonl(trace_path)

    assert len(trace) == 1
    assert trace[0].arrival_time_sec == 2.0
    assert trace[0].prompt == "keep valid decode"
    assert "Skipped 1 trace rows" in caplog.text


def test_session_is_reused_per_thread() -> None:
    """Two ``_session()`` calls on the same thread must return the same
    Session object. (The earlier ``threading.local()`` declared inside
    the function created a fresh storage on every call.)"""
    runner, _ = _build_runner()
    seen: list[bool] = []

    def in_thread() -> None:
        a = runner._session()
        b = runner._session()
        seen.append(a is b)

    threads = [threading.Thread(target=in_thread) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(seen)


def test_session_is_distinct_across_threads() -> None:
    """Different live threads must see different Session objects.

    Use a pair of events so all threads acquire their session and then
    pause until the main test has collected all identities. Without
    this hold, CPython could recycle a Session's memory address after
    one thread exits, defeating identity comparison via ``id()``.
    """
    runner, _ = _build_runner()
    n = 3
    ready = threading.Semaphore(0)
    release = threading.Event()
    sessions: list[Any] = []
    lock = threading.Lock()

    def in_thread() -> None:
        sess = runner._session()
        with lock:
            sessions.append(sess)
        ready.release()
        release.wait(timeout=5.0)

    threads = [threading.Thread(target=in_thread) for _ in range(n)]
    for t in threads:
        t.start()
    for _ in range(n):
        assert ready.acquire(timeout=5.0)
    # All n sessions are simultaneously live; identities must differ.
    try:
        assert len({id(s) for s in sessions}) == n
    finally:
        release.set()
        for t in threads:
            t.join(timeout=5.0)


def test_warmup_broadcasts_profile_samples_and_guard(monkeypatch) -> None:
    runner, rec = _build_runner(policy_names=["budget_range_p75_hedge"])

    def fake_send_via_transport(
        provider: str,
        prompt: str,
        max_tokens: int,
        timeout: int,
        ttft_event: threading.Event | None,
        ttft_info: dict[str, Any] | None,
        cancel_event: threading.Event | None = None,
    ) -> SingleRequestResult:
        assert prompt == WARMUP_PROBE_PROMPT
        return SingleRequestResult(
            ttft_ms=100.0,
            e2e_ms=150.0,
            status="success",
            provider=provider,
            billed_cost_usd=0.001,
            start_ts=time.time(),
            first_token_ts=time.time(),
        )

    monkeypatch.setattr(runner, "_send_via_transport", fake_send_via_transport)

    runner.warmup(probes_per_provider=5, sleep_sec=0.0, round_interval_sec=0.0)
    runner.validate_profile_bootstrap(min_success_samples=5)
    with pytest.raises(RuntimeError, match="profile bootstrap failed"):
        runner.validate_profile_bootstrap(min_success_samples=6)

    assert runner._profile_probe_counts["warmup"] == len(runner.inventory.providers) * 5
    assert runner._profile_probe_cost_usd == pytest.approx(
        len(runner.inventory.providers) * 5 * 0.001
    )
    rec.close()


def test_warmup_probes_all_providers_each_round(monkeypatch) -> None:
    runner, rec = _build_runner(policy_names=["budget_range_p75_hedge"])
    seen: list[str] = []

    def fake_send_via_transport(
        provider: str,
        prompt: str,
        max_tokens: int,
        timeout: int,
        ttft_event: threading.Event | None,
        ttft_info: dict[str, Any] | None,
        cancel_event: threading.Event | None = None,
    ) -> SingleRequestResult:
        seen.append(provider)
        return SingleRequestResult(
            ttft_ms=100.0,
            e2e_ms=150.0,
            status="success",
            provider=provider,
            start_ts=time.time(),
            first_token_ts=time.time(),
        )

    monkeypatch.setattr(runner, "_send_via_transport", fake_send_via_transport)

    runner.warmup(probes_per_provider=2, sleep_sec=0.0, round_interval_sec=0.0)

    providers = [spec.name for spec in runner.inventory.providers]
    assert sorted(seen) == sorted(providers + providers)
    rec.close()


def test_warmup_probes_providers_in_parallel(monkeypatch) -> None:
    runner, rec = _build_runner(policy_names=["budget_range_p75_hedge"])
    providers = [spec.name for spec in runner.inventory.providers]
    barrier = threading.Barrier(len(providers))
    lock = threading.Lock()
    active = 0
    peak_active = 0

    def fake_send_via_transport(
        provider: str,
        prompt: str,
        max_tokens: int,
        timeout: int,
        ttft_event: threading.Event | None,
        ttft_info: dict[str, Any] | None,
        cancel_event: threading.Event | None = None,
    ) -> SingleRequestResult:
        del provider, max_tokens, timeout, ttft_event, ttft_info, cancel_event
        assert prompt == WARMUP_PROBE_PROMPT
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        try:
            barrier.wait(timeout=5.0)
            now = time.time()
            return SingleRequestResult(
                ttft_ms=100.0,
                e2e_ms=150.0,
                status="success",
                provider="probe",
                start_ts=now,
                first_token_ts=now + 0.1,
            )
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(runner, "_send_via_transport", fake_send_via_transport)

    runner.warmup(probes_per_provider=1, sleep_sec=0.0, round_interval_sec=0.0)

    assert peak_active == len(providers)
    rec.close()


def test_warmup_probe_no_retry_and_records_synthetic_sample(monkeypatch) -> None:
    """Warmup probes never retry: a single failure records exactly one
    synthetic high-latency sample so the LP can still tentatively rank the
    provider on startup. Replay-time probes are handled separately (see
    ``test_replay_probe_failure_does_not_record_any_sample``)."""
    runner, rec = _build_runner(policy_names=["budget_range_p75_hedge"])
    bad_provider = runner.inventory.providers[0].name
    calls: dict[str, int] = {}

    def fake_send_via_transport(
        provider: str,
        prompt: str,
        max_tokens: int,
        timeout: int,
        ttft_event: threading.Event | None,
        ttft_info: dict[str, Any] | None,
        cancel_event: threading.Event | None = None,
    ) -> SingleRequestResult:
        del max_tokens, timeout, ttft_event, ttft_info, cancel_event
        assert prompt == WARMUP_PROBE_PROMPT
        calls[provider] = calls.get(provider, 0) + 1
        if provider == bad_provider:
            return SingleRequestResult(
                ttft_ms=-1.0,
                e2e_ms=-1.0,
                status="HTTP 429",
                provider=provider,
                http_status=429,
                rate_limited=True,
            )
        now = time.time()
        return SingleRequestResult(
            ttft_ms=120.0,
            e2e_ms=180.0,
            status="success",
            provider=provider,
            billed_cost_usd=0.001,
            physical_cost_usd=0.001,
            start_ts=now,
            first_token_ts=now + 0.12,
        )

    monkeypatch.setattr(runner, "_send_via_transport", fake_send_via_transport)
    runner.probe_profiles(probes_per_provider=1, sleep_sec=0.0, phase="warmup")

    providers = [spec.name for spec in runner.inventory.providers]
    assert calls[bad_provider] == 1, (
        "probe should not retry now that DEFAULT_PROBE_MAX_ATTEMPTS=1"
    )
    assert sum(calls.values()) == len(providers)

    state = runner.policies["budget_range_p75_hedge"].states[bad_provider]
    now = time.time()
    assert state.profile.sample_count(now) == 1, (
        "warmup failure must still inject one synthetic positive-TTFT sample"
    )
    assert state.profile.error_rate(now) == 0.0
    mean = state.profile.mean_ms(now)
    assert mean == PROBE_FAILURE_FALLBACK_TTFT_MS

    for healthy_spec in runner.inventory.providers[1:]:
        healthy_state = runner.policies["budget_range_p75_hedge"].states[healthy_spec.name]
        assert healthy_state.profile.sample_count(now) == 1, (
            "one provider's failure must not abort the warmup round"
        )
    rec.close()


def test_warmup_cadenced_skips_concurrency_limited_provider_in_flight(
    monkeypatch,
) -> None:
    """Multi-round cadenced warmup:

    - Uses start-to-start cadence (fast provider sees 3 probes at ~0.1s gaps)
    - Providers without ``concurrency_limit`` get a probe every round even
      while a previous probe is still in flight (no real production traffic
      can be displaced from a slot they don't have)
    - Providers WITH ``concurrency_limit`` (Featherless_SC = 1) get
      skip-in-flight, so our probe never occupies the only account slot
      while real production traffic also needs it.
    """
    runner, rec = _build_runner(policy_names=["budget_range_p75_hedge"])
    providers = list(runner.inventory.providers)
    capped_in_flight = next(
        p for p in providers if p.concurrency_limit is not None
    ).name
    uncapped_in_flight = next(
        p for p in providers if p.concurrency_limit is None
    ).name
    fast_provider = next(
        p
        for p in providers
        if p.name not in (capped_in_flight, uncapped_in_flight)
    ).name
    release = threading.Event()
    submit_ts: dict[str, list[float]] = {p.name: [] for p in providers}
    submit_lock = threading.Lock()

    def fake_send_via_transport(
        provider: str,
        prompt: str,
        max_tokens: int,
        timeout: int,
        ttft_event: threading.Event | None,
        ttft_info: dict[str, Any] | None,
        cancel_event: threading.Event | None = None,
    ) -> SingleRequestResult:
        del prompt, max_tokens, timeout, ttft_event, ttft_info, cancel_event
        with submit_lock:
            submit_ts[provider].append(time.time())
        if provider in (capped_in_flight, uncapped_in_flight):
            release.wait(timeout=5.0)
        now = time.time()
        return SingleRequestResult(
            ttft_ms=50.0,
            e2e_ms=80.0,
            status="success",
            provider=provider,
            billed_cost_usd=0.001,
            physical_cost_usd=0.001,
            start_ts=now,
            first_token_ts=now + 0.05,
        )

    monkeypatch.setattr(runner, "_send_via_transport", fake_send_via_transport)

    def run_probe() -> None:
        runner.probe_profiles(
            probes_per_provider=3,
            sleep_sec=0.0,
            round_interval_sec=0.1,
            phase="warmup",
        )

    worker = threading.Thread(target=run_probe)
    worker.start()
    time.sleep(0.35)
    release.set()
    worker.join(timeout=10.0)
    assert not worker.is_alive(), "probe_profiles should return after release"

    fast_calls = submit_ts[fast_provider]
    assert len(fast_calls) == 3, (
        f"fast provider should see 3 probes (one per round), "
        f"got {len(fast_calls)}"
    )
    import itertools

    for prev, nxt in itertools.pairwise(fast_calls):
        gap = nxt - prev
        assert 0.07 <= gap <= 0.5, (
            f"start-to-start cadence violated: gap={gap:.3f}s "
            f"(expected ~0.1s)"
        )
    # Uncapped slow provider: probed every tick despite in-flight.
    assert len(submit_ts[uncapped_in_flight]) == 3, (
        f"uncapped slow provider should be probed every round, got "
        f"{len(submit_ts[uncapped_in_flight])}"
    )
    # Capped slow provider: first probe submitted, subsequent rounds skip
    # until the in-flight one returns. Worker releases after all 3 ticks,
    # so we expect exactly 1 submit.
    assert len(submit_ts[capped_in_flight]) == 1, (
        f"capped slow provider should be skipped while in flight, got "
        f"{len(submit_ts[capped_in_flight])}"
    )
    rec.close()


def test_replay_probe_failure_does_not_record_any_sample(monkeypatch) -> None:
    """Replay-time probe failures (periodic / shared sidecar) must not write
    to the latency profile at all — neither a positive synthetic sample nor
    an error sample. Real-request 429s remain the only way a provider gets
    penalized (via the ``error_samples`` path with
    ``RATE_LIMIT_ERROR_PENALTY_MS``)."""
    runner, rec = _build_runner(policy_names=["budget_range_p75_hedge"])
    bad_provider = runner.inventory.providers[0].name

    def fake_send_via_transport(
        provider: str,
        prompt: str,
        max_tokens: int,
        timeout: int,
        ttft_event: threading.Event | None,
        ttft_info: dict[str, Any] | None,
        cancel_event: threading.Event | None = None,
    ) -> SingleRequestResult:
        del prompt, max_tokens, timeout, ttft_event, ttft_info, cancel_event
        if provider == bad_provider:
            return SingleRequestResult(
                ttft_ms=-1.0,
                e2e_ms=-1.0,
                status="HTTP 429",
                provider=provider,
                http_status=429,
                rate_limited=True,
            )
        now = time.time()
        return SingleRequestResult(
            ttft_ms=120.0,
            e2e_ms=180.0,
            status="success",
            provider=provider,
            billed_cost_usd=0.001,
            physical_cost_usd=0.001,
            start_ts=now,
            first_token_ts=now + 0.12,
        )

    monkeypatch.setattr(runner, "_send_via_transport", fake_send_via_transport)
    runner.probe_profiles(probes_per_provider=1, sleep_sec=0.0, phase="periodic")

    now = time.time()
    bad_state = runner.policies["budget_range_p75_hedge"].states[bad_provider]
    assert bad_state.profile.sample_count(now) == 0, (
        "non-warmup probe failure must not record a synthetic sample"
    )
    assert bad_state.profile.total_count(now) == 0, (
        "non-warmup probe failure must not record an error sample either"
    )
    rec.close()


def test_initial_profile_loads_into_all_policy_profiles(tmp_path) -> None:
    runner, rec = _build_runner(policy_names=["greedy_latency", "budget_range_p100"])
    now = time.time()
    profile_path = tmp_path / "initial_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {
                    spec.name: {
                        "samples": [{"ts": now - idx, "ttft_ms": 100.0 + idx} for idx in range(5)],
                        "errors": [{"ts": now, "error_type": "HTTP 500"}],
                    }
                    for spec in runner.inventory.providers
                },
            }
        )
    )

    runner.load_initial_profile(profile_path)
    runner.validate_profile_bootstrap(min_success_samples=5)

    for policy in runner.policies.values():
        for spec in runner.inventory.providers:
            state = policy.states[spec.name]
            assert state.profile.sample_count(time.time()) == 5
            assert state.profile.error_rate(time.time()) > 0

    rec.close()


def test_shared_profile_event_log_round_trip_and_dedupe(tmp_path) -> None:
    path = tmp_path / "shared_profile_events.jsonl"
    writer = SharedProfileEventLog(path)
    reader = SharedProfileEventLog(path)

    event = writer.append(
        provider="Chutes_SQ",
        ts=123.0,
        ttft_ms=456.0,
        error_type=None,
        source="natural",
        policy="greedy_latency",
    )

    assert writer.read_new() == []
    assert [e.event_id for e in reader.read_new()] == [event.event_id]
    assert reader.read_new() == []
    latest = reader.recent_last_event_by_provider(now=124.0, window_sec=15.0)
    assert latest["Chutes_SQ"].ttft_ms == pytest.approx(456.0)


def test_shared_profile_event_log_can_skip_seed_prefix(tmp_path) -> None:
    path = tmp_path / "shared_profile_events.jsonl"
    writer = SharedProfileEventLog(path)
    reader = SharedProfileEventLog(path)

    writer.append(
        provider="Chutes_SQ",
        ts=123.0,
        ttft_ms=456.0,
        error_type=None,
        source="warmup",
    )
    seed_offset = writer.end_offset()
    later = writer.append(
        provider="Chutes_SQ",
        ts=124.0,
        ttft_ms=789.0,
        error_type=None,
        source="natural",
    )

    reader.set_read_offset(seed_offset)

    events = reader.read_new()
    assert [event.event_id for event in events] == [later.event_id]
    assert events[0].ttft_ms == pytest.approx(789.0)


def test_shared_profile_event_log_concurrent_append_and_tail(tmp_path) -> None:
    path = tmp_path / "shared_profile_events.jsonl"
    writer = SharedProfileEventLog(path)
    reader = SharedProfileEventLog(path)
    n_events = 100

    def append_events() -> None:
        for idx in range(n_events):
            writer.append(
                provider="OR_Test",
                ts=float(idx),
                ttft_ms=100.0 + idx,
                error_type=None,
                source="natural",
            )

    thread = threading.Thread(target=append_events)
    thread.start()
    seen: set[str] = set()
    while thread.is_alive():
        seen.update(event.event_id for event in reader.read_new())
        time.sleep(0.001)
    thread.join(timeout=5.0)
    seen.update(event.event_id for event in reader.read_new())

    assert len(seen) == n_events
    assert writer.read_new() == []


def test_initial_profile_seed_offset_prevents_shared_log_duplicate(tmp_path) -> None:
    path = tmp_path / "shared_profile_events.jsonl"
    event_log = SharedProfileEventLog(path)
    runner, rec = _build_runner(
        policy_names=["greedy_latency"],
        shared_profile_events_path=path,
    )
    provider = runner.inventory.providers[0].name
    now = time.time()

    event_log.append(
        provider=provider,
        ts=now - 2.0,
        ttft_ms=111.0,
        error_type=None,
        source="warmup",
    )
    seed_offset = event_log.end_offset()
    event_log.append(
        provider=provider,
        ts=now - 1.0,
        ttft_ms=222.0,
        error_type=None,
        source="natural",
    )
    profile_path = tmp_path / "initial_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "version": 1,
                "_meta": {"shared_profile_seed_offset": seed_offset},
                "providers": {
                    provider: {
                        "samples": [{"ts": now - 2.0, "ttft_ms": 111.0}],
                        "errors": [],
                    }
                },
            }
        )
    )

    runner.load_initial_profile(profile_path)
    handle = runner._start_shared_profile_tailer_thread()
    try:
        deadline = time.time() + 2.0
        state = runner.policies["greedy_latency"].states[provider]
        while time.time() < deadline and state.profile.sample_count(time.time()) < 2:
            time.sleep(0.02)
        assert state.profile.sample_count(time.time()) == 2
        assert state.profile.mean_ms(time.time()) == pytest.approx((111.0 + 222.0) / 2)
    finally:
        runner._stop_periodic_profile_probe_thread(handle)
        rec.close()


def test_shared_profile_feedback_updates_all_local_policies(tmp_path) -> None:
    path = tmp_path / "shared_profile_events.jsonl"
    runner, rec = _build_runner(
        policy_names=["greedy_latency", "random"],
        shared_profile_events_path=path,
    )
    provider = runner.inventory.providers[0].name
    policy = runner.policies["greedy_latency"]
    now = time.time()
    first_token_ts = now + 0.321

    runner._feed_back_single(
        policy,
        provider,
        SingleRequestResult(
            ttft_ms=321.0,
            e2e_ms=500.0,
            status="success",
            provider=provider,
            start_ts=now,
            first_token_ts=first_token_ts,
        ),
    )

    for local_policy in runner.policies.values():
        state = local_policy.states[provider]
        assert state.profile.sample_count(time.time()) == 1
        assert state.profile.mean_ms(time.time()) == pytest.approx(321.0)
    latest = SharedProfileEventLog(path).recent_last_event_by_provider(
        now=first_token_ts + 1.0,
        window_sec=15.0,
    )
    assert latest[provider].ts == pytest.approx(first_token_ts)
    assert path.exists()
    rec.close()


def test_shared_profile_tailer_imports_external_events(tmp_path) -> None:
    path = tmp_path / "shared_profile_events.jsonl"
    runner, rec = _build_runner(
        policy_names=["greedy_latency"],
        shared_profile_events_path=path,
    )
    provider = runner.inventory.providers[0].name
    external = SharedProfileEventLog(path)
    external.append(
        provider=provider,
        ts=time.time(),
        ttft_ms=654.0,
        error_type=None,
        source="probe",
    )

    handle = runner._start_shared_profile_tailer_thread()
    try:
        deadline = time.time() + 2.0
        state = runner.policies["greedy_latency"].states[provider]
        while time.time() < deadline and state.profile.sample_count(time.time()) == 0:
            time.sleep(0.02)
        assert state.profile.mean_ms(time.time()) == pytest.approx(654.0)
    finally:
        runner._stop_periodic_profile_probe_thread(handle)
        rec.close()


def test_profile_bootstrap_guard_skips_profile_free_policies() -> None:
    runner, rec = _build_runner(policy_names=["or_auto", "or_sort_cost"])

    runner.validate_profile_bootstrap(min_success_samples=5)

    rec.close()


def test_periodic_profile_probe_runs_during_replay(monkeypatch) -> None:
    runner, rec = _build_runner(policy_names=["or_auto"])
    probe_calls: list[str] = []

    def fake_send_via_transport(
        provider: str,
        prompt: str,
        max_tokens: int,
        timeout: int,
        ttft_event: threading.Event | None,
        ttft_info: dict[str, Any] | None,
        cancel_event: threading.Event | None = None,
    ) -> SingleRequestResult:
        assert prompt == WARMUP_PROBE_PROMPT
        probe_calls.append(provider)
        return SingleRequestResult(
            ttft_ms=100.0,
            e2e_ms=150.0,
            status="success",
            provider=provider,
            billed_cost_usd=0.001,
            start_ts=time.time(),
            first_token_ts=time.time(),
        )

    monkeypatch.setattr(runner, "_send_via_transport", fake_send_via_transport)
    monkeypatch.setattr(
        runner,
        "_dispatch_one",
        lambda policy, req, idx: None,
    )

    runner.replay(
        [
            TraceRequest(
                arrival_time_sec=0.0,
                prompt="x",
                prompt_tokens=10,
                max_tokens=8,
            ),
            TraceRequest(
                arrival_time_sec=0.2,
                prompt="x",
                prompt_tokens=10,
                max_tokens=8,
            ),
        ],
        speedup=1.0,
        duration_sec=1.0,
        periodic_probe_interval_sec=0.05,
        periodic_probe_sleep_sec=0.0,
    )

    assert probe_calls
    assert runner._profile_probe_counts["periodic"] == len(probe_calls)
    rec.close()


def test_or_sentinels_round_trip_to_correct_sort_mode() -> None:
    """Each sentinel ``__or_sort_<mode>__`` must dispatch with the
    matching ``provider.sort`` mode set on the transport config."""
    runner, _ = _build_runner()
    captured: dict[str, str | None] = {}

    def fake_send(self, prompt, max_tokens, timeout, ttft_event, ttft_info, cancel_event=None):
        captured["sort_mode"] = self.cfg.sort_mode
        return SingleRequestResult(
            ttft_ms=10.0,
            e2e_ms=20.0,
            status="success",
            provider=self.cfg.name,
            first_token_ts=time.time(),
        )

    from experiments.real_evaluation.transports import (
        OpenAICompatStreamingTransport,
    )

    with patch.object(OpenAICompatStreamingTransport, "send", fake_send):
        for sentinel, expected in OR_SORT_SENTINEL_TO_MODE.items():
            captured.clear()
            runner._send_via_transport(
                provider=sentinel,
                prompt="x",
                max_tokens=8,
                timeout=5,
                ttft_event=None,
                ttft_info=None,
            )
            assert captured["sort_mode"] == expected, (
                f"sentinel {sentinel!r} should map to sort_mode {expected!r}"
            )

        # Auto sentinel must NOT set a sort mode.
        captured.clear()
        runner._send_via_transport(
            provider=OR_AUTO_SENTINEL,
            prompt="x",
            max_tokens=8,
            timeout=5,
            ttft_event=None,
            ttft_info=None,
        )
        assert captured["sort_mode"] is None


def test_or_sentinels_use_api_tier_allowlist_only() -> None:
    """OpenRouter auto/sort sentinels are API-only baselines.

    Sentinel dispatch must derive ``provider.only`` from the inventory's
    API-tier OpenRouter providers, ignoring any extra names that
    ``openrouter_provider_only`` carries for the joint-pool path. The
    sentinel allowlist should equal the set of ``provider_hint`` values
    for API-tier OpenRouter specs — nothing more, nothing less. This
    keeps the baseline a pure API comparison even when subscription
    tiers ride on the openrouter transport (e.g. Chutes via OR).
    """
    runner, _ = _build_runner()
    # Set an inventory-level allowlist that contains a synthetic name not in
    # any spec. If the sentinel were still forwarding the inventory list
    # verbatim, this name would surface in provider_only.
    runner.inventory.openrouter_provider_only = (
        "Friendli",
        "DeepInfra",
        "SubscriptionEndpointNotInProviders",
    )
    runner.inventory.openrouter_provider_ignore = ("BadProvider",)

    captured: dict[str, object] = {}

    def fake_send(self, prompt, max_tokens, timeout, ttft_event, ttft_info, cancel_event=None):
        captured["sort_mode"] = self.cfg.sort_mode
        captured["provider_only"] = self.cfg.provider_only
        captured["provider_ignore"] = self.cfg.provider_ignore
        captured["stream_cancel_billing"] = self.cfg.stream_cancel_billing
        captured["stream_cancel_billing_by_provider"] = (
            self.cfg.stream_cancel_billing_by_provider
        )
        return SingleRequestResult(
            ttft_ms=10.0,
            e2e_ms=20.0,
            status="success",
            provider=self.cfg.name,
            first_token_ts=time.time(),
        )

    from experiments.real_evaluation.transports import (
        OpenAICompatStreamingTransport,
    )

    expected_api_hints = tuple(
        sorted(
            {
                spec.transport_cfg.provider_hint
                for spec in runner.inventory.providers
                if spec.tier == "api"
                and spec.transport_cfg.transport == "openrouter"
                and spec.transport_cfg.provider_hint
            }
        )
    )
    assert expected_api_hints  # sanity: inventory really has API-tier OR providers

    with patch.object(OpenAICompatStreamingTransport, "send", fake_send):
        runner._send_via_transport(
            provider=OR_AUTO_SENTINEL,
            prompt="x",
            max_tokens=8,
            timeout=5,
            ttft_event=None,
            ttft_info=None,
        )
        assert captured["sort_mode"] is None
        assert captured["provider_only"] == expected_api_hints
        # The synthetic non-spec name from the inventory allowlist must be
        # filtered out — sentinels only see what's actually in providers[].
        assert "SubscriptionEndpointNotInProviders" not in captured["provider_only"]
        assert captured["provider_ignore"] == ("BadProvider",)
        assert captured["stream_cancel_billing"] == "continues"
        assert captured["stream_cancel_billing_by_provider"]["chutes"] == "stops"
        assert captured["stream_cancel_billing_by_provider"]["minimax"] == "continues"

        runner._send_via_transport(
            provider="__or_sort_latency__",
            prompt="x",
            max_tokens=8,
            timeout=5,
            ttft_event=None,
            ttft_info=None,
        )
        assert captured["sort_mode"] == "latency"
        assert captured["provider_only"] == expected_api_hints
        assert "SubscriptionEndpointNotInProviders" not in captured["provider_only"]
        assert captured["provider_ignore"] == ("BadProvider",)


def test_or_sentinel_excludes_or_routed_chutes_subscription_from_baseline() -> None:
    """End-to-end regression: when the inventory routes Chutes_SQ through
    OpenRouter as a subscription backup, the OR baseline sentinels must
    still NOT see ``Chutes`` in their candidate set.

    This guards the ``pilot_or_chutes_subscription_or8_top_24h.json``
    contingency, where ``openrouter_provider_only`` has to include
    ``Chutes`` so that ``Chutes_SQ`` survives the inventory filter, but
    OR auto/sort must keep behaving as an 8-API-provider baseline.
    """
    inventory_path = (
        "experiments/real_evaluation/data/pilot_or_chutes_subscription_or8_top_24h.json"
    )
    inventory = load_inventory(inventory_path)
    assert "Chutes" in inventory.openrouter_provider_only  # premise of the bug
    chutes_spec = next(spec for spec in inventory.providers if spec.name == "Chutes_SQ")
    assert chutes_spec.tier == "quota"
    assert chutes_spec.transport_cfg.transport == "openrouter"

    rec = Recorder(tempfile.mkdtemp())
    runner = RealExperimentRunner(
        inventory=inventory,
        policy_names=["or_auto", "or_sort_latency", "or_sort_cost"],
        recorder=rec,
        slo_ms=inventory.primary_slo_ms,
    )

    captured: dict[str, object] = {}

    def fake_send(self, prompt, max_tokens, timeout, ttft_event, ttft_info, cancel_event=None):
        captured["sentinel"] = self.cfg.name
        captured["provider_only"] = self.cfg.provider_only
        return SingleRequestResult(
            ttft_ms=10.0,
            e2e_ms=20.0,
            status="success",
            provider=self.cfg.name,
            first_token_ts=time.time(),
        )

    from experiments.real_evaluation.transports import (
        OpenAICompatStreamingTransport,
    )

    sentinels = [OR_AUTO_SENTINEL, "__or_sort_latency__", "__or_sort_cost__"]
    with patch.object(OpenAICompatStreamingTransport, "send", fake_send):
        for sentinel in sentinels:
            runner._send_via_transport(
                provider=sentinel,
                prompt="x",
                max_tokens=8,
                timeout=5,
                ttft_event=None,
                ttft_info=None,
            )
            assert "Chutes" not in captured["provider_only"], (
                f"sentinel {sentinel!r} leaked Chutes (quota tier) into the OR baseline "
                f"candidate set: {captured['provider_only']}"
            )
            # And it should still cover the API-tier OR providers.
            assert "Friendli" in captured["provider_only"]
            assert "DeepInfra" in captured["provider_only"]
    rec.close()


def test_dispatch_one_charges_backup_at_dispatch_time(monkeypatch) -> None:
    """When a checkpoint hedge fires, backup capacity is charged before send."""
    runner, _ = _build_runner(policy_names=["budget_range_p100_hedge"])
    policy = runner.policies["budget_range_p100_hedge"]

    # Seed every provider with profile data so the LP picks something.
    now = time.time()
    for spec in runner.inventory.providers:
        for _ in range(20):
            policy.add_sample(spec.name, now, 800.0)

    charge_calls: list[tuple[str, float]] = []
    original_charge = policy.charge_capacity

    def tracking_charge(provider: str, ts: float, expected_service_sec: float) -> int | None:
        charge_calls.append((provider, ts))
        return original_charge(provider, ts, expected_service_sec)

    monkeypatch.setattr(policy, "charge_capacity", tracking_charge)

    # Stub the transport: primary fails immediately, backup succeeds.
    def fake_send_via_transport(
        provider: str,
        prompt: str,
        max_tokens: int,
        timeout: int,
        ttft_event: threading.Event | None,
        ttft_info: dict[str, Any] | None,
        cancel_event: threading.Event | None = None,
    ) -> SingleRequestResult:
        if "primary_marker" in provider:
            if ttft_event is not None:
                ttft_event.set()
            if ttft_info is not None:
                ttft_info["status"] = "HTTP 500"
            return SingleRequestResult(
                ttft_ms=-1.0,
                e2e_ms=-1.0,
                status="HTTP 500",
                provider=provider,
                error_message="server_error",
                start_ts=time.time(),
            )
        # When the runner sends to anyone other than the synthetic primary,
        # treat it as a successful send.
        if ttft_info is not None:
            ttft_info.update(ttft_ms=200.0, first_token_ts=time.time(), status="success")
        if ttft_event is not None:
            ttft_event.set()
        return SingleRequestResult(
            ttft_ms=200.0,
            e2e_ms=300.0,
            status="success",
            provider=provider,
            first_token_ts=time.time(),
            start_ts=time.time(),
        )

    monkeypatch.setattr(runner, "_send_via_transport", fake_send_via_transport)

    # Force the policy to pick a known primary by patching its route() to
    # return a primary the fake_send treats as failing.
    primary_name = next(iter(policy.states.keys()))
    backup_name = next(iter(spec for spec in policy.states if spec != primary_name))
    primary_marker = f"{primary_name}_primary_marker"

    # Inject the marker into the states dict so charge_capacity matches.
    policy.states[primary_marker] = policy.states[primary_name]
    monkeypatch.setattr(
        policy,
        "route",
        lambda now, ctx: __import__(
            "experiments.real_evaluation.policies", fromlist=["RoutingDecision"]
        ).RoutingDecision(primary=primary_marker, notes="test"),
    )
    monkeypatch.setattr(
        "experiments.real_evaluation.runner.hedge_checkpoints_for_slo",
        lambda slo_sec: (0.0,),
    )
    checkpoint_decision_cls = __import__(
        "experiments.real_evaluation.policies",
        fromlist=["CheckpointHedgeDecision"],
    ).CheckpointHedgeDecision

    def checkpoint_and_charge(**kwargs):
        capacity_id = policy.charge_capacity(
            backup_name,
            kwargs["now"],
            kwargs["expected_service_sec"],
        )
        return (
            checkpoint_decision_cls(
                backup=backup_name,
                elapsed_sec=kwargs["elapsed_sec"],
                success_probability=0.99,
            ),
            capacity_id,
        )

    monkeypatch.setattr(
        policy,
        "checkpoint_backup_and_charge_capacity",
        checkpoint_and_charge,
    )

    from experiments.real_evaluation.runner import TraceRequest

    runner._dispatch_one(
        policy=policy,
        req=TraceRequest(arrival_time_sec=0.0, prompt="x", prompt_tokens=10, max_tokens=8),
        req_index=0,
    )

    # Both primary and backup must have been charged. Order: primary first
    # (route-time), then backup (dispatch-time, via callback).
    charged_providers = [c[0] for c in charge_calls]
    assert primary_marker in charged_providers
    assert backup_name in charged_providers
    primary_idx = charged_providers.index(primary_marker)
    backup_idx = charged_providers.index(backup_name)
    assert backup_idx > primary_idx
    # Backup ts must be > primary ts (dispatch happens later in time).
    primary_ts = charge_calls[primary_idx][1]
    backup_ts = charge_calls[backup_idx][1]
    assert backup_ts >= primary_ts


def test_hedged_request_falls_back_after_both_legs_429(monkeypatch) -> None:
    runner, rec = _build_runner(policy_names=["budget_range_p100_hedge"])
    policy = runner.policies["budget_range_p100_hedge"]
    providers = list(policy.states)
    primary, backup, fallback = providers[:3]
    calls: list[str] = []

    for provider in providers:
        for _ in range(10):
            policy.add_sample(provider, time.time(), 800.0)

    monkeypatch.setattr(
        policy,
        "route",
        lambda now, ctx: RoutingDecision(primary=primary, notes="hedge_429_test"),
    )
    monkeypatch.setattr(
        "experiments.real_evaluation.runner.hedge_checkpoints_for_slo",
        lambda slo_sec: (0.0,),
    )
    checkpoint_decision_cls = __import__(
        "experiments.real_evaluation.policies",
        fromlist=["CheckpointHedgeDecision"],
    ).CheckpointHedgeDecision

    def checkpoint_and_charge(**kwargs):
        capacity_id = policy.charge_capacity(
            backup,
            kwargs["now"],
            kwargs["expected_service_sec"],
        )
        return (
            checkpoint_decision_cls(
                backup=backup,
                elapsed_sec=kwargs["elapsed_sec"],
                success_probability=0.99,
            ),
            capacity_id,
        )

    monkeypatch.setattr(
        policy,
        "checkpoint_backup_and_charge_capacity",
        checkpoint_and_charge,
    )

    def fallback_candidates(
        now: float,
        ctx: Any,
        *,
        excluded: set[str],
    ) -> list[str]:
        del now, ctx
        assert {primary, backup}.issubset(excluded)
        return [fallback]

    monkeypatch.setattr(policy, "rate_limit_fallback_candidates", fallback_candidates)

    def fake_send_via_transport(
        provider: str,
        prompt: str,
        max_tokens: int,
        timeout: int,
        ttft_event: threading.Event | None,
        ttft_info: dict[str, Any] | None,
        cancel_event: threading.Event | None = None,
    ) -> SingleRequestResult:
        del prompt, max_tokens, timeout, cancel_event
        calls.append(provider)
        if provider in {primary, backup}:
            if ttft_info is not None:
                ttft_info["status"] = "HTTP 429"
            if ttft_event is not None:
                ttft_event.set()
            return SingleRequestResult(
                ttft_ms=-1.0,
                e2e_ms=1.0,
                status="HTTP 429",
                provider=provider,
                http_status=429,
                rate_limited=True,
                start_ts=time.time(),
            )
        return SingleRequestResult(
            ttft_ms=100.0,
            e2e_ms=150.0,
            status="success",
            provider=provider,
            prompt_tokens=10,
            completion_tokens=8,
            billed_cost_usd=0.01,
            start_ts=time.time(),
            first_token_ts=time.time(),
        )

    monkeypatch.setattr(runner, "_send_via_transport", fake_send_via_transport)

    runner._dispatch_one(
        policy=policy,
        req=TraceRequest(arrival_time_sec=0.0, prompt="x", prompt_tokens=10, max_tokens=8),
        req_index=0,
    )

    assert calls[:3] == [primary, backup, fallback]
    row = rec._rows[0]
    assert row.primary_provider == primary
    assert row.actual_provider == fallback
    assert row.status == "success"
    assert row.rate_limited is True
    assert row.retry_count == 2
    rec.close()


def test_non_hedged_greedy_cost_falls_back_to_next_provider_on_429(monkeypatch) -> None:
    runner, rec = _build_runner(policy_names=["greedy_cost"])
    policy = runner.policies["greedy_cost"]
    calls: list[str] = []

    def fake_send_via_transport(
        provider: str,
        prompt: str,
        max_tokens: int,
        timeout: int,
        ttft_event: threading.Event | None,
        ttft_info: dict[str, Any] | None,
        cancel_event: threading.Event | None = None,
    ) -> SingleRequestResult:
        del prompt, max_tokens, timeout, ttft_event, ttft_info, cancel_event
        calls.append(provider)
        if provider == "Featherless_SC":
            return SingleRequestResult(
                ttft_ms=-1.0,
                e2e_ms=1.0,
                status="HTTP 429",
                provider=provider,
                http_status=429,
                rate_limited=True,
                start_ts=time.time(),
            )
        return SingleRequestResult(
            ttft_ms=100.0,
            e2e_ms=150.0,
            status="success",
            provider=provider,
            prompt_tokens=10,
            completion_tokens=8,
            billed_cost_usd=0.01,
            start_ts=time.time(),
            first_token_ts=time.time(),
        )

    monkeypatch.setattr(runner, "_send_via_transport", fake_send_via_transport)

    runner._dispatch_one(
        policy=policy,
        req=TraceRequest(arrival_time_sec=0.0, prompt="x", prompt_tokens=10, max_tokens=8),
        req_index=0,
    )

    assert calls[:2] == ["Featherless_SC", "Chutes_SQ"]
    assert runner._cost_per_policy["greedy_cost"] == 0.01
    assert runner._total_cost_usd == 0.01
    row = rec._rows[0]
    assert row.primary_provider == "Featherless_SC"
    assert row.actual_provider == "Chutes_SQ"
    assert row.status == "success"
    assert row.rate_limited is True
    assert row.retry_count == 1
    rec.close()


def test_prepare_dispatch_holds_concurrency_capacity_until_release() -> None:
    runner, rec = _build_runner(policy_names=["greedy_cost"])
    policy = runner.policies["greedy_cost"]
    req = TraceRequest(arrival_time_sec=0.0, prompt="x", prompt_tokens=10, max_tokens=8)

    first = runner._prepare_dispatch(policy, req, req_index=0)
    second = runner._prepare_dispatch(policy, req, req_index=1)

    assert first is not None
    assert second is not None
    assert first.decision.primary == "Featherless_SC"
    assert second.decision.primary != "Featherless_SC"

    policy.release_capacity(first.decision.primary, first.primary_capacity_id, time.time())
    policy.release_capacity(second.decision.primary, second.primary_capacity_id, time.time())
    rec.close()


def test_prepare_dispatch_reroutes_after_stale_concurrency_choice(monkeypatch) -> None:
    runner, rec = _build_runner(policy_names=["greedy_cost"])
    policy = runner.policies["greedy_cost"]
    req = TraceRequest(arrival_time_sec=0.0, prompt="x", prompt_tokens=10, max_tokens=8)

    first = runner._prepare_dispatch(policy, req, req_index=0)
    assert first is not None
    assert first.decision.primary == "Featherless_SC"

    original_route = policy.route
    calls = 0

    def stale_then_current(now, ctx):
        nonlocal calls
        calls += 1
        if calls == 1:
            return RoutingDecision(primary="Featherless_SC", notes="stale_concurrency_view")
        return original_route(now, ctx)

    monkeypatch.setattr(policy, "route", stale_then_current)

    second = runner._prepare_dispatch(policy, req, req_index=1)

    assert second is not None
    assert calls >= 2
    assert second.decision.primary != "Featherless_SC"

    policy.release_capacity(first.decision.primary, first.primary_capacity_id, time.time())
    policy.release_capacity(second.decision.primary, second.primary_capacity_id, time.time())
    rec.close()


def test_dispatch_uses_trace_cached_input_tokens_for_routing_diagnostics(
    monkeypatch,
) -> None:
    runner, rec = _build_runner(
        policy_names=["budget_range_p100"],
        prefix_cache_routing=True,
    )
    policy = runner.policies["budget_range_p100"]
    provider = next(spec.name for spec in runner.inventory.providers if spec.tier == "api")

    monkeypatch.setattr(
        policy,
        "route",
        lambda now, ctx: RoutingDecision(primary=provider, notes="cache_test"),
    )

    def fake_send_via_transport(
        provider: str,
        prompt: str,
        max_tokens: int,
        timeout: int,
        ttft_event: threading.Event | None,
        ttft_info: dict[str, Any] | None,
        cancel_event: threading.Event | None = None,
    ) -> SingleRequestResult:
        return SingleRequestResult(
            ttft_ms=100.0,
            e2e_ms=150.0,
            status="success",
            provider=provider,
            prompt_tokens=20,
            completion_tokens=5,
            billed_cost_usd=0.01,
            start_ts=time.time(),
            first_token_ts=time.time(),
            cache_read_tokens_observed=7,
            cost_source="reported",
        )

    monkeypatch.setattr(runner, "_send_via_transport", fake_send_via_transport)

    runner._dispatch_one(
        policy=policy,
        req=TraceRequest(
            arrival_time_sec=0.0,
            prompt="x",
            prompt_tokens=20,
            max_tokens=5,
            prefix_id="conv-1",
            trace_cached_input_tokens=12,
        ),
        req_index=0,
    )

    row = rec._rows[0]
    assert row.primary_cached_input_tokens == 12
    assert row.primary_observed_cached_input_tokens == 7
    assert row.cost_source == "reported"
    assert row.billed_cost_usd == 0.01
    rec.close()


def test_dispatch_treats_missing_trace_cache_field_as_cold_miss(monkeypatch) -> None:
    runner, rec = _build_runner(
        policy_names=["budget_range_p100"],
        prefix_cache_routing=True,
    )
    policy = runner.policies["budget_range_p100"]
    provider = next(spec.name for spec in runner.inventory.providers if spec.tier == "api")

    monkeypatch.setattr(
        policy,
        "route",
        lambda now, ctx: RoutingDecision(primary=provider, notes="cache_test"),
    )

    def fake_send_via_transport(
        provider: str,
        prompt: str,
        max_tokens: int,
        timeout: int,
        ttft_event: threading.Event | None,
        ttft_info: dict[str, Any] | None,
        cancel_event: threading.Event | None = None,
    ) -> SingleRequestResult:
        return SingleRequestResult(
            ttft_ms=100.0,
            e2e_ms=150.0,
            status="success",
            provider=provider,
            prompt_tokens=20,
            completion_tokens=5,
            billed_cost_usd=0.01,
            start_ts=time.time(),
            first_token_ts=time.time(),
            cost_source="reported",
        )

    monkeypatch.setattr(runner, "_send_via_transport", fake_send_via_transport)

    runner._dispatch_one(
        policy=policy,
        req=TraceRequest(
            arrival_time_sec=0.0,
            prompt="x",
            prompt_tokens=20,
            max_tokens=5,
            prefix_id="conv-1",
            trace_cached_input_tokens=None,
        ),
        req_index=0,
    )

    row = rec._rows[0]
    assert row.primary_cached_input_tokens == 0
    rec.close()
