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

from experiments.real_evaluation.inventory import ProviderState, load_inventory
from experiments.real_evaluation.policies import (
    OR_AUTO_SENTINEL,
    OR_SORT_SENTINEL_TO_MODE,
    RoutingDecision,
)
from experiments.real_evaluation.recorder import Recorder
from experiments.real_evaluation.runner import (
    WARMUP_PROBE_PROMPT,
    RealExperimentRunner,
    TraceRequest,
    load_trace_jsonl,
)
from experiments.real_evaluation.transports import SingleRequestResult

_INVENTORY_PATH = (
    "experiments/real_evaluation/data/joint_minimax_m25_online.json"
)


def _build_runner(
    policy_names: list[str] | None = None,
    *,
    prefix_cache_routing: bool = False,
) -> tuple[RealExperimentRunner, Recorder]:
    inventory = load_inventory(_INVENTORY_PATH)
    rec = Recorder(tempfile.mkdtemp())
    runner = RealExperimentRunner(
        inventory=inventory,
        policy_names=policy_names or ["cheapest_fixed"],
        recorder=rec,
        slo_ms=inventory.primary_slo_ms,
        prefix_cache_routing=prefix_cache_routing,
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
    assert [(w.requests, w.window_sec) for w in chutes.quota_windows] == [
        (5000, 86400.0)
    ]

    minimax = specs["MiniMax_SQ"]
    assert minimax.subscription_plan == "minimax_subscription_plus"
    assert [(w.requests, w.window_sec) for w in minimax.quota_windows] == [
        (4500, 18000.0),
        (45000, 604800.0),
    ]
    state = ProviderState.from_spec(minimax)
    assert state.quota is not None
    assert hasattr(state.quota, "windows")


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
                    },
                    {
                        "name": "OR_DeepInfra",
                        "tier": "api",
                        "transport": "openrouter",
                        "model": "test/model",
                        "provider_hint": "DeepInfra",
                        "input_price_per_m": 0.2,
                        "output_price_per_m": 1.2,
                    },
                ],
            }
        )
    )

    inventory = load_inventory(inventory_path)

    assert [spec.name for spec in inventory.providers] == ["Chutes_SQ", "OR_Chutes"]
    assert inventory.openrouter_provider_only == ("Chutes",)


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


def test_warmup_probes_round_robin_by_provider(monkeypatch) -> None:
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
    assert seen == providers + providers
    rec.close()


def test_profile_bootstrap_guard_skips_profile_free_policies() -> None:
    runner, rec = _build_runner(policy_names=["openrouter_auto", "sort_price"])

    runner.validate_profile_bootstrap(min_success_samples=5)

    rec.close()


def test_periodic_profile_probe_runs_during_replay(monkeypatch) -> None:
    runner, rec = _build_runner(policy_names=["openrouter_auto"])
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
    """Each sentinel ``__openrouter_sort_<mode>__`` must dispatch with the
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


def test_or_sentinels_inherit_inventory_provider_filters() -> None:
    """OpenRouter auto/sort sentinels should apply the inventory-level
    provider filter instead of routing over the full OpenRouter pool."""
    runner, _ = _build_runner()
    runner.inventory.openrouter_provider_only = ("Chutes", "DeepInfra")
    runner.inventory.openrouter_provider_ignore = ("BadProvider",)
    captured: dict[str, tuple[str, ...] | str | None] = {}

    def fake_send(self, prompt, max_tokens, timeout, ttft_event, ttft_info, cancel_event=None):
        captured["sort_mode"] = self.cfg.sort_mode
        captured["provider_only"] = self.cfg.provider_only
        captured["provider_ignore"] = self.cfg.provider_ignore
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
        runner._send_via_transport(
            provider=OR_AUTO_SENTINEL,
            prompt="x",
            max_tokens=8,
            timeout=5,
            ttft_event=None,
            ttft_info=None,
        )
        assert captured["sort_mode"] is None
        assert captured["provider_only"] == ("Chutes", "DeepInfra")
        assert captured["provider_ignore"] == ("BadProvider",)

        runner._send_via_transport(
            provider="__openrouter_sort_latency__",
            prompt="x",
            max_tokens=8,
            timeout=5,
            ttft_event=None,
            ttft_info=None,
        )
        assert captured["sort_mode"] == "latency"
        assert captured["provider_only"] == ("Chutes", "DeepInfra")
        assert captured["provider_ignore"] == ("BadProvider",)


def test_dispatch_one_charges_backup_at_dispatch_time(monkeypatch) -> None:
    """When a hedge is triggered through the runner, backup capacity
    must be charged via the executor's ``on_backup_dispatch`` callback —
    i.e. before the backup completes, not after the hedged request
    returns."""
    runner, _ = _build_runner(policy_names=["budget_range_p100_hedge"])
    policy = runner.policies["budget_range_p100_hedge"]

    # Seed every provider with profile data so the LP picks something.
    now = time.time()
    for spec in runner.inventory.providers:
        for _ in range(20):
            policy.add_sample(spec.name, now, 800.0)

    charge_calls: list[tuple[str, float]] = []
    original_charge = policy.charge_capacity

    def tracking_charge(provider: str, ts: float, expected_service_sec: float) -> None:
        charge_calls.append((provider, ts))
        original_charge(provider, ts, expected_service_sec)

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
            ttft_info.update(
                ttft_ms=200.0, first_token_ts=time.time(), status="success"
            )
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
        lambda now, ctx: __import__("experiments.real_evaluation.policies", fromlist=["RoutingDecision"]).RoutingDecision(
            primary=primary_marker, notes="test"
        ),
    )
    monkeypatch.setattr(
        "experiments.real_evaluation.runner.select_safe_cheapest_backup",
        lambda **kwargs: backup_name,
    )
    monkeypatch.setattr(
        "experiments.real_evaluation.runner.compute_hedge_time_sec",
        lambda **kwargs: 0.05,
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


def test_coalesced_replay_executes_identical_action_once(monkeypatch) -> None:
    """Coalescing is a physical-execution optimization only: two policies
    that pick the same provider share one API call, while each policy still
    receives its own virtual cost/profile/accounting row."""
    runner, rec = _build_runner(policy_names=["cheapest_fixed", "fastest_fixed"])
    provider = runner.inventory.providers[0].name

    for policy in runner.policies.values():
        monkeypatch.setattr(
            policy,
            "route",
            lambda now, ctx, _provider=provider: RoutingDecision(
                primary=_provider,
                notes="coalesce_test",
            ),
        )

    send_count = 0

    def fake_send_via_transport(
        provider: str,
        prompt: str,
        max_tokens: int,
        timeout: int,
        ttft_event: threading.Event | None,
        ttft_info: dict[str, Any] | None,
    cancel_event: threading.Event | None = None,
    ) -> SingleRequestResult:
        nonlocal send_count
        send_count += 1
        if ttft_info is not None:
            ttft_info.update(
                ttft_ms=100.0, first_token_ts=time.time(), status="success"
            )
        if ttft_event is not None:
            ttft_event.set()
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

    from experiments.real_evaluation.runner import TraceRequest

    runner.replay(
        [
            TraceRequest(
                arrival_time_sec=0.0,
                prompt="x",
                prompt_tokens=10,
                max_tokens=8,
            )
        ],
        speedup=100.0,
        coalesce_identical_actions=True,
    )
    rec.close()

    assert send_count == 1
    assert runner._cost_per_policy["cheapest_fixed"] == 0.01
    assert runner._cost_per_policy["fastest_fixed"] == 0.01
    assert runner._total_cost_usd == 0.01


def test_dispatch_updates_policy_local_prefix_cache_and_records_diagnostics(
    monkeypatch,
) -> None:
    runner, rec = _build_runner(
        policy_names=["budget_range_p100"],
        prefix_cache_routing=True,
    )
    policy = runner.policies["budget_range_p100"]
    provider = next(
        spec.name for spec in runner.inventory.providers if spec.tier == "api"
    )

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
        ),
        req_index=0,
    )

    assert policy.provider_prefix_cache[provider]["conv-1"] == 25
    row = rec._rows[0]
    assert row.primary_cached_input_tokens == 0
    assert row.primary_observed_cached_input_tokens == 7
    assert row.cost_source == "reported"
    assert row.billed_cost_usd == 0.01
    rec.close()
