"""Trace-replay loop must dispatch every request in the trace, even when
inter-arrival gaps exceed the per-iteration sleep cap.

Regression for the bug where ``for ... continue`` advanced the loop
counter while still in the wait window, silently dropping sparse arrivals.
"""

from __future__ import annotations

import tempfile
import threading

from experiments.real_evaluation.inventory import load_inventory
from experiments.real_evaluation.recorder import Recorder
from experiments.real_evaluation.runner import (
    RealExperimentRunner,
    TraceRequest,
)

_INVENTORY_PATH = (
    "experiments/real_evaluation/data/joint_minimax_m25_online.json"
)


def test_long_inter_arrival_gap_not_skipped() -> None:
    """Trace with a 20s gap (well over the 5s sleep cap) at high speedup
    should still dispatch both requests in order."""
    inventory = load_inventory(_INVENTORY_PATH)
    trace = [
        TraceRequest(
            arrival_time_sec=0.0, prompt="x", prompt_tokens=10, max_tokens=8
        ),
        TraceRequest(
            arrival_time_sec=20.0, prompt="x", prompt_tokens=10, max_tokens=8
        ),
    ]
    seen: list[int] = []

    with tempfile.TemporaryDirectory() as tmp:
        rec = Recorder(tmp)
        runner = RealExperimentRunner(
            inventory=inventory,
            policy_names=["budget_range_p100_hedge"],
            recorder=rec,
            slo_ms=inventory.primary_slo_ms,
        )
        runner._dispatch_one = lambda policy, req, idx: seen.append(idx)  # type: ignore[method-assign]
        # 200x speedup collapses the 20s gap to 0.1s wall-clock.
        runner.replay(trace, speedup=200.0, duration_sec=10.0)
        rec.close()

    assert seen == [0, 1]


def test_default_replay_runs_full_trace_for_each_policy() -> None:
    """Multiple policies are fair comparisons by default: every policy
    receives every trace request at the same replay arrival."""
    inventory = load_inventory(_INVENTORY_PATH)
    trace = [
        TraceRequest(
            arrival_time_sec=0.0, prompt="x", prompt_tokens=10, max_tokens=8
        ),
        TraceRequest(
            arrival_time_sec=0.01, prompt="x", prompt_tokens=10, max_tokens=8
        ),
    ]
    seen: list[tuple[str, int]] = []

    with tempfile.TemporaryDirectory() as tmp:
        rec = Recorder(tmp)
        runner = RealExperimentRunner(
            inventory=inventory,
            policy_names=["greedy_cost", "greedy_latency"],
            recorder=rec,
            slo_ms=inventory.primary_slo_ms,
        )
        runner._dispatch_one = (  # type: ignore[method-assign]
            lambda policy, req, idx: seen.append((policy.name, idx))
        )
        runner.replay(trace, speedup=200.0, duration_sec=10.0)
        rec.close()

    assert sorted(seen) == sorted([
        ("greedy_cost", 0),
        ("greedy_cost", 1),
        ("greedy_latency", 0),
        ("greedy_latency", 1),
    ])


def test_default_replay_starts_all_policies_for_same_arrival_concurrently() -> None:
    """A single trace arrival should fan out to all policies immediately,
    not wait for one policy's request to complete before starting the next."""
    inventory = load_inventory(_INVENTORY_PATH)
    trace = [
        TraceRequest(
            arrival_time_sec=0.0, prompt="x", prompt_tokens=10, max_tokens=8
        )
    ]
    entered: list[tuple[str, int]] = []
    entered_lock = threading.Lock()
    ready = threading.Semaphore(0)
    release = threading.Event()

    with tempfile.TemporaryDirectory() as tmp:
        rec = Recorder(tmp)
        runner = RealExperimentRunner(
            inventory=inventory,
            policy_names=["greedy_cost", "greedy_latency"],
            recorder=rec,
            slo_ms=inventory.primary_slo_ms,
        )

        def blocking_dispatch(policy, req, idx):  # type: ignore[no-untyped-def]
            with entered_lock:
                entered.append((policy.name, idx))
            ready.release()
            release.wait(timeout=5.0)

        runner._dispatch_one = blocking_dispatch  # type: ignore[method-assign]
        replay_thread = threading.Thread(
            target=runner.replay,
            args=(trace,),
            kwargs={"speedup": 1.0, "duration_sec": 10.0},
        )
        replay_thread.start()
        try:
            assert ready.acquire(timeout=5.0)
            assert ready.acquire(timeout=5.0)
            assert sorted(entered) == [
                ("greedy_cost", 0),
                ("greedy_latency", 0),
            ]
        finally:
            release.set()
            replay_thread.join(timeout=5.0)
            rec.close()


def test_duration_cap_stops_replay() -> None:
    """Replay halts cleanly when ``duration_sec`` is reached during a wait."""
    inventory = load_inventory(_INVENTORY_PATH)
    trace = [
        TraceRequest(
            arrival_time_sec=0.0, prompt="x", prompt_tokens=10, max_tokens=8
        ),
        # Far in the future — replay must terminate before reaching this.
        TraceRequest(
            arrival_time_sec=10_000.0,
            prompt="x",
            prompt_tokens=10,
            max_tokens=8,
        ),
    ]
    seen: list[int] = []

    with tempfile.TemporaryDirectory() as tmp:
        rec = Recorder(tmp)
        runner = RealExperimentRunner(
            inventory=inventory,
            policy_names=["budget_range_p100_hedge"],
            recorder=rec,
            slo_ms=inventory.primary_slo_ms,
        )
        runner._dispatch_one = lambda policy, req, idx: seen.append(idx)  # type: ignore[method-assign]
        runner.replay(trace, speedup=1.0, duration_sec=0.5)
        rec.close()

    # First request fires; the second should be cut off by duration cap.
    assert seen == [0]
