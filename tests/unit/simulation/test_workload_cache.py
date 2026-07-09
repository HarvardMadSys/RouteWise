"""Tests for section-simulator compact workload caching."""

from __future__ import annotations

import json

from experiments.simulation import common
from routewise.schemas import Request


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_load_workload_writes_compact_cache_next_to_resolved_source(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    source = scratch / "burstgpt_30d.jsonl"
    _write_jsonl(
        source,
        [
            {
                "arrived_at": 100.0,
                "session_id": "s1",
                "num_prefill_tokens": 10,
                "num_decode_tokens": 20,
                "model": "qwen",
                "prompt_text": "large prompt omitted from simulator cache",
                "response_text": "large response omitted from simulator cache",
                "sharegpt_conversation_id": "c1",
                "sharegpt_turn_index": 0,
                "log_type": "chat",
                "elapsed_time_sec": 1.2,
            },
            {
                "arrived_at": 105.0,
                "session_id": "s2",
                "num_prefill_tokens": 11,
                "num_decode_tokens": 21,
                "model": "qwen",
            },
        ],
    )
    repo_data = tmp_path / "repo" / "data"
    repo_data.mkdir(parents=True)
    link = repo_data / "burstgpt_30d.jsonl"
    link.symlink_to(source)
    monkeypatch.setitem(common._WORKLOAD_PATHS, "unit_cache", link)
    common._load_cached_workload.cache_clear()

    requests = common.load_workload(dataset="unit_cache")

    cache_path = source.with_name(f"{source.name}.simcache.pkl")
    manifest_path = source.with_name(f"{source.name}.simcache.manifest.json")
    assert cache_path.exists()
    assert manifest_path.exists()
    assert [request.id for request in requests] == [0, 1]
    assert [request.timestamp for request in requests] == [0.0, 5.0]
    assert requests[0].metadata["session_id"] == "s1"
    assert requests[0].metadata["sharegpt_conversation_id"] == "c1"
    assert "prompt_text" not in requests[0].metadata
    assert "response_text" not in requests[0].metadata


def test_load_workload_uses_cache_for_truncation_without_reparsing(tmp_path, monkeypatch):
    source = tmp_path / "burstgpt_30d.jsonl"
    _write_jsonl(
        source,
        [
            {
                "arrived_at": 100.0 + index,
                "session_id": f"s{index}",
                "num_prefill_tokens": 10 + index,
                "num_decode_tokens": 20 + index,
            }
            for index in range(4)
        ],
    )
    monkeypatch.setitem(common._WORKLOAD_PATHS, "unit_truncate", source)
    common._load_cached_workload.cache_clear()

    assert len(common.load_workload(dataset="unit_truncate")) == 4
    assert len(common.load_workload(dataset="unit_truncate", max_requests=2)) == 2
    by_duration = common.load_workload(dataset="unit_truncate", duration_sec=1.5)
    assert [request.metadata["session_id"] for request in by_duration] == ["s0", "s1"]


def test_truncated_smoke_load_does_not_build_full_cache(tmp_path, monkeypatch):
    source = tmp_path / "burstgpt_30d.jsonl"
    _write_jsonl(
        source,
        [
            {
                "arrived_at": 100.0 + index,
                "session_id": f"s{index}",
                "num_prefill_tokens": 10 + index,
                "num_decode_tokens": 20 + index,
            }
            for index in range(4)
        ],
    )
    monkeypatch.setitem(common._WORKLOAD_PATHS, "unit_smoke", source)
    common._load_cached_workload.cache_clear()

    assert len(common.load_workload(dataset="unit_smoke", max_requests=2)) == 2
    assert not source.with_name(f"{source.name}.simcache.pkl").exists()


def test_load_workload_supports_dataset_cache_traces(monkeypatch):
    cached_requests = (
        Request(id=7, timestamp=1000.0, request_tokens=10, response_tokens=20),
        Request(id=8, timestamp=1005.0, request_tokens=11, response_tokens=21),
        Request(id=9, timestamp=1009.0, request_tokens=12, response_tokens=22),
    )
    monkeypatch.setattr(common, "_TRACE_CACHE_WORKLOADS", ("cached_unit",))
    monkeypatch.setattr(
        "experiments.simulation.dataset_cache.load_cached",
        lambda dataset: cached_requests,
    )
    common._load_cached_trace_workload.cache_clear()

    requests = common.load_workload(dataset="cached_unit", duration_sec=5.0)

    assert [request.id for request in requests] == [0, 1]
    assert [request.timestamp for request in requests] == [0.0, 5.0]
    assert [request.request_tokens for request in requests] == [10, 11]


def test_trace_workload_cache_rebuilds_old_layout_pickles(monkeypatch):
    """Old ``rwsim`` pickle caches are rebuilt under the new package layout."""
    cached_requests = (
        Request(id=11, timestamp=2000.0, request_tokens=30, response_tokens=40),
        Request(id=12, timestamp=2007.0, request_tokens=31, response_tokens=41),
    )
    load_calls = 0
    build_calls = []

    def load_cached(dataset):
        nonlocal load_calls
        load_calls += 1
        if load_calls == 1:
            raise ModuleNotFoundError("No module named 'rwsim.schemas'", name="rwsim.schemas")
        return cached_requests

    def build_cache(dataset, *, force):
        build_calls.append((dataset, force))

    monkeypatch.setattr(common, "_TRACE_CACHE_WORKLOADS", ("cached_unit",))
    monkeypatch.setattr("experiments.simulation.dataset_cache.load_cached", load_cached)
    monkeypatch.setattr("experiments.simulation.dataset_cache.build_cache", build_cache)
    common._load_cached_trace_workload.cache_clear()

    requests = common.load_workload(dataset="cached_unit")

    assert build_calls == [("cached_unit", True)]
    assert load_calls == 2
    assert [request.id for request in requests] == [0, 1]
    assert [request.timestamp for request in requests] == [0.0, 7.0]
    assert [request.request_tokens for request in requests] == [30, 31]
