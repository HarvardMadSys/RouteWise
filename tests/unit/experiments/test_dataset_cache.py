"""Pytest coverage for the dataset cache module.

Tests the cache build / load / status / clear lifecycle using a small
synthetic dataset to avoid depending on the real 14 GB trace files.
Also covers the source-fingerprint manifest and staleness detection.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from unittest.mock import patch

import pytest

from rwsim.schemas import Request

from experiments.simulation.dataset_cache import (
    CacheStalenessError,
    build_cache,
    cache_status,
    clear_cache,
    ensure_caches,
    load_cached,
    verify_cache,
    _manifest_path,
    _read_manifest,
    _source_fingerprint,
)
from experiments.simulation.lp_budget_eval import (
    TRACE_WORKLOAD_DATASETS,
    _dataset_cache_path,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _synthetic_source(tmp_path: Path, monkeypatch):
    """Create a tiny JSONL file and patch the resolution functions so
    ``build_cache`` sees it as the "sharegpt" raw source."""
    jsonl = tmp_path / "sharegpt_test.jsonl"
    lines = []
    for i in range(50):
        lines.append(
            f'{{"arrived_at": {1000 + i}, "num_prefill_tokens": {10 + i}, '
            f'"num_decode_tokens": {20 + i}}}'
        )
    jsonl.write_text("\n".join(lines))

    cache_root = tmp_path / "cache"

    # Patch the two functions / constants that control path resolution
    # in lp_budget_eval so the cache module picks up our fake source.
    monkeypatch.setattr(
        "experiments.simulation.lp_budget_eval._resolve_trace_dataset_path",
        lambda name: jsonl if name == "sharegpt" else None,
    )
    monkeypatch.setattr(
        "experiments.simulation.lp_budget_eval._DATASET_CACHE_ROOT",
        cache_root,
    )
    # Also patch the dataset_cache module's re-import of these.
    monkeypatch.setattr(
        "experiments.simulation.dataset_cache._resolve_trace_dataset_path",
        lambda name: jsonl if name == "sharegpt" else None,
    )
    monkeypatch.setattr(
        "experiments.simulation.dataset_cache._DATASET_CACHE_ROOT",
        cache_root,
    )
    return tmp_path, cache_root, jsonl


# ---------------------------------------------------------------------------
# Core lifecycle
# ---------------------------------------------------------------------------


def test_build_and_load_roundtrip(_synthetic_source):
    """Build cache from synthetic JSONL, then load and verify contents."""
    _, cache_root, _ = _synthetic_source
    path = build_cache("sharegpt", force=True)

    assert path.exists()
    assert path.suffix == ".pkl"
    assert path.parent == cache_root

    requests = load_cached("sharegpt")
    assert len(requests) == 50
    assert isinstance(requests[0], Request)
    assert requests[0].request_tokens == 10
    assert requests[-1].request_tokens == 59


def test_build_skips_existing(_synthetic_source, capsys):
    """build_cache without force=True skips if cache exists."""
    build_cache("sharegpt", force=True)  # first build
    build_cache("sharegpt", force=False)  # should skip
    captured = capsys.readouterr()
    assert "[skip]" in captured.out


def test_build_force_rebuilds(_synthetic_source):
    """build_cache with force=True overwrites existing cache."""
    path1 = build_cache("sharegpt", force=True)
    mtime1 = path1.stat().st_mtime_ns
    path2 = build_cache("sharegpt", force=True)
    mtime2 = path2.stat().st_mtime_ns
    assert mtime2 >= mtime1


def test_build_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="Unknown dataset"):
        build_cache("not_a_dataset")


def test_load_cached_raises_if_no_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "experiments.simulation.dataset_cache._DATASET_CACHE_ROOT",
        tmp_path / "empty",
    )
    monkeypatch.setattr(
        "experiments.simulation.lp_budget_eval._DATASET_CACHE_ROOT",
        tmp_path / "empty",
    )
    with pytest.raises(FileNotFoundError, match="No cache"):
        load_cached("sharegpt")


def test_clear_cache(_synthetic_source):
    _, cache_root, _ = _synthetic_source
    build_cache("sharegpt", force=True)
    assert (cache_root / "sharegpt.pkl").exists()

    cleared = clear_cache("sharegpt")
    assert "sharegpt" in cleared
    assert not (cache_root / "sharegpt.pkl").exists()
    # Manifest should also be deleted.
    assert not (cache_root / "sharegpt.manifest.json").exists()


def test_clear_cache_noop_when_missing(_synthetic_source, capsys):
    cleared = clear_cache("sharegpt")
    assert cleared == []
    assert "[skip]" in capsys.readouterr().out


def test_cache_status_keys():
    """cache_status returns an entry for every known trace dataset."""
    status = cache_status()
    for name in TRACE_WORKLOAD_DATASETS:
        assert name in status
        entry = status[name]
        assert "cache_exists" in entry
        assert "source_exists" in entry


def test_cache_status_reflects_build(_synthetic_source):
    _, cache_root, _ = _synthetic_source
    status_before = cache_status()
    assert not status_before["sharegpt"]["cache_exists"]

    build_cache("sharegpt", force=True)
    status_after = cache_status()
    assert status_after["sharegpt"]["cache_exists"]
    assert status_after["sharegpt"]["cache_size_mb"] >= 0
    assert status_after["sharegpt"]["cache_valid"] is True


# ---------------------------------------------------------------------------
# Source manifest and staleness
# ---------------------------------------------------------------------------


def test_manifest_written_on_build(_synthetic_source):
    """build_cache writes a sidecar .manifest.json with SHA-256."""
    _, cache_root, _ = _synthetic_source
    build_cache("sharegpt", force=True)

    manifest = _read_manifest("sharegpt")
    assert manifest is not None
    assert "source_sha256" in manifest
    assert len(manifest["source_sha256"]) == 64  # hex SHA-256
    assert manifest["n_requests"] == 50
    assert manifest["source_size"] > 0


def test_verify_cache_valid(_synthetic_source):
    """verify_cache returns True when manifest matches source."""
    build_cache("sharegpt", force=True)
    assert verify_cache("sharegpt", quick=False) is True


def test_verify_cache_quick(_synthetic_source):
    """Quick verification (size + mtime only) should pass."""
    build_cache("sharegpt", force=True)
    assert verify_cache("sharegpt", quick=True) is True


def test_verify_cache_no_manifest(_synthetic_source):
    """Cache without manifest raises CacheStalenessError."""
    _, cache_root, _ = _synthetic_source
    build_cache("sharegpt", force=True)
    # Delete manifest to simulate legacy cache.
    _manifest_path("sharegpt").unlink()

    with pytest.raises(CacheStalenessError, match="no manifest"):
        verify_cache("sharegpt")


def test_verify_cache_size_mismatch(_synthetic_source):
    """If source file size changes, staleness is detected."""
    _, cache_root, jsonl = _synthetic_source
    build_cache("sharegpt", force=True)

    # Append data to change file size.
    with jsonl.open("a") as f:
        f.write('\n{"arrived_at": 9999, "num_prefill_tokens": 1, "num_decode_tokens": 1}\n')

    with pytest.raises(CacheStalenessError, match="size changed"):
        verify_cache("sharegpt", quick=True)


def test_verify_cache_sha256_mismatch(_synthetic_source):
    """If source content changes (same size), verify catches it.

    Writing new content also updates mtime, so the quick check (size +
    mtime) fires first.  Both quick and full should raise.
    """
    _, cache_root, jsonl = _synthetic_source
    build_cache("sharegpt", force=True)

    # Rewrite with same byte count but different content.
    content = jsonl.read_text()
    modified = content.replace('"arrived_at": 1000', '"arrived_at": 2000', 1)
    jsonl.write_text(modified)

    # Quick mode catches via mtime.
    with pytest.raises(CacheStalenessError, match="mtime changed"):
        verify_cache("sharegpt", quick=True)

    # Full mode also catches (mtime fires before SHA, but either is fine).
    with pytest.raises(CacheStalenessError):
        verify_cache("sharegpt", quick=False)


def test_build_auto_rebuilds_stale(_synthetic_source, capsys):
    """build_cache without force rebuilds if manifest shows staleness."""
    _, cache_root, jsonl = _synthetic_source
    build_cache("sharegpt", force=True)

    # Change source size to trigger staleness.
    with jsonl.open("a") as f:
        f.write('\n{"arrived_at": 9999, "num_prefill_tokens": 1, "num_decode_tokens": 1}\n')

    build_cache("sharegpt", force=False)
    captured = capsys.readouterr()
    assert "[stale]" in captured.out or "[build]" in captured.out

    # After rebuild, verify should pass.
    assert verify_cache("sharegpt", quick=True) is True


def test_verify_cache_missing_returns_false(_synthetic_source):
    """verify_cache returns False (not raises) when cache is absent."""
    assert verify_cache("sharegpt") is False


# ---------------------------------------------------------------------------
# ensure_caches (parallel runner entry point)
# ---------------------------------------------------------------------------


def test_ensure_caches_builds_missing(_synthetic_source, capsys):
    """ensure_caches builds any missing caches."""
    ensure_caches(["sharegpt"])
    captured = capsys.readouterr()
    assert "[build]" in captured.out

    # Second call should find valid cache.
    ensure_caches(["sharegpt"])
    captured2 = capsys.readouterr()
    assert "[ok]" in captured2.out


def test_ensure_caches_skips_synthetic(_synthetic_source):
    """'synthetic' is silently ignored."""
    ensure_caches(["synthetic"])  # should not raise


def test_ensure_caches_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown dataset"):
        ensure_caches(["nonexistent"])
