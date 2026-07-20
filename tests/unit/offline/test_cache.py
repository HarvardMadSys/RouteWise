"""Tests for repository-only offline dataset caching."""

from __future__ import annotations

import logging
import sys

import pytest

from llm_routewise.offline import cache
from llm_routewise.offline.schemas import Request


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")


def _request() -> Request:
    return Request(
        id=1,
        timestamp=2,
        request_tokens=3,
        response_tokens=4,
        total_tokens=7,
    )


def test_dataset_cache_roundtrip() -> None:
    cache.save_dataset_cache("unit", [_request()])

    assert cache.load_cached_dataset("unit") == [_request()]


def test_old_routewise_pickle_is_treated_as_cache_miss(
    caplog: pytest.LogCaptureFixture,
    monkeypatch,
) -> None:
    cache_path = cache.get_dataset_cache_path("legacy")
    cache_path.write_bytes(b"croutewise.offline.schemas\nRequest\n.")
    monkeypatch.setitem(sys.modules, "routewise", None)

    with caplog.at_level(logging.WARNING):
        assert cache.load_cached_dataset("legacy") is None

    assert cache_path.exists()
    assert "Ignoring unreadable dataset cache" in caplog.text


def test_corrupt_pickle_is_treated_as_cache_miss() -> None:
    cache_path = cache.get_dataset_cache_path("corrupt")
    cache_path.write_bytes(b"not a pickle")

    assert cache.load_cached_dataset("corrupt") is None
    assert cache_path.exists()


def test_unexpected_pickle_errors_propagate(monkeypatch) -> None:
    cache_path = cache.get_dataset_cache_path("unexpected")
    cache_path.touch()

    def fail(_handle):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(cache.pickle, "load", fail)

    with pytest.raises(RuntimeError, match="unexpected"):
        cache.load_cached_dataset("unexpected")
