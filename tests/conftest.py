"""Shared test fixtures for repository-only external data."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def require_burstgpt_data() -> None:
    """Skip trace-backed tests when the optional BurstGPT artifact is absent."""

    path = Path(__file__).resolve().parents[1] / "data" / "burstgpt_30d.jsonl"
    if not path.is_file():
        pytest.skip("requires optional data/burstgpt_30d.jsonl")
