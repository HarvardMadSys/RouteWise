"""Latency-layer simulator section planning module."""

from __future__ import annotations

SECTION_NAME = "latency-layer"


def list_scenarios() -> tuple[str, ...]:
    """Return scenario names once this section is implemented."""
    return ()


def make_scenarios() -> dict[str, object]:
    """Return scenarios once this section is implemented."""
    return {}


__all__ = ["SECTION_NAME", "list_scenarios", "make_scenarios"]
