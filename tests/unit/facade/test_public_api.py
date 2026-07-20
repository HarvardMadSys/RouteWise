"""Installed-library surface tests for the RouteWise facade."""

from __future__ import annotations

import llm_routewise as rw


def test_top_level_exports_are_the_api_v1_surface() -> None:
    assert set(rw.__all__) == {
        "Attempt",
        "Candidate",
        "Decision",
        "NoProviderError",
        "OutcomeError",
        "Provider",
        "RouteOnceResult",
        "RouteWiseError",
        "Router",
        "StatsSnapshot",
        "Tuning",
        "ValidationError",
        "route_once",
    }


def test_package_version_matches_preview_release() -> None:
    assert rw.__version__ == "0.1.0"
