"""Installed-library surface tests for the RouteWise facade."""

from __future__ import annotations

import ast
from pathlib import Path

import llm_routewise as rw


def test_top_level_exports_are_the_public_surface() -> None:
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


def test_package_version_matches_project_metadata() -> None:
    pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    in_project_table = False
    for line in pyproject.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "[project]":
            in_project_table = True
            continue
        if in_project_table and stripped.startswith("["):
            break
        if in_project_table and stripped.startswith("version ="):
            project_version = ast.literal_eval(stripped.split("=", 1)[1].strip())
            break
    else:
        raise AssertionError("[project].version is missing from pyproject.toml")

    assert rw.__version__ == project_version
