"""Dispatch manifest targets to experiment entrypoints.

The runner reads `manifest.yaml` and invokes each target's entrypoint module
with the pinned arguments. It never reads `expected.yaml`; result checking
lives exclusively in `artifact.verify`.

Transitional note: dispatch currently reuses the `routewise_cli` section
registry while that package still exists; the registry moves here when the
old CLI is removed.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import yaml

ARTIFACT_DIR = Path(__file__).resolve().parent
ROOT_DIR = ARTIFACT_DIR.parent
MANIFEST_PATH = ARTIFACT_DIR / "manifest.yaml"

_RUNNABLE_STATUSES = ("runnable",)


class TargetError(RuntimeError):
    """A manifest target cannot be run in the current checkout state."""


def load_manifest() -> dict[str, Any]:
    """Load and lightly validate the artifact manifest."""
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict) or "targets" not in manifest:
        raise TargetError(f"malformed manifest: {MANIFEST_PATH}")
    return manifest


def list_targets() -> list[dict[str, Any]]:
    """Return manifest targets as display-ready rows."""
    manifest = load_manifest()
    rows = []
    for name, target in manifest["targets"].items():
        rows.append(
            {
                "name": name,
                "class": target.get("class", "?"),
                "status": target.get("status", "runnable"),
                "claim": target.get("claim", ""),
                "est_minutes": target.get("est_minutes"),
            }
        )
    return rows


def _check_preconditions(target: dict[str, Any]) -> None:
    for relative_path, hint in (target.get("requires") or {}).items():
        if not (ROOT_DIR / relative_path).exists():
            raise TargetError(f"missing input {relative_path!r}.\n  How to provide it: {hint}")


def run_target(name: str) -> int:
    """Run one manifest target and return its exit code."""
    manifest = load_manifest()
    try:
        target = manifest["targets"][name]
    except KeyError as exc:
        known = ", ".join(manifest["targets"])
        raise TargetError(f"unknown target {name!r}; expected one of: {known}") from exc

    status = target.get("status", "runnable")
    if status not in _RUNNABLE_STATUSES:
        raise TargetError(
            f"target {name!r} is not runnable (status: {status}). "
            f"{target.get('status_reason', '')}".strip()
        )

    _check_preconditions(target)
    module = importlib.import_module(target["entrypoint"])
    args = [str(item) for item in target.get("args", ())]
    # Manifest paths are repo-relative; entrypoints must resolve them against
    # the repo root no matter where the reviewer invoked the command from.
    previous_cwd = os.getcwd()
    os.chdir(ROOT_DIR)
    try:
        return int(module.main(args))
    finally:
        os.chdir(previous_cwd)


def targets_in_group(group: str) -> list[str]:
    """Return runnable target names whose class matches `group`."""
    manifest = load_manifest()
    return [
        name
        for name, target in manifest["targets"].items()
        if target.get("class") == group and target.get("status", "runnable") in _RUNNABLE_STATUSES
    ]
