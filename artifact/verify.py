"""Compare produced summaries against expected values with tolerances.

`expected.yaml` is read only here. Each entry names a produced JSON file and a
list of checks; a check extracts one numeric value via a dotted path (list
selectors use `[key=value]`) and compares it under an absolute or relative
tolerance. Smoke checks on the fixed fixture use tight relative tolerances;
paper-number checks use justified per-metric tolerances.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

ARTIFACT_DIR = Path(__file__).resolve().parent
ROOT_DIR = ARTIFACT_DIR.parent
EXPECTED_PATH = ARTIFACT_DIR / "expected.yaml"
MANIFEST_PATH = ARTIFACT_DIR / "manifest.yaml"

_SELECTOR_RE = re.compile(r"^(?P<name>[^\[\]]*)(?:\[(?P<key>[^=\]]+)=(?P<value>[^\]]+)\])?$")


class VerificationError(RuntimeError):
    """A check could not be evaluated (missing file, bad path, bad spec)."""


def _resolve_step(node: Any, step: str, *, context: str) -> Any:
    match = _SELECTOR_RE.match(step)
    if match is None:
        raise VerificationError(f"bad path step {step!r} in {context}")
    name, key, value = match.group("name"), match.group("key"), match.group("value")
    if name:
        if not isinstance(node, dict) or name not in node:
            raise VerificationError(f"missing key {name!r} in {context}")
        node = node[name]
    if key is not None:
        if not isinstance(node, list):
            raise VerificationError(f"selector on non-list at {step!r} in {context}")
        matches = [item for item in node if isinstance(item, dict) and str(item.get(key)) == value]
        if len(matches) != 1:
            raise VerificationError(
                f"selector [{key}={value}] matched {len(matches)} items in {context}"
            )
        node = matches[0]
    return node


def extract(payload: Any, path: str, *, context: str) -> float:
    """Extract a numeric value at a dotted path with optional list selectors."""
    node = payload
    for step in path.split("."):
        node = _resolve_step(node, step, context=context)
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        raise VerificationError(f"value at {path!r} is not numeric in {context}")
    return float(node)


def _within(actual: float, check: dict[str, Any]) -> tuple[bool, str]:
    expected = float(check["expected"])
    if "absolute_tolerance" in check:
        tolerance = float(check["absolute_tolerance"])
        return abs(actual - expected) <= tolerance, f"abs tol {tolerance:g}"
    tolerance = float(check.get("relative_tolerance", 1e-9))
    scale = max(abs(expected), 1e-12)
    return abs(actual - expected) / scale <= tolerance, f"rel tol {tolerance:g}"


def load_expected() -> dict[str, Any]:
    with EXPECTED_PATH.open(encoding="utf-8") as handle:
        expected = yaml.safe_load(handle) or {}
    if not isinstance(expected, dict):
        raise VerificationError(f"malformed expectations file: {EXPECTED_PATH}")
    return expected


def _manifest_target_names() -> set[str]:
    try:
        with MANIFEST_PATH.open(encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle) or {}
        return set(manifest.get("targets", {}))
    except OSError:
        return set()


def verify_targets(target_names: list[str] | None = None) -> int:
    """Verify the named targets (all with expectations when omitted).

    Prints one line per check and a final summary; returns a process exit
    code (0 = all pass). A name that exists in the manifest but has no
    expectations yet is a warning; a name unknown to both files is an error.
    Missing output files count as target-level failures, separately from
    metric checks.
    """
    expected = load_expected()
    selected = target_names or list(expected)
    known_targets = _manifest_target_names()
    passed_checks = 0
    failed_checks = 0
    target_failures = 0
    for name in selected:
        entry = expected.get(name)
        if entry is None:
            if name in known_targets:
                print(f"[warn] {name}: target exists but has no expectations recorded yet")
                continue
            print(f"[FAIL] {name}: unknown target (not in the manifest or expected.yaml)")
            target_failures += 1
            continue
        source_path = ROOT_DIR / entry["source"]
        if not source_path.exists():
            print(f"[FAIL] {name}: missing output {entry['source']} (run `reproduce` first)")
            target_failures += 1
            continue
        with source_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        for check in entry.get("checks", ()):
            context = f"{name}:{entry['source']}"
            try:
                actual = extract(payload, check["path"], context=context)
            except VerificationError as exc:
                print(f"[FAIL] {name} {check['path']}: {exc}")
                failed_checks += 1
                continue
            ok, tolerance_note = _within(actual, check)
            status = "ok" if ok else "FAIL"
            print(
                f"[{status:>4}] {name} {check['path']}: "
                f"actual {actual:g} vs expected {check['expected']:g} ({tolerance_note})"
            )
            if ok:
                passed_checks += 1
            else:
                failed_checks += 1
    total_checks = passed_checks + failed_checks
    summary = f"verify: {passed_checks}/{total_checks} checks passed"
    if target_failures:
        summary += f"; {target_failures} target-level failure(s)"
    print(summary)
    return 1 if failed_checks or target_failures else 0
