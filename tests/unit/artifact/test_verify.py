"""Unit tests for the artifact verification layer."""

from __future__ import annotations

import json
import textwrap

import pytest

from artifact import verify


@pytest.fixture
def isolated_files(tmp_path, monkeypatch):
    """Point verify at temp expected/manifest files and a temp output root."""
    expected_path = tmp_path / "expected.yaml"
    manifest_path = tmp_path / "manifest.yaml"
    monkeypatch.setattr(verify, "EXPECTED_PATH", expected_path)
    monkeypatch.setattr(verify, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(verify, "ROOT_DIR", tmp_path)
    return tmp_path, expected_path, manifest_path


def _write(path, content: str) -> None:
    path.write_text(textwrap.dedent(content), encoding="utf-8")


def test_unknown_target_fails(isolated_files, capsys):
    _, expected_path, manifest_path = isolated_files
    _write(expected_path, "")
    _write(manifest_path, "targets: {}\n")

    assert verify.verify_targets(["no-such-target"]) == 1
    out = capsys.readouterr().out
    assert "unknown target" in out
    assert "-1/" not in out


def test_known_target_without_expectations_warns_but_passes(isolated_files, capsys):
    _, expected_path, manifest_path = isolated_files
    _write(expected_path, "")
    _write(
        manifest_path,
        """
        targets:
          figure-x:
            class: A
            status: pending-calibration
        """,
    )

    assert verify.verify_targets(["figure-x"]) == 0
    assert "no expectations recorded yet" in capsys.readouterr().out


def test_missing_output_is_a_target_level_failure(isolated_files, capsys):
    _, expected_path, manifest_path = isolated_files
    _write(manifest_path, "targets: {}\n")
    _write(
        expected_path,
        """
        figure-x:
          source: outputs/missing.json
          checks:
            - path: value
              expected: 1.0
        """,
    )

    assert verify.verify_targets(["figure-x"]) == 1
    out = capsys.readouterr().out
    assert "missing output" in out
    assert "0/0 checks passed; 1 target-level failure(s)" in out


def test_checks_pass_and_fail_with_tolerances(isolated_files, capsys):
    tmp_path, expected_path, manifest_path = isolated_files
    _write(manifest_path, "targets: {}\n")
    payload = {"rows": [{"policy": "greedy", "cost": 2.0}], "top": 5.0}
    output_path = tmp_path / "summary.json"
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(
        expected_path,
        """
        figure-x:
          source: summary.json
          checks:
            - path: "rows.[policy=greedy].cost"
              expected: 2.0
              relative_tolerance: 1.0e-9
            - path: top
              expected: 5.4
              absolute_tolerance: 0.5
            - path: top
              expected: 6.0
              absolute_tolerance: 0.5
        """,
    )

    assert verify.verify_targets(["figure-x"]) == 1
    out = capsys.readouterr().out
    assert "2/3 checks passed" in out


def test_malformed_expected_raises(isolated_files):
    _, expected_path, _ = isolated_files
    _write(expected_path, "- just\n- a list\n")
    with pytest.raises(verify.VerificationError):
        verify.load_expected()


def test_extract_rejects_ambiguous_selector_and_non_numeric():
    payload = {"rows": [{"k": "a", "v": 1}, {"k": "a", "v": 2}], "name": "text"}
    with pytest.raises(verify.VerificationError, match="matched 2"):
        verify.extract(payload, "rows.[k=a].v", context="t")
    with pytest.raises(verify.VerificationError, match="not numeric"):
        verify.extract(payload, "name", context="t")
    assert verify.extract({"a": {"b": 3}}, "a.b", context="t") == 3.0
