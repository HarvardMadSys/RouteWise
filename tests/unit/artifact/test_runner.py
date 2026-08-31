"""Unit tests for the artifact runner dispatch."""

from __future__ import annotations

import os
import sys
import textwrap
import types

import pytest

from artifact import runner


@pytest.fixture
def isolated_manifest(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.yaml"
    monkeypatch.setattr(runner, "MANIFEST_PATH", manifest_path)

    def write(content: str) -> None:
        manifest_path.write_text(textwrap.dedent(content), encoding="utf-8")

    return write


def test_unknown_target_raises(isolated_manifest):
    isolated_manifest("targets: {}\n")
    with pytest.raises(runner.TargetError, match="unknown target"):
        runner.run_target("nope")


def test_pending_target_refuses_with_reason(isolated_manifest):
    isolated_manifest(
        """
        targets:
          figure-x:
            class: A
            status: pending-calibration
            status_reason: being pinned
        """
    )
    with pytest.raises(runner.TargetError, match="being pinned"):
        runner.run_target("figure-x")


def test_missing_required_input_names_the_hint(isolated_manifest):
    isolated_manifest(
        """
        targets:
          figure-x:
            entrypoint: artifact_test_stub
            requires:
              data/nope.jsonl: run scripts/prepare_workload.py
        """
    )
    with pytest.raises(runner.TargetError, match="prepare_workload"):
        runner.run_target("figure-x")


def test_run_target_executes_from_repo_root_and_restores_cwd(
    isolated_manifest, tmp_path, monkeypatch
):
    isolated_manifest(
        """
        targets:
          figure-x:
            entrypoint: artifact_test_stub
            args: [--flag]
        """
    )
    seen: dict[str, object] = {}
    stub = types.ModuleType("artifact_test_stub")

    def stub_main(argv):
        seen["cwd"] = os.getcwd()
        seen["argv"] = list(argv)
        return 0

    stub.main = stub_main
    monkeypatch.setitem(sys.modules, "artifact_test_stub", stub)

    monkeypatch.chdir(tmp_path)
    assert runner.run_target("figure-x") == 0
    assert seen["cwd"] == str(runner.ROOT_DIR)
    assert seen["argv"] == ["--flag"]
    assert os.getcwd() == str(tmp_path)


def test_targets_in_group_skips_non_runnable(isolated_manifest):
    isolated_manifest(
        """
        targets:
          a-ready:
            class: A
          a-pending:
            class: A
            status: pending-records
          b-ready:
            class: B1
        """
    )
    assert runner.targets_in_group("A") == ["a-ready"]
