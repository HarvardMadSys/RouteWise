"""Golden baseline regression test entrypoint."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
@pytest.mark.integration
def test_golden_baselines_match() -> None:
    """Re-capture all golden families and compare against stored baselines."""
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(repo_root / "tests" / "golden_capture.py"),
        "--mode",
        "compare",
    ]
    result = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "golden baseline comparison failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
