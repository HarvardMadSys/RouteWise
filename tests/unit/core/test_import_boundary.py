"""Import-boundary tests for the public lightweight RouteWise core."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HEAVY_MODULES = ("numpy", "scipy", "pandas", "matplotlib")


def test_routewise_core_import_does_not_load_heavy_runtime_dependencies() -> None:
    """Lock the public core import chain to stdlib-only modules.

    This catches accidental eager imports through ``routewise``,
    ``routewise.const``, or ``routewise.core``.
    """
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(ROOT)
        if not existing_pythonpath
        else f"{ROOT}{os.pathsep}{existing_pythonpath}"
    )
    code = """
import json
import sys

import routewise.core as core

payload = {
    "heavy": {name: name in sys.modules for name in ("numpy", "scipy", "pandas", "matplotlib")},
    "solver_module": core.solve_budget_lp.__module__,
}
print(json.dumps(payload, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["solver_module"] == "routewise.core.lp"
    assert payload["heavy"] == dict.fromkeys(HEAVY_MODULES, False)
