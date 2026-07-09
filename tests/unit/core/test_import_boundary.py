"""Import-boundary tests for the public lightweight RouteWise core."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HEAVY_MODULES = ("numpy", "scipy", "pandas", "matplotlib")


def test_lightweight_routewise_imports_do_not_load_heavy_runtime_dependencies() -> None:
    """Lock public shared import chains to stdlib-only modules.

    This catches accidental eager imports through ``routewise``,
    ``routewise.const``, ``routewise.capacity``, ``routewise.schemas``, or
    ``routewise.core``.
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

import routewise
import routewise.capacity as capacity
import routewise.const as const
import routewise.core as core
import routewise.schemas as schemas

payload = {
    "heavy": {name: name in sys.modules for name in ("numpy", "scipy", "pandas", "matplotlib")},
    "capacity_tier_module": capacity.ProviderTier.__module__,
    "const_module": const.DEFAULT_PRIMARY_SLO_MS.__class__.__module__,
    "request_module": schemas.Request.__module__,
    "routewise_all": routewise.__all__,
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

    assert payload["capacity_tier_module"] == "routewise.capacity"
    assert payload["request_module"] == "routewise.schemas"
    assert payload["solver_module"] == "routewise.core.lp"
    assert payload["heavy"] == dict.fromkeys(HEAVY_MODULES, False)
