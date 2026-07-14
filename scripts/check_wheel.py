#!/usr/bin/env python3
"""Fail when a RouteWise wheel leaks repository-only research code or data."""

from __future__ import annotations

import ast
import sys
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path

ALLOWED_LIBRARY_MEMBERS = {
    "routewise/__init__.py",
    "routewise/_capacity_controller.py",
    "routewise/_output_length.py",
    # Kept provisionally for source-checkout compatibility while OQ10 is open.
    "routewise/capacity.py",
    "routewise/const.py",
    "routewise/core/__init__.py",
    "routewise/core/beliefs.py",
    "routewise/core/cost.py",
    "routewise/core/hedging.py",
    "routewise/core/latency_profile.py",
    "routewise/core/lp.py",
    "routewise/core/pricing.py",
    "routewise/core/provider_view.py",
    "routewise/core/router.py",
    "routewise/core/types.py",
    "routewise/errors.py",
    "routewise/facade.py",
    "routewise/py.typed",
    "routewise/schemas.py",
    "routewise/stateless.py",
}


def _expected_project_metadata() -> tuple[str, str, str]:
    init_path = Path(__file__).resolve().parents[1] / "routewise" / "__init__.py"
    tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "__version__" for target in statement.targets):
            version = ast.literal_eval(statement.value)
            if isinstance(version, str):
                return "routewise", version, ">=3.10"
    raise RuntimeError(f"could not read __version__ from {init_path}")


def _metadata_errors(archive: zipfile.ZipFile, names: set[str]) -> list[str]:
    metadata_members = sorted(name for name in names if name.endswith(".dist-info/METADATA"))
    if len(metadata_members) != 1:
        return ["wheel must contain exactly one .dist-info/METADATA file"]

    metadata = BytesParser(policy=default).parsebytes(archive.read(metadata_members[0]))
    expected_name, expected_version, expected_python = _expected_project_metadata()
    errors: list[str] = []
    expected_fields = {
        "Name": expected_name,
        "Version": expected_version,
        "Requires-Python": expected_python,
    }
    for field, expected in expected_fields.items():
        actual = metadata.get(field)
        if actual != expected:
            errors.append(f"METADATA {field} must be {expected!r}, got {actual!r}")
    requirements = metadata.get_all("Requires-Dist", [])
    if requirements:
        errors.append(f"wheel must have no runtime dependencies, got {requirements!r}")
    return errors


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: check_wheel.py PATH_TO_WHEEL", file=sys.stderr)
        return 2
    wheel = Path(argv[1])
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_errors = _metadata_errors(archive, names)

    dist_info_prefixes = {name.split("/", 1)[0] + "/" for name in names if ".dist-info/" in name}
    dist_info = {
        name for name in names if any(name.startswith(prefix) for prefix in dist_info_prefixes)
    }
    unexpected = sorted(names - ALLOWED_LIBRARY_MEMBERS - dist_info)
    missing = sorted(ALLOWED_LIBRARY_MEMBERS - names)
    entry_points = sorted(name for name in names if name.endswith("entry_points.txt"))
    if len(dist_info_prefixes) != 1 or unexpected or missing or entry_points or metadata_errors:
        if len(dist_info_prefixes) != 1:
            print("wheel must contain exactly one .dist-info directory", file=sys.stderr)
        if unexpected:
            print("unexpected wheel members:", *unexpected, sep="\n  ", file=sys.stderr)
        if missing:
            print("missing wheel members:", *missing, sep="\n  ", file=sys.stderr)
        if entry_points:
            print("unexpected console entry points:", *entry_points, sep="\n  ", file=sys.stderr)
        if metadata_errors:
            print("invalid wheel metadata:", *metadata_errors, sep="\n  ", file=sys.stderr)
        return 1
    print(f"wheel allowlist check passed: {wheel} ({len(names)} members)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
