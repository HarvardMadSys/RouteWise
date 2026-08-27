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
    "llm_routewise/__init__.py",
    "llm_routewise/_capacity_controller.py",
    "llm_routewise/_output_length.py",
    "llm_routewise/const.py",
    "llm_routewise/core/__init__.py",
    "llm_routewise/core/beliefs.py",
    "llm_routewise/core/cost.py",
    "llm_routewise/core/hedging.py",
    "llm_routewise/core/latency_profile.py",
    "llm_routewise/core/lp.py",
    "llm_routewise/core/pricing.py",
    "llm_routewise/core/provider_view.py",
    "llm_routewise/core/router.py",
    "llm_routewise/core/types.py",
    "llm_routewise/errors.py",
    "llm_routewise/facade.py",
    "llm_routewise/py.typed",
    "llm_routewise/stateless.py",
}


def _expected_project_metadata() -> tuple[str, str, str]:
    init_path = Path(__file__).resolve().parents[1] / "llm_routewise" / "__init__.py"
    tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "__version__" for target in statement.targets):
            version = ast.literal_eval(statement.value)
            if isinstance(version, str):
                return "llm-routewise", version, ">=3.10"
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
    expected_name, expected_version, _ = _expected_project_metadata()
    expected_dist_info_prefix = (
        f"{expected_name.replace('-', '_')}-{expected_version}.dist-info/"
    )
    invalid_dist_info_prefix = dist_info_prefixes != {expected_dist_info_prefix}
    dist_info = {
        name for name in names if any(name.startswith(prefix) for prefix in dist_info_prefixes)
    }
    unexpected = sorted(names - ALLOWED_LIBRARY_MEMBERS - dist_info)
    missing = sorted(ALLOWED_LIBRARY_MEMBERS - names)
    entry_points = sorted(name for name in names if name.endswith("entry_points.txt"))
    if invalid_dist_info_prefix or unexpected or missing or entry_points or metadata_errors:
        if len(dist_info_prefixes) != 1:
            print("wheel must contain exactly one .dist-info directory", file=sys.stderr)
        elif invalid_dist_info_prefix:
            print(
                f"wheel .dist-info directory must be {expected_dist_info_prefix!r}",
                file=sys.stderr,
            )
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
