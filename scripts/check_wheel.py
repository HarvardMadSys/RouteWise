#!/usr/bin/env python3
"""Fail when a RouteWise wheel leaks repository-only research code or data."""

from __future__ import annotations

import sys
import zipfile
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


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: check_wheel.py PATH_TO_WHEEL", file=sys.stderr)
        return 2
    wheel = Path(argv[1])
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())

    dist_info_prefixes = {name.split("/", 1)[0] + "/" for name in names if ".dist-info/" in name}
    dist_info = {
        name for name in names if any(name.startswith(prefix) for prefix in dist_info_prefixes)
    }
    unexpected = sorted(names - ALLOWED_LIBRARY_MEMBERS - dist_info)
    missing = sorted(ALLOWED_LIBRARY_MEMBERS - names)
    entry_points = sorted(name for name in names if name.endswith("entry_points.txt"))
    if len(dist_info_prefixes) != 1 or unexpected or missing or entry_points:
        if len(dist_info_prefixes) != 1:
            print("wheel must contain exactly one .dist-info directory", file=sys.stderr)
        if unexpected:
            print("unexpected wheel members:", *unexpected, sep="\n  ", file=sys.stderr)
        if missing:
            print("missing wheel members:", *missing, sep="\n  ", file=sys.stderr)
        if entry_points:
            print("unexpected console entry points:", *entry_points, sep="\n  ", file=sys.stderr)
        return 1
    print(f"wheel allowlist check passed: {wheel} ({len(names)} members)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
