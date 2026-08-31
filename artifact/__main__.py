"""Reviewer-facing artifact commands: smoke, list, reproduce, verify.

    uv run python -m artifact smoke
    uv run python -m artifact list
    uv run python -m artifact reproduce <target> [<target> ...] | --group A
    uv run python -m artifact verify [<target> ...]

`smoke` is the kick-the-tires path: import checks, one simulator target on the
committed fixture, verification of its summary numbers, and the fast unit
tests. It needs no network access and no API keys.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import time

from artifact import runner, verify

SMOKE_TARGET = "smoke-cost-layer"
SMOKE_IMPORT_MODULES = (
    "llm_routewise",
    "llm_routewise.sim.engine.simulator",
    "experiments.simulation.cost_layer",
    "experiments.simulation.latency_layer",
    "experiments.simulation.hedging",
    "experiments.simulation.end_to_end",
)
SMOKE_TEST_ARGS = (
    "-m",
    "pytest",
    "-q",
    "-m",
    "not slow",
    "tests/test_architecture_scaffold.py",
    "tests/unit/simulation",
)


def _print_target_rows() -> None:
    rows = runner.list_targets()
    width = max(len(row["name"]) for row in rows)
    for row in rows:
        est = f" ~{row['est_minutes']}min" if row.get("est_minutes") else ""
        print(f"{row['name']:<{width}}  [{row['class']:>5}] {row['status']}{est}")
        if row["claim"]:
            print(f"{'':<{width}}  {row['claim']}")


def _run_smoke() -> int:
    started = time.monotonic()
    print("smoke: import checks")
    for module_name in SMOKE_IMPORT_MODULES:
        importlib.import_module(module_name)
        print(f"  imported {module_name}")

    print(f"smoke: running target {SMOKE_TARGET} on the committed fixture")
    exit_code = runner.run_target(SMOKE_TARGET)
    if exit_code != 0:
        print(f"smoke: target failed with exit code {exit_code}")
        return exit_code

    print("smoke: verifying summary numbers")
    exit_code = verify.verify_targets([SMOKE_TARGET])
    if exit_code != 0:
        return exit_code

    print("smoke: fast unit tests")
    completed = subprocess.run([sys.executable, *SMOKE_TEST_ARGS], cwd=runner.ROOT_DIR)
    if completed.returncode != 0:
        return completed.returncode

    elapsed = time.monotonic() - started
    print(f"smoke: PASS in {elapsed:.0f}s")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m artifact", description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("smoke", help="Kick-the-tires pipeline check (no keys, no network).")
    subparsers.add_parser("list", help="List claim/figure targets from the manifest.")
    reproduce_parser = subparsers.add_parser("reproduce", help="Run manifest targets.")
    reproduce_parser.add_argument("targets", nargs="*", help="Target names from `list`.")
    reproduce_parser.add_argument("--group", help="Run every runnable target in a class (e.g. A).")
    verify_parser = subparsers.add_parser("verify", help="Check outputs against expected.yaml.")
    verify_parser.add_argument("targets", nargs="*", help="Targets to verify (default: all).")

    args = parser.parse_args(argv)
    if args.command == "smoke":
        return _run_smoke()
    if args.command == "list":
        _print_target_rows()
        return 0
    if args.command == "reproduce":
        names = list(args.targets)
        if args.group:
            names.extend(runner.targets_in_group(args.group))
        if not names:
            print("nothing to run: pass target names or --group; see `python -m artifact list`")
            return 2
        for name in names:
            print(f"reproduce: {name}")
            try:
                exit_code = runner.run_target(name)
            except runner.TargetError as exc:
                print(f"error: {exc}")
                return 2
            if exit_code != 0:
                return exit_code
        return 0
    if args.command == "verify":
        try:
            return verify.verify_targets(list(args.targets) or None)
        except verify.VerificationError as exc:
            print(f"error: {exc}")
            return 2
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
