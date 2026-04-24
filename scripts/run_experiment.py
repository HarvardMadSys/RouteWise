#!/usr/bin/env python3
"""Inspect config-driven RouteWise experiments.

This is intentionally a thin CLI. It validates and describes external configs;
simulation execution still belongs in migrated `rwsim` runners.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments import available_experiments, get_experiment


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=available_experiments(),
        default="tiered_capacity",
        help="Experiment package to inspect.",
    )
    parser.add_argument("--list", action="store_true", help="List scenario configs.")
    parser.add_argument("--scenario", help="Scenario name to describe or validate.")
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Load a scenario config and print a JSON summary.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Load scenario configs without running simulation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the experiment inspection CLI."""
    args = _build_parser().parse_args(argv)
    experiment = get_experiment(args.experiment)

    if args.list:
        print(_json_dump({"experiment": args.experiment, "scenarios": experiment.list_scenarios()}))
        return 0

    if args.describe:
        if not args.scenario:
            raise SystemExit("--describe requires --scenario")
        try:
            print(_json_dump(experiment.summarize(args.scenario)))
        except RuntimeError as exc:
            raise SystemExit(f"error: {exc}") from exc
        return 0

    if args.validate:
        try:
            if args.scenario:
                scenario = experiment.load_scenario(args.scenario)
                payload = {"experiment": args.experiment, "validated": [scenario.name]}
            else:
                scenarios = experiment.load_all_scenarios()
                payload = {"experiment": args.experiment, "validated": [item.name for item in scenarios]}
        except RuntimeError as exc:
            raise SystemExit(f"error: {exc}") from exc
        print(_json_dump(payload))
        return 0

    parser = _build_parser()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
