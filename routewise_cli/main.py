"""RouteWise command-line interface.

This package is intentionally outside :mod:`rwsim`: the CLI is an application
layer that may import both `experiments` and `rwsim`, while the simulator core
must not depend on experiment recipes.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from experiments import available_experiments, get_experiment
from experiments.suites import available_suites, get_suite, run_suite


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="routewise", description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list", help="List experiments, scenarios, and suites.")
    list_parser.add_argument("--experiment", choices=available_experiments())
    list_parser.add_argument(
        "--suites",
        action="store_true",
        help="List full-sweep suites instead of config-driven scenarios.",
    )

    describe_parser = subparsers.add_parser("describe", help="Describe one scenario config.")
    describe_parser.add_argument("experiment", choices=available_experiments())
    describe_parser.add_argument("--scenario", required=True)

    validate_parser = subparsers.add_parser("validate", help="Load scenario configs.")
    validate_parser.add_argument("experiment", choices=available_experiments())
    validate_parser.add_argument("--scenario")

    run_parser = subparsers.add_parser("run", help="Run one config-driven scenario/policy.")
    run_parser.add_argument("experiment", choices=available_experiments())
    run_parser.add_argument("--scenario", required=True)
    run_parser.add_argument("--policy", required=True)
    run_parser.add_argument("--seed", type=int, default=42)

    suite_parser = subparsers.add_parser("suite", help="Run a registered full-sweep suite.")
    suite_parser.add_argument("suite", choices=available_suites())
    suite_parser.add_argument(
        "suite_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the suite after an optional -- separator.",
    )

    return parser


def _list_payload(experiment_name: str | None, *, suites: bool) -> dict[str, Any]:
    if suites:
        return {
            "suites": [
                {
                    "name": name,
                    "module": get_suite(name).module,
                    "description": get_suite(name).description,
                }
                for name in available_suites()
            ]
        }
    if experiment_name is not None:
        experiment = get_experiment(experiment_name)
        return {"experiment": experiment_name, "scenarios": experiment.list_scenarios()}
    return {"experiments": available_experiments(), "suites": available_suites()}


def main(argv: list[str] | None = None) -> int:
    """Run the RouteWise CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        print(_json_dump(_list_payload(args.experiment, suites=args.suites)))
        return 0

    if args.command == "describe":
        experiment = get_experiment(args.experiment)
        try:
            print(_json_dump(experiment.summarize(args.scenario)))
        except RuntimeError as exc:
            raise SystemExit(f"error: {exc}") from exc
        return 0

    if args.command == "validate":
        experiment = get_experiment(args.experiment)
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

    if args.command == "run":
        experiment = get_experiment(args.experiment)
        if not hasattr(experiment, "run_policy"):
            raise SystemExit(f"experiment {args.experiment!r} does not support run yet")
        try:
            payload = experiment.run_policy(args.scenario, args.policy, seed=args.seed)
        except RuntimeError as exc:
            raise SystemExit(f"error: {exc}") from exc
        print(_json_dump(payload))
        return 0

    if args.command == "suite":
        suite_args = args.suite_args
        if suite_args and suite_args[0] == "--":
            suite_args = suite_args[1:]
        return run_suite(args.suite, suite_args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
