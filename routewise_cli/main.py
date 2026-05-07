"""RouteWise command-line interface.

This package is intentionally outside :mod:`rwsim`: the CLI is an application
layer that may import both `experiments` and `rwsim`, while the simulator core
must not depend on experiment recipes.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Any

from experiments import available_experiments, get_experiment

SIMULATOR_SECTIONS = {
    "cost-layer": "experiments.simulation.cost_layer",
}

SIMULATOR_SECTION_DESCRIPTIONS = {
    "cost-layer": "paper §3.2 — same latency / different cost",
}

ABLATION_COMMANDS = {
    "effective-cost": "experiments.ablations.effective_cost.harness",
}

ABLATION_COMMAND_DESCRIPTIONS = {
    "effective-cost": "effective-cost formula ablation harness",
}


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="routewise", description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list", help="List config-driven experiments.")
    list_parser.add_argument("--experiment", choices=available_experiments())

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

    simulator_parser = subparsers.add_parser(
        "simulator", help="Run section-based simulator experiments."
    )
    simulator_subparsers = simulator_parser.add_subparsers(dest="simulator_command")
    simulator_subparsers.add_parser("list", help="List registered simulator sections.")
    for section_name in SIMULATOR_SECTIONS:
        section_parser = simulator_subparsers.add_parser(section_name, help=f"Run {section_name}.")
        section_parser.add_argument(
            "section_args",
            nargs=argparse.REMAINDER,
            help="Arguments passed through to the section after an optional -- separator.",
        )

    ablation_parser = subparsers.add_parser("ablation", help="Run ablation experiment harnesses.")
    ablation_subparsers = ablation_parser.add_subparsers(dest="ablation_command")
    ablation_subparsers.add_parser("list", help="List registered ablation harnesses.")
    for command_name in ABLATION_COMMANDS:
        command_parser = ablation_subparsers.add_parser(
            command_name,
            help=ABLATION_COMMAND_DESCRIPTIONS.get(command_name, f"Run {command_name}."),
        )
        command_parser.add_argument(
            "command_args",
            nargs=argparse.REMAINDER,
            help="Arguments passed through to the ablation harness after an optional -- separator.",
        )

    return parser


def _list_payload(experiment_name: str | None) -> dict[str, Any]:
    if experiment_name is not None:
        experiment = get_experiment(experiment_name)
        return {"experiment": experiment_name, "scenarios": experiment.list_scenarios()}
    return {"experiments": available_experiments()}


def _simulator_list_payload() -> dict[str, Any]:
    sections: list[dict[str, Any]] = []
    for name, module_name in SIMULATOR_SECTIONS.items():
        module = importlib.import_module(module_name)
        list_scenarios = getattr(module, "list_scenarios", None)
        policies_for_section = getattr(module, "policies_for_section", None)
        sections.append(
            {
                "name": name,
                "module": module_name,
                "description": SIMULATOR_SECTION_DESCRIPTIONS.get(name, ""),
                "scenarios": list(list_scenarios() if callable(list_scenarios) else ()),
                "policies": list(policies_for_section() if callable(policies_for_section) else ()),
            }
        )
    return {"sections": sections}


def _ablation_list_payload() -> dict[str, Any]:
    return {
        "ablations": [
            {
                "name": name,
                "module": module_name,
                "description": ABLATION_COMMAND_DESCRIPTIONS.get(name, ""),
            }
            for name, module_name in ABLATION_COMMANDS.items()
        ]
    }


def _dispatch_simulator(raw_args: list[str]) -> int | None:
    if not raw_args or raw_args[0] != "simulator" or len(raw_args) < 2:
        return None
    command = raw_args[1]
    if command == "list":
        print(_json_dump(_simulator_list_payload()))
        return 0
    if command in SIMULATOR_SECTIONS:
        module = importlib.import_module(SIMULATOR_SECTIONS[command])
        return int(module.main(raw_args[2:]))
    return None


def _dispatch_ablation(raw_args: list[str]) -> int | None:
    if not raw_args or raw_args[0] != "ablation" or len(raw_args) < 2:
        return None
    command = raw_args[1]
    if command == "list":
        print(_json_dump(_ablation_list_payload()))
        return 0
    if command in ABLATION_COMMANDS:
        module = importlib.import_module(ABLATION_COMMANDS[command])
        return int(module.main(raw_args[2:]))
    return None


def main(argv: list[str] | None = None) -> int:
    """Run the RouteWise CLI."""
    raw_args = list(sys.argv[1:] if argv is None else argv)
    simulator_result = _dispatch_simulator(raw_args)
    if simulator_result is not None:
        return simulator_result
    ablation_result = _dispatch_ablation(raw_args)
    if ablation_result is not None:
        return ablation_result

    parser = _build_parser()
    args = parser.parse_args(raw_args)

    if args.command == "list":
        print(_json_dump(_list_payload(args.experiment)))
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
                payload = {
                    "experiment": args.experiment,
                    "validated": [item.name for item in scenarios],
                }
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

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
