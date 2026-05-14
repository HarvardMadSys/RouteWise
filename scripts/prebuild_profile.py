#!/usr/bin/env python3
"""Build one shared real-eval initial latency profile."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

from experiments.real_evaluation.inventory import load_inventory
from experiments.real_evaluation.recorder import Recorder
from experiments.real_evaluation.runner import (
    DEFAULT_PROFILE_PROBE_PARALLELISM,
    DEFAULT_PROFILE_PROBE_SLEEP_SEC,
    DEFAULT_TIMEOUT_SEC,
    DEFAULT_WARMUP_PROBE_INTERVAL_SEC,
    DEFAULT_WARMUP_PROBES_PER_PROVIDER,
    RealExperimentRunner,
)
from experiments.real_evaluation.shared_profile import SharedProfileEventLog


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--probes-per-provider",
        type=int,
        default=DEFAULT_WARMUP_PROBES_PER_PROVIDER,
    )
    parser.add_argument(
        "--profile-probe-sleep-sec",
        type=float,
        default=DEFAULT_PROFILE_PROBE_SLEEP_SEC,
    )
    parser.add_argument(
        "--profile-probe-parallelism",
        type=int,
        default=DEFAULT_PROFILE_PROBE_PARALLELISM,
        help="Max concurrent provider probes per round. Use 0 for all providers.",
    )
    parser.add_argument(
        "--round-interval-sec",
        type=float,
        default=DEFAULT_WARMUP_PROBE_INTERVAL_SEC,
        help=(
            "Start-to-start cadence between probe rounds. With 24 rounds and "
            "5s cadence the warmup runs ≈2 minutes total."
        ),
    )
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--profile-window-sec", type=float, default=15 * 60.0)
    parser.add_argument("--max-cost-usd", type=float, default=5.0)
    parser.add_argument(
        "--shared-profile-events",
        type=Path,
        default=None,
        help=(
            "Also seed the shared profile JSONL event log with the warmup "
            "samples exported to --output."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    inventory = load_inventory(args.inventory)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="routewise_profile_probe_") as tmp_dir:
        recorder = Recorder(tmp_dir)
        try:
            runner = RealExperimentRunner(
                inventory=inventory,
                policy_names=["greedy_latency"],
                recorder=recorder,
                max_cost_usd=args.max_cost_usd,
                timeout_sec=args.timeout_sec,
                profile_window_sec=args.profile_window_sec,
            )
            runner.probe_profiles(
                probes_per_provider=args.probes_per_provider,
                sleep_sec=args.profile_probe_sleep_sec,
                round_interval_sec=args.round_interval_sec,
                phase="warmup",
                parallelism=args.profile_probe_parallelism,
            )
            profile = runner.export_initial_profile()
            shared_seed_count = 0
            shared_seed_offset = None
            if args.shared_profile_events is not None:
                event_log = SharedProfileEventLog(args.shared_profile_events)
                for provider, entry in profile.get("providers", {}).items():
                    if not isinstance(entry, dict):
                        continue
                    for sample in entry.get("samples", []):
                        if not isinstance(sample, dict):
                            continue
                        event_log.append(
                            provider=str(provider),
                            ts=float(sample["ts"]),
                            ttft_ms=float(sample["ttft_ms"]),
                            error_type=None,
                            source="warmup",
                        )
                        shared_seed_count += 1
                    for error in entry.get("errors", []):
                        if not isinstance(error, dict):
                            continue
                        event_log.append(
                            provider=str(provider),
                            ts=float(error["ts"]),
                            ttft_ms=-1.0,
                            error_type=str(error.get("error_type") or "error"),
                            source="warmup",
                        )
                        shared_seed_count += 1
                shared_seed_offset = event_log.end_offset()
            profile["_meta"] = {
                "inventory": str(args.inventory),
                "probes_per_provider": args.probes_per_provider,
                "profile_probe_parallelism": args.profile_probe_parallelism,
                "profile_probe_cost_usd": round(runner._profile_probe_cost_usd, 8),
                "profile_probe_counts": dict(runner._profile_probe_counts),
            }
            if args.shared_profile_events is not None:
                profile["_meta"]["shared_profile_events"] = str(args.shared_profile_events)
                profile["_meta"]["shared_profile_seed_count"] = shared_seed_count
                profile["_meta"]["shared_profile_seed_offset"] = shared_seed_offset
        finally:
            recorder.close()

    args.output.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    logging.info("wrote initial profile: %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
