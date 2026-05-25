#!/usr/bin/env python3
"""Maintain a shared real-eval latency profile with idle-only probes."""

from __future__ import annotations

import argparse
import logging
import signal
import tempfile
import threading
import time
from pathlib import Path

from experiments.real_evaluation.inventory import load_inventory
from experiments.real_evaluation.recorder import Recorder
from experiments.real_evaluation.runner import (
    DEFAULT_PROFILE_PROBE_SLEEP_SEC,
    DEFAULT_TIMEOUT_SEC,
    RealExperimentRunner,
)
from experiments.real_evaluation.shared_profile import SharedProfileEventLog


def _filter_probe_providers(providers: list[object], excluded: set[str]) -> list[object]:
    return [spec for spec in providers if getattr(spec, "name") not in excluded]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--shared-profile-events", type=Path, required=True)
    parser.add_argument("--interval-sec", type=float, default=5.0)
    parser.add_argument("--idle-threshold-sec", type=float, default=5.0)
    parser.add_argument("--profile-window-sec", type=float, default=15 * 60.0)
    parser.add_argument(
        "--max-probes-per-tick",
        type=int,
        default=0,
        help="Max idle providers to probe per tick. Use 0 for all idle providers.",
    )
    parser.add_argument(
        "--exclude-provider",
        action="append",
        default=[],
        help="Provider name to exclude from active shared probes. May be repeated.",
    )
    parser.add_argument(
        "--profile-probe-sleep-sec",
        type=float,
        default=DEFAULT_PROFILE_PROBE_SLEEP_SEC,
    )
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--max-cost-usd", type=float, default=5.0)
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser


def _install_signal_handlers(stop_event: threading.Event) -> None:
    def _handle(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.interval_sec <= 0:
        raise ValueError("--interval-sec must be positive")
    if args.idle_threshold_sec < 0:
        raise ValueError("--idle-threshold-sec must be non-negative")

    inventory = load_inventory(args.inventory)
    event_log = SharedProfileEventLog(args.shared_profile_events)
    stop_event = threading.Event()
    _install_signal_handlers(stop_event)
    cursor = 0

    with tempfile.TemporaryDirectory(prefix="routewise_shared_profile_probe_") as tmp_dir:
        recorder = Recorder(tmp_dir)
        try:
            runner = RealExperimentRunner(
                inventory=inventory,
                policy_names=[],
                recorder=recorder,
                max_cost_usd=args.max_cost_usd,
                timeout_sec=args.timeout_sec,
                profile_window_sec=args.profile_window_sec,
                shared_profile_events_path=args.shared_profile_events,
            )
            excluded_providers = set(args.exclude_provider)
            providers = _filter_probe_providers(
                list(inventory.providers),
                excluded_providers,
            )
            logging.info(
                "shared profile prober started: providers=%d excluded=%s path=%s interval=%.1fs idle=%.1fs max_per_tick=%d",
                len(providers),
                ",".join(sorted(excluded_providers)) or "-",
                args.shared_profile_events,
                args.interval_sec,
                args.idle_threshold_sec,
                args.max_probes_per_tick,
            )
            while not stop_event.is_set():
                now = time.time()
                latest = event_log.recent_last_event_by_provider(
                    now=now,
                    window_sec=args.profile_window_sec,
                )
                idle_indices: list[int] = []
                for offset in range(len(providers)):
                    idx = (cursor + offset) % len(providers)
                    spec = providers[idx]
                    if spec.name in runner._providers_missing_api_key:
                        continue
                    event = latest.get(spec.name)
                    if event is None or now - event.ts >= args.idle_threshold_sec:
                        idle_indices.append(idx)
                if idle_indices:
                    max_probes = (
                        len(idle_indices)
                        if args.max_probes_per_tick <= 0
                        else min(args.max_probes_per_tick, len(idle_indices))
                    )
                    selected_indices = idle_indices[:max_probes]
                    selected = [providers[idx] for idx in selected_indices]
                    logging.info(
                        "shared profile idle probe: providers=%s parallelism=%d",
                        ",".join(spec.name for spec in selected),
                        len(selected),
                    )
                    runner._probe_profile_round_parallel(
                        selected,
                        round_idx=0,
                        probes_per_provider=1,
                        retry_sleep_sec=args.profile_probe_sleep_sec,
                        phase="shared",
                        stop_event=stop_event,
                        parallelism=0,
                    )
                    cursor = (selected_indices[-1] + 1) % len(providers)
                else:
                    logging.debug("shared profile prober: all providers active in idle window")
                stop_event.wait(args.interval_sec)
        finally:
            recorder.close()
    logging.info("shared profile prober stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
