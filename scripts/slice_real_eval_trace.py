"""Slice a composed BurstGPT/ShareGPT JSONL trace for real evaluation.

The input is expected to be the already-composed workload emitted by
``scripts/prepare_workload.py``: BurstGPT arrivals and token counts with
ShareGPT prompt text. This script only cuts a time window and optionally
rebases ``arrived_at`` to the window start.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR


def slice_trace(
    source: Path,
    output: Path,
    *,
    day: int,
    start_hour: int,
    hours: int,
    rebase: bool = True,
) -> dict[str, Any]:
    """Write ``source[day/start_hour, +hours)`` to ``output``.

    ``day`` is a zero-based day index. This matches the existing
    ``burstgpt_day2_h17_1h.jsonl`` artifact, whose first raw arrival falls in
    zero-based day 2, hour 17.
    """
    if day < 0:
        raise ValueError(f"day must be >= 0, got {day}")
    if not 0 <= start_hour < 24:
        raise ValueError(f"start_hour must be in [0, 23], got {start_hour}")
    if hours <= 0:
        raise ValueError(f"hours must be positive, got {hours}")

    window_start = day * SECONDS_PER_DAY + start_hour * SECONDS_PER_HOUR
    window_end = window_start + hours * SECONDS_PER_HOUR
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    first_arrived_at: float | None = None
    last_arrived_at: float | None = None
    per_hour = [0 for _ in range(hours)]

    with source.open() as src, output.open("w") as dst:
        for line_num, line in enumerate(src, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_num}: invalid JSON: {exc.msg}") from exc

            try:
                arrived_at = float(record["arrived_at"])
            except KeyError as exc:
                raise ValueError(f"{source}:{line_num}: missing arrived_at") from exc
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{source}:{line_num}: arrived_at must be numeric") from exc

            if arrived_at < window_start:
                continue
            if arrived_at >= window_end:
                break

            out_record = dict(record)
            out_arrived_at = arrived_at - window_start if rebase else arrived_at
            out_record["arrived_at"] = out_arrived_at
            dst.write(json.dumps(out_record, ensure_ascii=False) + "\n")

            count += 1
            first_arrived_at = out_arrived_at if first_arrived_at is None else first_arrived_at
            last_arrived_at = out_arrived_at
            hour_index = int((arrived_at - window_start) // SECONDS_PER_HOUR)
            if 0 <= hour_index < hours:
                per_hour[hour_index] += 1

    return {
        "source": str(source),
        "output": str(output),
        "day": day,
        "start_hour": start_hour,
        "hours": hours,
        "window_start_sec": window_start,
        "window_end_sec": window_end,
        "rebase": rebase,
        "total_requests": count,
        "first_arrived_at": first_arrived_at,
        "last_arrived_at": last_arrived_at,
        "requests_per_hour": per_hour,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Slice a composed BurstGPT/ShareGPT JSONL trace.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/burstgpt_30d.jsonl"),
        help="Composed BurstGPT/ShareGPT JSONL source.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/real_eval/burstgpt_day2_h17_8h.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--day",
        type=int,
        default=2,
        help="Zero-based day index. Default matches existing day2 real-eval slice.",
    )
    parser.add_argument(
        "--start-hour",
        type=int,
        default=17,
        help="Hour of day where the slice starts, in [0, 23].",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=8,
        help="Number of hours to include.",
    )
    parser.add_argument(
        "--no-rebase",
        action="store_true",
        help="Keep raw arrived_at timestamps instead of rebasing to window start.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    stats = slice_trace(
        args.source,
        args.output,
        day=args.day,
        start_hour=args.start_hour,
        hours=args.hours,
        rebase=not args.no_rebase,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
