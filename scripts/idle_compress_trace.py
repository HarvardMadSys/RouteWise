"""Compress long idle gaps in a real-eval JSONL trace.

Bursts in workloads like BurstGPT live inside short sub-second arrival
gaps; the wall-clock time between bursts is mostly idle. Real-eval at
``speedup=1.0`` honors every gap, so a 24h trace takes 24h to replay
even when 96% of those hours have no activity.

This script writes a new JSONL where every gap longer than
``--cap-gap-sec`` is capped to that value. By default, gaps ``<= cap``
(i.e. all intra-burst arrivals) are preserved exactly, so the workload's
within-burst structure — peak QPS, hot-second density, request order —
is unchanged. When ``--min-gap-sec`` is set, the compressed schedule is
post-processed to enforce a lower bound between adjacent requests while
letting later idle gaps absorb that shift.

The runner can replay the output at ``--speedup=1.0`` and the new
quota-clock sanity check (``check_quota_clock_alignment``) still
passes, because the trace's ``arrived_at`` field already encodes the
compressed wall-clock.

Caveats:

* Inventory quota windows are NOT scaled. If the inventory configures
  Chutes as ``5000 / 24h`` and you compress 24h → 8h, the policy can
  still spend up to 5000 inside the 8h replay because the quota window
  (24h) is longer than the replay (8h). That matches the real Chutes
  account: the wall-clock 24h subscription window does not reset
  during the 8h run. The paper claim "given 5000/day quota and this
  24h workload" is preserved.

* The compressed trace is no longer a real time-series — interpreting
  the rewritten ``arrived_at`` as "when this request actually happened"
  is wrong. It is a replay schedule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def compress_idle_gaps(
    source: Path,
    output: Path,
    *,
    cap_gap_sec: float,
    min_gap_sec: float = 0.0,
) -> dict[str, Any]:
    """Rewrite ``source`` to ``output`` capping every inter-arrival gap.

    Returns summary stats including raw span, compressed span, and the
    number of gaps that were left untouched.
    """
    if cap_gap_sec <= 0:
        raise ValueError(f"cap_gap_sec must be positive, got {cap_gap_sec}")
    if min_gap_sec < 0:
        raise ValueError(f"min_gap_sec must be non-negative, got {min_gap_sec}")

    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[float, dict[str, Any]]] = []
    with source.open() as src:
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
            rows.append((arrived_at, record))

    rows.sort(key=lambda r: r[0])

    raw_first = rows[0][0] if rows else 0.0
    raw_last = rows[-1][0] if rows else 0.0
    raw_span = raw_last - raw_first

    gaps_untouched = 0
    gaps_capped = 0
    sum_savings = 0.0

    new_arrived_at: list[float] = []
    if rows:
        new_arrived_at.append(0.0)
        for prev, current in zip(rows, rows[1:], strict=False):
            gap = current[0] - prev[0]
            if gap <= cap_gap_sec + 1e-12:
                effective_gap = gap
                gaps_untouched += 1
            else:
                effective_gap = cap_gap_sec
                gaps_capped += 1
                sum_savings += gap - cap_gap_sec
            new_arrived_at.append(new_arrived_at[-1] + effective_gap)

    min_gap_shifts = 0
    max_min_gap_shift_sec = 0.0
    final_min_gap_shift_sec = 0.0
    if min_gap_sec > 0.0 and new_arrived_at:
        adjusted: list[float] = []
        previous = 0.0
        for i, base_ts in enumerate(new_arrived_at):
            if i == 0:
                new_ts = 0.0
            else:
                new_ts = max(base_ts, previous + min_gap_sec)
            shift = max(0.0, new_ts - base_ts)
            if shift > 0.0:
                min_gap_shifts += 1
            max_min_gap_shift_sec = max(max_min_gap_shift_sec, shift)
            final_min_gap_shift_sec = shift
            adjusted.append(new_ts)
            previous = new_ts
        new_arrived_at = adjusted

    with output.open("w") as dst:
        for (_, record), new_ts in zip(rows, new_arrived_at, strict=True):
            out_record = dict(record)
            out_record["arrived_at"] = new_ts
            dst.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    compressed_span = new_arrived_at[-1] - new_arrived_at[0] if new_arrived_at else 0.0

    return {
        "mode": "cap_idle_gaps",
        "source": str(source),
        "output": str(output),
        "cap_gap_sec": float(cap_gap_sec),
        "min_gap_sec": float(min_gap_sec),
        "total_requests": len(rows),
        "raw_first_arrived_at": raw_first,
        "raw_last_arrived_at": raw_last,
        "raw_span_sec": raw_span,
        "raw_span_hours": raw_span / 3600.0 if raw_span else 0.0,
        "compressed_span_sec": compressed_span,
        "compressed_span_hours": compressed_span / 3600.0 if compressed_span else 0.0,
        "gaps_untouched": gaps_untouched,
        "gaps_capped": gaps_capped,
        "fraction_gaps_untouched": (
            gaps_untouched / max(1, gaps_untouched + gaps_capped)
        ),
        "wall_clock_savings_sec": sum_savings,
        "wall_clock_savings_hours": sum_savings / 3600.0,
        "min_gap_shifts": min_gap_shifts,
        "max_min_gap_shift_sec": max_min_gap_shift_sec,
        "final_min_gap_shift_sec": final_min_gap_shift_sec,
    }


def smooth_to_target_span(
    source: Path,
    output: Path,
    *,
    target_span_hours: float,
    min_gap_sec: float,
) -> dict[str, Any]:
    """Rewrite arrivals onto a fixed span while enforcing a minimum gap.

    This keeps request order and the coarse shape of the original 24h trace,
    but avoids replaying large same-timestamp bursts as an instantaneous spike.
    Each original arrival is first linearly scaled into the target span, then
    shifted forward just enough to satisfy ``min_gap_sec`` from the previous
    request. Later idle time can absorb that shift when the scaled trace has
    enough slack.
    """
    if target_span_hours <= 0:
        raise ValueError(
            f"target_span_hours must be positive, got {target_span_hours}"
        )
    if min_gap_sec < 0:
        raise ValueError(f"min_gap_sec must be non-negative, got {min_gap_sec}")

    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[float, dict[str, Any]]] = []
    with source.open() as src:
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
            rows.append((arrived_at, record))

    rows.sort(key=lambda r: r[0])

    raw_first = rows[0][0] if rows else 0.0
    raw_last = rows[-1][0] if rows else 0.0
    raw_span = raw_last - raw_first
    target_span_sec = target_span_hours * 3600.0

    new_arrived_at: list[float] = []
    max_shift_sec = 0.0
    final_shift_sec = 0.0
    if rows:
        previous = 0.0
        for i, (arrival, _) in enumerate(rows):
            if raw_span > 0:
                scaled = (arrival - raw_first) / raw_span * target_span_sec
            else:
                scaled = 0.0
            if i == 0:
                new_ts = 0.0
            else:
                new_ts = max(scaled, previous + min_gap_sec)
            shift = max(0.0, new_ts - scaled)
            max_shift_sec = max(max_shift_sec, shift)
            final_shift_sec = shift
            new_arrived_at.append(new_ts)
            previous = new_ts

    with output.open("w") as dst:
        for (_, record), new_ts in zip(rows, new_arrived_at, strict=True):
            out_record = dict(record)
            out_record["arrived_at"] = new_ts
            dst.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    compressed_span = new_arrived_at[-1] - new_arrived_at[0] if new_arrived_at else 0.0
    gaps = [
        b - a
        for a, b in zip(new_arrived_at, new_arrived_at[1:], strict=False)
    ]

    return {
        "mode": "smooth_target_span",
        "source": str(source),
        "output": str(output),
        "target_span_hours": float(target_span_hours),
        "target_span_sec": float(target_span_sec),
        "min_gap_sec": float(min_gap_sec),
        "total_requests": len(rows),
        "raw_first_arrived_at": raw_first,
        "raw_last_arrived_at": raw_last,
        "raw_span_sec": raw_span,
        "raw_span_hours": raw_span / 3600.0 if raw_span else 0.0,
        "compressed_span_sec": compressed_span,
        "compressed_span_hours": compressed_span / 3600.0 if compressed_span else 0.0,
        "max_schedule_shift_sec": max_shift_sec,
        "final_schedule_shift_sec": final_shift_sec,
        "min_observed_gap_sec": min(gaps) if gaps else None,
        "max_observed_gap_sec": max(gaps) if gaps else None,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compress long idle gaps in a real-eval trace while keeping "
            "intra-burst arrival structure intact."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Input JSONL trace (e.g. data/real_eval/burstgpt_day0_24h.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL trace.",
    )
    parser.add_argument(
        "--cap-gap-sec",
        type=float,
        default=10.0,
        help=(
            "Maximum inter-arrival gap, in seconds. Gaps <= cap are kept "
            "verbatim; gaps > cap are truncated to this value. Default 10s "
            "preserves typical burst structure and compresses BurstGPT day0 "
            "24h to roughly 8h wall-clock."
        ),
    )
    parser.add_argument(
        "--smooth-target-span-hours",
        type=float,
        default=None,
        help=(
            "Linearly compress the whole trace into this many hours and "
            "enforce --min-gap-sec between adjacent requests. This mode is "
            "intended for 24h->8h replay schedules that avoid same-timestamp "
            "bursts. When set, --cap-gap-sec is ignored."
        ),
    )
    parser.add_argument(
        "--min-gap-sec",
        type=float,
        default=0.0,
        help=(
            "Minimum inter-arrival gap. In default cap-gap mode this is applied "
            "after idle-gap compression, letting later idle gaps absorb burst "
            "shifts. With --smooth-target-span-hours it is enforced during the "
            "target-span rewrite."
        ),
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if not args.source.exists():
        raise FileNotFoundError(f"source trace not found: {args.source}")
    if args.smooth_target_span_hours is not None:
        stats = smooth_to_target_span(
            args.source,
            args.output,
            target_span_hours=args.smooth_target_span_hours,
            min_gap_sec=args.min_gap_sec,
        )
    else:
        stats = compress_idle_gaps(
            args.source,
            args.output,
            cap_gap_sec=args.cap_gap_sec,
            min_gap_sec=args.min_gap_sec,
        )
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
