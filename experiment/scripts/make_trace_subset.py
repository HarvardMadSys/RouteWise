#!/usr/bin/env python3
"""Extract a contiguous time window from day3_trace.jsonl with normalized arrival times.

Arrival times in the output are re-offset so that the first request in the
window has arrived_at=0. This lets phase5/phase6 drivers consume the subset
as a fresh trace without modifying their loaders.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Path to source trace JSONL (e.g. day3_trace.jsonl)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output trace JSONL (normalized arrival times)",
    )
    parser.add_argument(
        "--start-offset-sec",
        type=int,
        required=True,
        help="Seconds into source trace where the window begins",
    )
    parser.add_argument(
        "--window-sec",
        type=int,
        default=3600,
        help="Window length in seconds (default: 3600 for 1h)",
    )
    args = parser.parse_args()

    start = args.start_offset_sec
    end = start + args.window_sec

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with in_path.open() as f_in, out_path.open("w") as f_out:
        for line in f_in:
            req = json.loads(line)
            t = req["arrived_at"]
            if t < start:
                continue
            if t >= end:
                continue
            req["arrived_at"] = t - start
            f_out.write(json.dumps(req) + "\n")
            kept += 1

    print(f"Wrote {kept} requests to {out_path}")
    print(f"  source window: [{start}s, {end}s) = {args.window_sec}s")
    print(f"  normalized arrival times: [0, {args.window_sec}s)")


if __name__ == "__main__":
    main()
