#!/usr/bin/env python3
"""Create a looped JSONL trace by appending the trace to itself with offset timestamps.

The second copy starts right after the first copy ends, preserving the original
arrival pattern and token distribution. This allows running longer experiments
(e.g., 7 days) from a shorter trace (e.g., 5.84 days).

Usage:
    python experiment/scripts/make_looped_trace.py \
        --input data/sharegpt_burstgpt/sharegpt_prompts_burstgpt_timestamps.jsonl \
        --output data/sharegpt_burstgpt/sharegpt_prompts_7d_looped.jsonl \
        --target-days 7
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Loop a JSONL trace to reach target duration.")
    parser.add_argument("--input", required=True, help="Input JSONL trace file")
    parser.add_argument("--output", required=True, help="Output JSONL trace file")
    parser.add_argument("--target-days", type=float, default=7.0, help="Target duration in days")
    parser.add_argument("--gap-sec", type=float, default=60.0,
                        help="Gap between loops in seconds (default: 60)")
    args = parser.parse_args()

    target_sec = args.target_days * 86400

    # First pass: read all records and find duration
    print(f"Reading {args.input}...")
    records = []
    first_ts = None
    last_ts = None
    for line in open(args.input):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = record.get("arrived_at", 0)
        if first_ts is None:
            first_ts = ts
        last_ts = ts
        records.append(record)

    duration = last_ts - first_ts
    print(f"  {len(records)} records, duration: {duration:.0f}s ({duration/86400:.2f} days)")

    if duration >= target_sec:
        print(f"  Trace already covers {args.target_days} days. Copying as-is.")
        with open(args.output, "w") as out:
            for r in records:
                out.write(json.dumps(r) + "\n")
        return

    # Calculate how many loops needed
    loop_period = duration + args.gap_sec
    n_loops = 1
    while loop_period * n_loops < target_sec + duration:
        n_loops += 1

    total_records = 0
    print(f"  Writing {n_loops} loops (period={loop_period:.0f}s) to {args.output}...")

    with open(args.output, "w") as out:
        for loop_idx in range(n_loops):
            offset = loop_idx * loop_period
            for record in records:
                new_ts = record["arrived_at"] - first_ts + offset + first_ts
                if new_ts - first_ts > target_sec:
                    break
                new_record = dict(record)
                new_record["arrived_at"] = new_ts
                out.write(json.dumps(new_record) + "\n")
                total_records += 1
            # Check if we've reached target
            if offset + duration >= target_sec:
                break

    final_duration = (new_ts - first_ts) / 86400
    print(f"  Done: {total_records} records, {final_duration:.2f} days")


if __name__ == "__main__":
    main()
