#!/usr/bin/env python3
"""Generate a 7-day trace by combining BurstGPT timestamps with ShareGPT prompts.

Instead of looping the 5.84-day trace, this script:
1. Takes a 7-day window of arrival timestamps from BurstGPT (276K entries)
2. Pairs them with unique ShareGPT prompts (153K unique, cycling only after exhausted)
3. Tokenizes each ShareGPT prompt with tiktoken to get accurate num_prefill_tokens
4. Uses BurstGPT's response token counts for num_decode_tokens (output length distribution)

Since the evaluation only processes ~32K requests in 7 days (limited by API
throughput), and unique prompts number 153K, no prompt repetition occurs in
practice during the actual experiment run.

Usage:
    python experiment/scripts/make_7d_trace.py [--output PATH] [--days 7]
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import tiktoken

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BURSTGPT_CSV = PROJECT_ROOT / "data" / "BurstGPT_1.csv"
SHAREGPT_JSONL = (
    PROJECT_ROOT
    / "data"
    / "sharegpt_burstgpt"
    / "sharegpt_prompts_burstgpt_timestamps.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "sharegpt_burstgpt" / "sharegpt_prompts_7d.jsonl"
)

# The current trace starts at this BurstGPT timestamp (day ~14.5).
# We use the same starting point for consistency with 24h experiments.
TRACE_START_TS = 1255458


def load_sharegpt_prompts(path: Path) -> list[dict]:
    """Load unique ShareGPT prompts with accurate token counts.

    Uses tiktoken (cl100k_base) to count actual prompt tokens, ensuring
    num_prefill_tokens matches the real prompt content.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    seen = set()
    prompts = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            prompt_text = record.get("prompt_text", "")
            if prompt_text in seen:
                continue
            seen.add(prompt_text)
            prompts.append(
                {
                    "prompt_text": prompt_text,
                    "response_text": record.get("response_text", ""),
                    "num_prefill_tokens": len(enc.encode(prompt_text)),
                }
            )
    return prompts


def load_burstgpt_window(
    path: Path, start_ts: float, duration_sec: float
) -> list[dict]:
    """Load BurstGPT requests within a time window."""
    end_ts = start_ts + duration_sec
    entries = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = float(row["Timestamp"])
            if ts < start_ts:
                continue
            if ts > end_ts:
                break
            entries.append(
                {
                    "arrived_at": ts,
                    "request_tokens": int(row["Request tokens"]),
                    "response_tokens": int(row["Response tokens"]),
                }
            )
    return entries


def make_block_hash_ids(num_prefill_tokens: int, block_size: int = 16) -> str:
    """Generate block_hash_ids string from token count."""
    n_blocks = (num_prefill_tokens + block_size - 1) // block_size
    return json.dumps(list(range(n_blocks)))


def main():
    parser = argparse.ArgumentParser(
        description="Generate 7-day trace from BurstGPT + ShareGPT"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--days",
        type=float,
        default=7.0,
        help="Trace duration in days (default: 7.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for prompt shuffling (default: 42)",
    )
    args = parser.parse_args()

    duration_sec = args.days * 86400

    # Step 1: Load unique ShareGPT prompts
    print(f"Loading ShareGPT prompts from {SHAREGPT_JSONL}...")
    prompts = load_sharegpt_prompts(SHAREGPT_JSONL)
    print(f"  Loaded {len(prompts)} unique prompts")

    # Step 2: Load BurstGPT arrival timestamps for the 7-day window
    print(f"Loading BurstGPT timestamps from {BURSTGPT_CSV}...")
    print(f"  Window: ts={TRACE_START_TS} to ts={TRACE_START_TS + duration_sec}")
    print(f"  Duration: {args.days} days ({duration_sec:.0f}s)")
    arrivals = load_burstgpt_window(BURSTGPT_CSV, TRACE_START_TS, duration_sec)
    print(f"  Loaded {len(arrivals)} arrivals")

    if not arrivals:
        print("Error: No arrivals found in BurstGPT window")
        sys.exit(1)

    # Step 3: Shuffle prompts for diversity (deterministic with seed)
    rng = random.Random(args.seed)
    rng.shuffle(prompts)

    # Step 4: Combine arrivals with prompts
    print(f"Combining {len(arrivals)} arrivals with {len(prompts)} unique prompts...")
    n_prompts = len(prompts)
    recycled = 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for i, arrival in enumerate(arrivals):
            prompt_idx = i % n_prompts
            if i >= n_prompts and prompt_idx == 0:
                recycled += 1

            prompt = prompts[prompt_idx]
            # Use tokenized prompt length (accurate) instead of BurstGPT's
            # unrelated request token count.
            num_prefill = prompt["num_prefill_tokens"]
            # Use BurstGPT's response tokens for output length distribution.
            num_decode = arrival["response_tokens"]
            block_size = 16

            record = {
                "arrived_at": arrival["arrived_at"],
                "num_prefill_tokens": num_prefill,
                "num_decode_tokens": num_decode,
                "block_hash_ids": make_block_hash_ids(num_prefill, block_size),
                "block_size": block_size,
                "session_id": i,
                "prompt_text": prompt["prompt_text"],
                "response_text": prompt["response_text"],
            }
            f.write(json.dumps(record) + "\n")

    # Step 5: Verify
    span_sec = arrivals[-1]["arrived_at"] - arrivals[0]["arrived_at"]
    print(f"\nGenerated: {output_path}")
    print(f"  Entries: {len(arrivals)}")
    print(f"  Time span: {span_sec:.0f}s = {span_sec / 3600:.1f}h = {span_sec / 86400:.2f}d")
    print(f"  Unique prompts used: {min(len(arrivals), n_prompts)}")
    if len(arrivals) > n_prompts:
        print(
            f"  Prompts recycled: {len(arrivals) - n_prompts} "
            f"({(len(arrivals) - n_prompts) / len(arrivals) * 100:.1f}%)"
        )
        print(
            f"  NOTE: Eval processes ~32K requests in 7d, "
            f"well within {n_prompts} unique prompts (no repetition in practice)"
        )
    else:
        print("  No prompt recycling needed")


if __name__ == "__main__":
    main()
