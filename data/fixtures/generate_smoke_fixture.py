"""Regenerate the deterministic smoke-fixture workload.

The fixture is a tiny synthetic trace in the simulator workload schema
(`arrived_at`, `num_prefill_tokens`, `num_decode_tokens`, plus session
metadata). It exists so `python -m artifact smoke` can exercise the full
simulator pipeline with zero network downloads and zero API keys. It carries
no text payloads and is not derived from any external dataset.

Usage:
    python3 data/fixtures/generate_smoke_fixture.py

The output is committed; rerunning this script must reproduce it byte for
byte (fixed seed, stdlib-only randomness).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

FIXTURE_PATH = Path(__file__).resolve().parent / "burstgpt_smoke.jsonl"
NUM_REQUESTS = 120
SEED = 42

MEAN_ARRIVAL_GAP_SEC = 1.5
SESSION_CONTINUATION_PROBABILITY = 0.3
PREFILL_LOG_MU, PREFILL_LOG_SIGMA = 6.0, 0.8
DECODE_LOG_MU, DECODE_LOG_SIGMA = 5.0, 0.9
PREFILL_BOUNDS = (16, 8000)
DECODE_BOUNDS = (8, 4000)
DECODE_TOKENS_PER_SEC = 20.0


def _bounded_lognormal(rng: random.Random, mu: float, sigma: float, bounds: tuple[int, int]) -> int:
    low, high = bounds
    return max(low, min(high, round(rng.lognormvariate(mu, sigma))))


def generate_records() -> list[dict[str, object]]:
    rng = random.Random(SEED)
    records: list[dict[str, object]] = []
    arrived_at = 0.0
    session_index = 0
    turn_index = 0
    for _ in range(NUM_REQUESTS):
        arrived_at += rng.expovariate(1.0 / MEAN_ARRIVAL_GAP_SEC)
        if records and rng.random() < SESSION_CONTINUATION_PROBABILITY:
            turn_index += 1
        else:
            session_index += 1
            turn_index = 0
        prefill = _bounded_lognormal(rng, PREFILL_LOG_MU, PREFILL_LOG_SIGMA, PREFILL_BOUNDS)
        decode = _bounded_lognormal(rng, DECODE_LOG_MU, DECODE_LOG_SIGMA, DECODE_BOUNDS)
        records.append(
            {
                "arrived_at": round(arrived_at, 3),
                "session_id": f"smoke-session-{session_index:04d}",
                "num_prefill_tokens": prefill,
                "num_decode_tokens": decode,
                "model": "smoke-fixture",
                "log_type": "synthetic",
                "elapsed_time_sec": round(decode / DECODE_TOKENS_PER_SEC + 0.5, 3),
                "sharegpt_conversation_id": f"smoke-conv-{session_index:04d}",
                "sharegpt_turn_index": turn_index,
            }
        )
    return records


def main() -> int:
    records = generate_records()
    with FIXTURE_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"wrote {len(records)} requests to {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
