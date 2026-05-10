from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.slice_real_eval_trace import SECONDS_PER_DAY, SECONDS_PER_HOUR, slice_trace

if TYPE_CHECKING:
    from pathlib import Path


def test_slice_trace_rebases_window_and_preserves_payload(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "out.jsonl"
    window_start = 2 * SECONDS_PER_DAY + 17 * SECONDS_PER_HOUR
    rows = [
        {"arrived_at": window_start - 1, "prompt_text": "before"},
        {
            "arrived_at": window_start + 10,
            "prompt_text": "first",
            "sharegpt_conversation_id": "sg-1",
            "num_prefill_tokens": 10,
            "num_decode_tokens": 2,
        },
        {
            "arrived_at": window_start + SECONDS_PER_HOUR + 5,
            "prompt_text": "second",
            "sharegpt_conversation_id": "sg-2",
            "num_prefill_tokens": 20,
            "num_decode_tokens": 4,
        },
        {"arrived_at": window_start + 2 * SECONDS_PER_HOUR, "prompt_text": "after"},
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    stats = slice_trace(source, output, day=2, start_hour=17, hours=2)

    written = [json.loads(line) for line in output.read_text().splitlines()]
    assert stats["total_requests"] == 2
    assert stats["requests_per_hour"] == [1, 1]
    assert [row["arrived_at"] for row in written] == [10.0, 3605.0]
    assert [row["prompt_text"] for row in written] == ["first", "second"]
    assert written[0]["sharegpt_conversation_id"] == "sg-1"
