"""Tests for the BurstGPT + ShareGPT workload composer."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

from scripts.prepare_workload import compose_rednote_token_workload, compose_workload

if TYPE_CHECKING:
    from pathlib import Path


def _write_burstgpt_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_burstgpt(path: Path) -> None:
    _write_burstgpt_rows(
        path,
        [
        {
            "Timestamp": "100",
            "Session ID": "burst-a",
            "Elapsed time": "1",
            "Model": "GPT-4",
            "Request tokens": "10",
            "Response tokens": "2",
            "Total tokens": "12",
            "Log Type": "Conversation log",
        },
        {
            "Timestamp": "101",
            "Session ID": "",
            "Elapsed time": "1",
            "Model": "ChatGPT",
            "Request tokens": "11",
            "Response tokens": "3",
            "Total tokens": "14",
            "Log Type": "API log",
        },
        {
            "Timestamp": "102",
            "Session ID": "burst-a",
            "Elapsed time": "1",
            "Model": "GPT-4",
            "Request tokens": "12",
            "Response tokens": "4",
            "Total tokens": "16",
            "Log Type": "Conversation log",
        },
        {
            "Timestamp": "103",
            "Session ID": "burst-b",
            "Elapsed time": "1",
            "Model": "ChatGPT",
            "Request tokens": "13",
            "Response tokens": "5",
            "Total tokens": "18",
            "Log Type": "Conversation log",
        },
        {
            "Timestamp": "104",
            "Session ID": "burst-a",
            "Elapsed time": "1",
            "Model": "GPT-4",
            "Request tokens": "14",
            "Response tokens": "6",
            "Total tokens": "20",
            "Log Type": "Conversation log",
        },
        ],
    )


def _write_sharegpt(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "id": "sg-a",
                    "conversations": [
                        {"from": "human", "value": "A first"},
                        {"from": "gpt", "value": "A answer"},
                        {"from": "human", "value": "A second"},
                        {"from": "gpt", "value": "A second answer"},
                    ],
                },
                {
                    "id": "sg-api",
                    "conversations": [
                        {"from": "human", "value": "API prompt"},
                        {"from": "gpt", "value": "API answer"},
                    ],
                },
                {
                    "id": "sg-b",
                    "conversations": [
                        {"from": "human", "value": "B first"},
                        {"from": "gpt", "value": "B answer"},
                    ],
                },
            ]
        )
    )


def _write_rednote(path: Path) -> None:
    rows = [
        {
            "timestamp": "2025-10-16 14:26:27.037363+00",
            "request_id": "rn-1",
            "model_id": "glm-4.5",
            "provider": "router",
            "prompt_tokens": "1000",
            "completion_tokens": "200",
            "total_tokens": "1200",
            "prompt": json.dumps(
                [
                    {"role": "system", "content": "You are a coding assistant."},
                    {"role": "user", "content": "Fix the failing test."},
                ]
            ),
            "response": json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "The failing assertion is fixed.",
                            }
                        }
                    ]
                }
            ),
            "prompt_hash": "hash-1",
        },
        {
            "timestamp": "2025-10-16 14:27:27.037363+00",
            "request_id": "rn-2",
            "model_id": "claude-4.5-sonnet",
            "provider": "anthropic",
            "prompt_tokens": "3000",
            "completion_tokens": "900",
            "total_tokens": "3900",
            "prompt": json.dumps(
                [{"role": "user", "content": [{"type": "text", "text": "Review this diff."}]}]
            ),
            "response": "Looks good.",
            "prompt_hash": "hash-2",
        },
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_compose_keeps_sharegpt_conversation_within_burst_session(tmp_path: Path):
    burstgpt = tmp_path / "BurstGPT_3.csv"
    sharegpt = tmp_path / "sharegpt.json"
    output = tmp_path / "workload.jsonl"
    _write_burstgpt(burstgpt)
    _write_sharegpt(sharegpt)

    compose_workload(burstgpt, sharegpt, output, days=1)

    records = [json.loads(line) for line in output.read_text().splitlines()]
    burst_a = [rec for rec in records if rec["session_id"] == "burst-a"]

    assert [rec["sharegpt_conversation_id"] for rec in burst_a] == [
        "sg-a",
        "sg-a",
        "sg-a",
    ]
    assert [rec["sharegpt_turn_index"] for rec in burst_a] == [0, 1, 1]
    assert burst_a[1]["prompt_text"].startswith(burst_a[0]["prompt_text"])
    assert burst_a[2]["prompt_text"] == burst_a[1]["prompt_text"]

    api_record = records[1]
    assert api_record["session_id"] == "api:1"
    assert api_record["sharegpt_conversation_id"] == "sg-api"


def test_compose_rednote_uses_burst_arrivals_with_rednote_token_pairs(tmp_path: Path):
    burstgpt = tmp_path / "BurstGPT_3.csv"
    rednote = tmp_path / "rednote.csv"
    output = tmp_path / "burstgpt_rednote.jsonl"
    _write_burstgpt(burstgpt)
    _write_rednote(rednote)

    stats = compose_rednote_token_workload(
        burstgpt,
        rednote,
        output,
        days=1,
        max_requests=3,
    )

    records = [json.loads(line) for line in output.read_text().splitlines()]

    assert stats["total_requests"] == 3
    assert stats["rednote_cycles"] == 1
    assert [record["arrived_at"] for record in records] == [0.0, 1.0, 2.0]
    assert [record["session_id"] for record in records] == ["burst-a", "api:1", "burst-a"]
    assert [record["num_prefill_tokens"] for record in records] == [1000, 3000, 1000]
    assert [record["num_decode_tokens"] for record in records] == [200, 900, 200]
    assert records[0]["prompt_text"] == (
        "System: You are a coding assistant.\n\nUser: Fix the failing test."
    )
    assert records[0]["response_text"] == "The failing assertion is fixed."
    assert records[1]["prompt_text"] == "User: Review this diff."
    assert records[2]["rednote_request_id"] == "rn-1"


def test_compose_rednote_can_write_simulator_only_token_pairs(tmp_path: Path):
    burstgpt = tmp_path / "BurstGPT_3.csv"
    rednote = tmp_path / "rednote.csv"
    output = tmp_path / "burstgpt_rednote_tokens.jsonl"
    _write_burstgpt(burstgpt)
    _write_rednote(rednote)

    compose_rednote_token_workload(
        burstgpt,
        rednote,
        output,
        days=1,
        max_requests=1,
        include_text=False,
    )

    record = json.loads(output.read_text().splitlines()[0])
    assert record["num_prefill_tokens"] == 1000
    assert record["num_decode_tokens"] == 200
    assert "prompt_text" not in record
    assert "response_text" not in record
