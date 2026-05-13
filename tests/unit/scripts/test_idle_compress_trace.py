from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts.idle_compress_trace import compress_idle_gaps, smooth_to_target_span

if TYPE_CHECKING:
    from pathlib import Path


def _write_trace(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_compress_idle_gaps_preserves_short_gaps_and_caps_long_ones(
    tmp_path: Path,
) -> None:
    """Inter-arrival gaps <= cap are kept verbatim; gaps > cap are truncated."""
    source = tmp_path / "in.jsonl"
    output = tmp_path / "out.jsonl"
    rows = [
        {"arrived_at": 0.0, "prompt_text": "a", "num_prefill_tokens": 1, "num_decode_tokens": 1},
        {"arrived_at": 0.5, "prompt_text": "b", "num_prefill_tokens": 2, "num_decode_tokens": 1},
        {"arrived_at": 100.5, "prompt_text": "c", "num_prefill_tokens": 3, "num_decode_tokens": 1},
        {"arrived_at": 101.2, "prompt_text": "d", "num_prefill_tokens": 4, "num_decode_tokens": 1},
    ]
    _write_trace(source, rows)

    stats = compress_idle_gaps(source, output, cap_gap_sec=10.0)

    written = [json.loads(line) for line in output.read_text().splitlines()]
    # gap 0.5 stays as 0.5; gap 100.0 caps to 10.0; gap 0.7 stays as 0.7.
    assert [row["arrived_at"] for row in written] == pytest.approx([0.0, 0.5, 10.5, 11.2])
    assert [row["prompt_text"] for row in written] == ["a", "b", "c", "d"]
    assert stats["total_requests"] == 4
    assert stats["gaps_untouched"] == 2
    assert stats["gaps_capped"] == 1
    assert stats["wall_clock_savings_sec"] == pytest.approx(90.0)
    # Compressed span = (0.5 - 0.0) + 10.0 + (101.2 - 100.5) = 11.2
    assert stats["compressed_span_sec"] == pytest.approx(11.2)


def test_compress_idle_gaps_sorts_input_by_arrived_at(tmp_path: Path) -> None:
    """Out-of-order rows must be sorted before gap calculation."""
    source = tmp_path / "in.jsonl"
    output = tmp_path / "out.jsonl"
    rows = [
        {"arrived_at": 100.0, "prompt_text": "c", "num_prefill_tokens": 3, "num_decode_tokens": 1},
        {"arrived_at": 0.0, "prompt_text": "a", "num_prefill_tokens": 1, "num_decode_tokens": 1},
        {"arrived_at": 2.0, "prompt_text": "b", "num_prefill_tokens": 2, "num_decode_tokens": 1},
    ]
    _write_trace(source, rows)

    compress_idle_gaps(source, output, cap_gap_sec=5.0)

    written = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["prompt_text"] for row in written] == ["a", "b", "c"]
    # Gap 0.0->2.0 kept (2.0 <= 5.0); 2.0->100.0 capped to 5.0.
    assert [row["arrived_at"] for row in written] == [0.0, 2.0, 7.0]


def test_compress_idle_gaps_can_enforce_min_gap_after_capping(
    tmp_path: Path,
) -> None:
    source = tmp_path / "in.jsonl"
    output = tmp_path / "out.jsonl"
    rows = [
        {"arrived_at": 0.0, "prompt_text": "a"},
        {"arrived_at": 0.0, "prompt_text": "b"},
        {"arrived_at": 0.2, "prompt_text": "c"},
        {"arrived_at": 100.0, "prompt_text": "d"},
    ]
    _write_trace(source, rows)

    stats = compress_idle_gaps(
        source,
        output,
        cap_gap_sec=10.0,
        min_gap_sec=1.0,
    )

    written = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["prompt_text"] for row in written] == ["a", "b", "c", "d"]
    # The burst shift is absorbed by the following idle gap, so the last
    # request keeps its cap-compressed base timestamp instead of sliding by
    # the full amount of the earlier burst expansion.
    assert [row["arrived_at"] for row in written] == pytest.approx(
        [0.0, 1.0, 2.0, 10.2]
    )
    assert stats["mode"] == "cap_idle_gaps"
    assert stats["min_gap_sec"] == pytest.approx(1.0)
    assert stats["min_gap_shifts"] == 2
    assert stats["max_min_gap_shift_sec"] == pytest.approx(1.8)


def test_compress_idle_gaps_preserves_full_record_payload(tmp_path: Path) -> None:
    """Every original field except arrived_at must be carried over unchanged."""
    source = tmp_path / "in.jsonl"
    output = tmp_path / "out.jsonl"
    payload = {
        "arrived_at": 5.0,
        "session_id": "abc-123",
        "num_prefill_tokens": 100,
        "num_decode_tokens": 50,
        "prompt_text": "hello",
        "metadata": {"nested": True},
    }
    _write_trace(source, [payload])

    compress_idle_gaps(source, output, cap_gap_sec=10.0)

    written = [json.loads(line) for line in output.read_text().splitlines()]
    assert written[0]["session_id"] == "abc-123"
    assert written[0]["num_prefill_tokens"] == 100
    assert written[0]["num_decode_tokens"] == 50
    assert written[0]["prompt_text"] == "hello"
    assert written[0]["metadata"] == {"nested": True}
    # First request is rebased to 0.0 (no preceding gap).
    assert written[0]["arrived_at"] == 0.0


def test_compress_idle_gaps_rejects_nonpositive_cap(tmp_path: Path) -> None:
    source = tmp_path / "in.jsonl"
    output = tmp_path / "out.jsonl"
    _write_trace(source, [{"arrived_at": 0.0, "num_prefill_tokens": 1, "num_decode_tokens": 1}])
    with pytest.raises(ValueError, match="cap_gap_sec"):
        compress_idle_gaps(source, output, cap_gap_sec=0.0)
    with pytest.raises(ValueError, match="cap_gap_sec"):
        compress_idle_gaps(source, output, cap_gap_sec=-1.0)
    with pytest.raises(ValueError, match="min_gap_sec"):
        compress_idle_gaps(source, output, cap_gap_sec=10.0, min_gap_sec=-0.1)


def test_smooth_to_target_span_enforces_minimum_gap(tmp_path: Path) -> None:
    source = tmp_path / "in.jsonl"
    output = tmp_path / "out.jsonl"
    rows = [
        {"arrived_at": 0.0, "prompt_text": "a"},
        {"arrived_at": 0.0, "prompt_text": "b"},
        {"arrived_at": 10.0, "prompt_text": "c"},
    ]
    _write_trace(source, rows)

    stats = smooth_to_target_span(
        source,
        output,
        target_span_hours=10.0 / 3600.0,
        min_gap_sec=1.0,
    )

    written = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["prompt_text"] for row in written] == ["a", "b", "c"]
    assert [row["arrived_at"] for row in written] == pytest.approx([0.0, 1.0, 10.0])
    assert stats["mode"] == "smooth_target_span"
    assert stats["min_observed_gap_sec"] == pytest.approx(1.0)
    assert stats["compressed_span_sec"] == pytest.approx(10.0)


def test_smooth_to_target_span_rejects_invalid_parameters(tmp_path: Path) -> None:
    source = tmp_path / "in.jsonl"
    output = tmp_path / "out.jsonl"
    _write_trace(source, [{"arrived_at": 0.0}])

    with pytest.raises(ValueError, match="target_span_hours"):
        smooth_to_target_span(source, output, target_span_hours=0.0, min_gap_sec=0.1)
    with pytest.raises(ValueError, match="min_gap_sec"):
        smooth_to_target_span(source, output, target_span_hours=1.0, min_gap_sec=-0.1)
