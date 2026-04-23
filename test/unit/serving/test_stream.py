from __future__ import annotations

import json

import pytest

from serving.stream import make_final_usage_chunk, make_stream_chunk


@pytest.mark.unit
def test_make_stream_chunk_structure():
    line = make_stream_chunk(model="m", content="hi")
    assert line.startswith("data: ")
    payload = json.loads(line[len("data: ") :])
    assert payload["object"] == "chat.completion.chunk"
    assert payload["model"] == "m"
    assert payload["choices"][0]["delta"]["content"] == "hi"


@pytest.mark.unit
def test_make_final_usage_chunk_estimates_tokens():
    messages = [{"role": "user", "content": "hello"}]
    # total_content with 2 words should produce non-zero completion tokens
    line = make_final_usage_chunk(model="m", messages=messages, total_content="ok then")
    payload = json.loads(line[len("data: ") :])
    usage = payload["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


@pytest.mark.unit
@pytest.mark.perf
def test_make_stream_chunk_performance():
    """Ensure generating many chunks is fast enough for regressions."""
    import time

    start = time.perf_counter()
    for _ in range(1000):
        _ = make_stream_chunk(model="m", content="content")
    elapsed = time.perf_counter() - start
    assert elapsed < 0.1  # 1000 chunks within 100ms on CI-class machines
