from __future__ import annotations

import json

import pytest

from serving.servers.sse import SSEParser


@pytest.mark.unit
def test_sse_parser_basic_and_split_chunks():
    parser = SSEParser()
    events = []

    # Basic
    events.extend(parser.feed(b'data: {"x":1}\n\n'))
    # Split across chunks
    events.extend(parser.feed(b'data: {"y"'))
    events.extend(parser.feed(b":2}\n\n"))

    payloads = [json.loads(e.data) for e in events if e.data and e.data != "[DONE]"]
    assert payloads == [{"x": 1}, {"y": 2}]


@pytest.mark.unit
def test_sse_parser_ignores_empty_and_handles_done():
    parser = SSEParser()
    events = []
    events.extend(parser.feed(b"data: \n\n"))  # empty data ignored
    events.extend(parser.feed(b"data: [DONE]\n\n"))

    # Parser surfaces empty-data events; stream_post filters them later.
    assert any(e.data == "" for e in events)
    assert any(e.data == "[DONE]" for e in events)


@pytest.mark.unit
def test_sse_parser_handles_utf8_boundary_split():
    """UTF-8 multibyte characters split across chunks are parsed correctly."""
    parser = SSEParser()
    events = []
    events.extend(parser.feed(b'data: {"text":"'))
    # '中' in UTF-8 split across chunk boundary
    events.extend(parser.feed(b"\xe4\xb8\xad"))
    events.extend(parser.feed(b'"}\n\n'))

    # Last event should be a valid JSON with the Chinese character
    payloads = [json.loads(e.data) for e in events if e.data and e.data != "[DONE]"]
    assert payloads[-1] == {"text": "中"}
