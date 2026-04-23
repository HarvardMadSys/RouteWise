from __future__ import annotations

import codecs
from dataclasses import dataclass


@dataclass
class SSEMessage:
    """Represents a single SSE frame."""

    data: str
    event: str | None = None
    id: str | None = None


class SSEParser:
    """Incremental parser for Server-Sent Events streams."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._buffer: str = ""

    def feed(self, chunk: bytes) -> list[SSEMessage]:
        """Feed raw bytes and yield complete SSE messages when available."""

        try:
            decoded = self._decoder.decode(chunk, final=False)
        except UnicodeDecodeError:
            return []

        if decoded:
            # Providers use CRLF; normalise to LF so frame boundaries are stable.
            self._buffer += decoded.replace("\r\n", "\n")

        messages: list[SSEMessage] = []
        while "\n\n" in self._buffer:
            frame, self._buffer = self._buffer.split("\n\n", 1)
            if not frame.strip():
                continue
            messages.append(self._parse_frame(frame))
        return messages

    @staticmethod
    def _parse_frame(frame: str) -> SSEMessage:
        data_lines: list[str] = []
        event: str | None = None
        msg_id: str | None = None

        for raw_line in frame.splitlines():
            line = raw_line.rstrip("\r")
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
            elif line.startswith("event:"):
                event = line[6:].lstrip()
            elif line.startswith("id:"):
                msg_id = line[3:].lstrip()

        return SSEMessage(data="\n".join(data_lines), event=event, id=msg_id)
