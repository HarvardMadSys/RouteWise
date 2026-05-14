"""Cross-process latency-profile event log for real evaluation.

The real-eval launcher usually runs one process per policy. A plain in-memory
``broadcast`` only shares probe samples among policies in the same process, so
process-per-policy runs need a small shared event log to keep latency profiles
on the same evidence.
"""

from __future__ import annotations

import fcntl
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SharedProfileEvent:
    event_id: str
    ts: float
    provider: str
    ttft_ms: float
    error_type: str | None
    source: str
    policy: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SharedProfileEvent:
        return cls(
            event_id=str(raw["id"]),
            ts=float(raw["ts"]),
            provider=str(raw["provider"]),
            ttft_ms=float(raw["ttft_ms"]),
            error_type=(
                None
                if raw.get("error_type") in (None, "")
                else str(raw.get("error_type"))
            ),
            source=str(raw.get("source") or "unknown"),
            policy=None if raw.get("policy") in (None, "") else str(raw.get("policy")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.event_id,
            "ts": self.ts,
            "provider": self.provider,
            "ttft_ms": self.ttft_ms,
            "error_type": self.error_type,
            "source": self.source,
            "policy": self.policy,
        }


class SharedProfileEventLog:
    """Append-only JSONL event log with POSIX file locks."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._offset = 0
        self._seen_event_ids: set[str] = set()

    def append(
        self,
        *,
        provider: str,
        ts: float,
        ttft_ms: float,
        error_type: str | None,
        source: str,
        policy: str | None = None,
    ) -> SharedProfileEvent:
        event = SharedProfileEvent(
            event_id=uuid.uuid4().hex,
            ts=float(ts),
            provider=provider,
            ttft_ms=float(ttft_ms),
            error_type=error_type,
            source=source,
            policy=policy,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
        with self.path.open("a+", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                fh.write(line)
                fh.flush()
                # Mark the event as consumed by this log instance before
                # releasing the file lock. Otherwise this process's tailer can
                # read the just-appended line and double-apply it before the
                # writer marks it seen.
                self._seen_event_ids.add(event.event_id)
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
        return event

    def mark_seen(self, event_id: str) -> None:
        self._seen_event_ids.add(event_id)

    def read_new(self) -> list[SharedProfileEvent]:
        """Return events appended since the last read in this process."""
        if not self.path.exists():
            return []

        events: list[SharedProfileEvent] = []
        with self.path.open("r", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_SH)
            try:
                end = self.path.stat().st_size
                if self._offset > end:
                    self._offset = 0
                fh.seek(self._offset)
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    event = SharedProfileEvent.from_dict(json.loads(line))
                    if event.event_id in self._seen_event_ids:
                        continue
                    self._seen_event_ids.add(event.event_id)
                    events.append(event)
                self._offset = fh.tell()
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
        return events

    def recent_last_event_by_provider(
        self,
        *,
        now: float | None = None,
        window_sec: float = 15 * 60.0,
    ) -> dict[str, SharedProfileEvent]:
        """Return each provider's most recent event inside ``window_sec``."""
        if not self.path.exists():
            return {}
        current = time.time() if now is None else float(now)
        cutoff = current - float(window_sec)
        latest: dict[str, SharedProfileEvent] = {}
        with self.path.open("r", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_SH)
            try:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    event = SharedProfileEvent.from_dict(json.loads(line))
                    if event.ts < cutoff:
                        continue
                    previous = latest.get(event.provider)
                    if previous is None or event.ts > previous.ts:
                        latest[event.provider] = event
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
        return latest


__all__ = ["SharedProfileEvent", "SharedProfileEventLog"]
