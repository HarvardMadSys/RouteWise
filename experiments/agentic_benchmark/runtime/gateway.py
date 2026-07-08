"""Client for the live hybridInference gateway: send chat, read request logs back."""

from __future__ import annotations

import contextlib
import json
import time
from datetime import datetime, timezone
from typing import Any

import requests

from .config import GatewayConfig
from .schemas import ChatResult, LLMRequestRecord


class GatewayError(RuntimeError):
    """Raised when a gateway call returns an unexpected status."""


def _iso_utc(dt: datetime) -> str:
    """Render a timezone-aware ISO8601 string the gateway accepts."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


class GatewayClient:
    """Talks to freeinference.org: chat completions out, request logs back in.

    Chat calls authenticate with the public ``api_key``. Reading per-request
    logs uses ``admin_api_key`` (ADMIN_TOKEN) against ``/admin/export/requests``,
    since the user dashboard log endpoint requires a dashboard JWT rather than an
    API key.
    """

    def __init__(self, config: GatewayConfig) -> None:
        self._config = config
        self._root = config.base_url.rstrip("/").removesuffix("/v1")

    @classmethod
    def from_env(cls, **overrides: object) -> GatewayClient:
        """Build a client from environment-derived gateway config."""
        return cls(GatewayConfig.from_env(**overrides))

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        session_id: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        route_pin: str | None = None,
        api_key: str | None = None,
    ) -> ChatResult:
        """Send one non-streaming chat completion, tagging it with ``session_id``."""
        key = api_key or self._config.require_api_key()
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "X-Session-ID": session_id,
        }
        if route_pin:
            headers["X-Route-Pin"] = route_pin
        payload = {
            "model": self._config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        started = time.time()
        resp = requests.post(
            f"{self._root}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=self._config.timeout_sec,
        )
        finished = time.time()
        body: dict[str, Any] = {}
        with contextlib.suppress(ValueError):
            body = resp.json()
        content = ""
        choices = body.get("choices") or []
        if choices:
            content = (choices[0].get("message") or {}).get("content") or ""
        return ChatResult(
            session_id=session_id,
            request_id=body.get("id"),
            status_code=resp.status_code,
            started_at=started,
            finished_at=finished,
            content=content,
            raw=body,
        )

    def export_requests(
        self,
        *,
        start_time: datetime,
        end_time: datetime | None = None,
        model_id: str | None = None,
        include_content: bool = True,
    ) -> list[LLMRequestRecord]:
        """Pull per-request log rows via the admin export endpoint (ndjson stream)."""
        if not self._config.admin_api_key:
            raise GatewayError(
                "No admin token found. Set ADMIN_TOKEN (or FREEINFERENCE_ADMIN_API_KEY) "
                "to read per-request logs from /admin/export/requests."
            )
        params = {
            "start_time": _iso_utc(start_time),
            "include_content": str(include_content).lower(),
        }
        if end_time is not None:
            params["end_time"] = _iso_utc(end_time)
        if model_id:
            params["model_id"] = model_id
        headers = {"Authorization": f"Bearer {self._config.admin_api_key}"}
        resp = requests.get(
            f"{self._root}/admin/export/requests",
            headers=headers,
            params=params,
            timeout=self._config.timeout_sec,
        )
        if resp.status_code != 200:
            raise GatewayError(
                f"export/requests returned {resp.status_code}: {resp.text[:200]}"
            )
        records: list[LLMRequestRecord] = []
        for line in resp.text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            records.append(LLMRequestRecord.from_api(json.loads(stripped)))
        return records

    def export_by_session(
        self,
        session_id: str,
        *,
        start_time: datetime,
        end_time: datetime | None = None,
        model_id: str | None = None,
    ) -> list[LLMRequestRecord]:
        """Return only the rows tagged with ``session_id`` (needs include_content).

        ``model_id`` narrows the export server-side; pass the model actually used
        for this session (e.g. ``minimax-fast`` for a RouteWise run) rather than
        relying on the client's default model.
        """
        rows = self.export_requests(
            start_time=start_time,
            end_time=end_time,
            model_id=model_id or self._config.model,
            include_content=True,
        )
        return [r for r in rows if r.session_id == session_id]
