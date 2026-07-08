"""Pipeline gate: one live minimax-m2.5 call, then read its cost/latency back.

Run::

    python -m experiments.agentic_benchmark.runtime.verify_pipeline

Sends a single chat completion through freeinference.org tagged with a unique
``X-Session-ID``, then pulls the matching row from ``/admin/export/requests`` and
prints the gateway-recorded provider, cost, and latency. Proves the data path
works before any agent or SWE-bench machinery is built on top.
"""

from __future__ import annotations

import sys
import time
import uuid
from datetime import datetime, timedelta, timezone

from .config import GatewayConfig
from .gateway import GatewayClient, GatewayError


def main() -> int:
    """Send one tagged request and confirm it is retrievable with cost/latency."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    config = GatewayConfig.from_env()
    client = GatewayClient(config)
    session_id = f"probe-{uuid.uuid4().hex[:12]}"

    print(f"gateway   : {config.base_url}")
    print(f"model     : {config.model}")
    print(f"session_id: {session_id}")
    print(f"api_key   : {'set' if config.api_key else 'MISSING'}")
    print(f"admin_key : {'set' if config.admin_api_key else 'MISSING'}")

    chat_key = config.api_key or config.admin_api_key
    if not config.api_key and config.admin_api_key:
        print("\nWARN: public api_key missing; using ADMIN_TOKEN for the chat call.")
        print("      (probe only -- real RouteWise runs need a public hyi- key.)")

    window_start = datetime.now(timezone.utc) - timedelta(minutes=1)
    try:
        result = client.chat_completion(
            messages=[{"role": "user", "content": "Reply with the single word: ok."}],
            session_id=session_id,
            max_tokens=16,
            api_key=chat_key,
        )
    except Exception as exc:
        print(f"\nchat call FAILED: {exc}")
        return 1

    print(f"\nchat status : {result.status_code}")
    print(f"response id : {result.request_id}")
    print(f"wall latency: {result.wall_latency_ms:.0f} ms")
    print(f"content     : {result.content[:80]!r}")
    if result.status_code != 200:
        print(f"raw         : {result.raw}")
        return 1

    matched: list = []
    for attempt in range(6):
        time.sleep(2.0)
        try:
            matched = client.export_by_session(session_id, start_time=window_start)
        except GatewayError as exc:
            print(f"\nexport FAILED: {exc}")
            return 1
        if matched:
            break
        print(f"  ... no row yet (attempt {attempt + 1}/6)")

    if not matched:
        print("\nNO matching row in /admin/export/requests within timeout.")
        print("Pipeline gate FAILED: request was sent but not retrievable by session_id.")
        return 1

    print(f"\nfound {len(matched)} request row(s) for this session:")
    for rec in matched:
        print(f"  request_id : {rec.request_id}")
        print(f"  provider   : {rec.provider}")
        print(f"  cost_usd   : {rec.cost_usd}")
        print(f"  latency_ms : {rec.latency_ms}")
        print(f"  ttft_ms    : {rec.ttft_ms}")
        print(f"  tokens     : prompt={rec.prompt_tokens} completion={rec.completion_tokens}")
        print(f"  routewise  : {rec.routewise}")
    print("\nPipeline gate PASSED: cost/latency/provider are retrievable per session.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
