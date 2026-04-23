#!/usr/bin/env python3
"""Probe a conservative Ollama Cloud quota lower bound for a fixed workload.

This script sends non-streaming chat requests directly to ``https://ollama.com/api/chat``
using ``OLLAMA_API_KEY`` authentication and records Ollama's usage metrics:

- ``total_duration``
- ``load_duration``
- ``prompt_eval_count``
- ``prompt_eval_duration``
- ``eval_count``
- ``eval_duration``

The primary use case is estimating a safe lower bound on the usable session
budget of an Ollama Cloud Pro plan under the workload used in our experiments.
To avoid conflating quota with concurrency, the script runs one outstanding
request at a time and is intended for a fixed prompt set.

Example:
    source .venv/bin/activate
    export OLLAMA_API_KEY=...
    python scripts/perf/ollama_quota_probe.py \
      --model glm-4.7:cloud \
      --prompt "Summarize why tail latency matters in LLM serving." \
      --output-dir experiment/results/ollama_quota_probe/pro_run1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass


DEFAULT_BASE_URL = "https://ollama.com/api"
DEFAULT_TIMEOUT_SECONDS = 600.0
DEFAULT_MAX_REQUESTS = 100_000
DEFAULT_MAX_DURATION_MINUTES = 310.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 5.0
DEFAULT_OUTPUT_ROOT = "experiment/results/ollama_quota_probe"
TRANSIENT_STATUSES = {500, 502, 503, 504}


@dataclass
class PromptExample:
    """One request example in the probe workload."""

    messages: list[dict[str, str]]
    label: str


@dataclass
class RequestRecord:
    """One probe request and its observed usage."""

    index: int
    label: str
    status: int
    ok: bool
    limit_hit: bool
    error: str | None
    wall_time_seconds: float
    total_duration_ns: int
    load_duration_ns: int
    prompt_eval_count: int
    prompt_eval_duration_ns: int
    eval_count: int
    eval_duration_ns: int
    effective_usage_ns: int
    effective_usage_source: str
    response_chars: int
    created_at: str | None


@dataclass
class SharedState:
    """Thread-safe state for concurrent quota probing."""

    start_time: float
    max_requests: int
    max_duration_minutes: float
    examples: list[PromptExample]
    lock: threading.Lock
    stop_event: threading.Event
    next_index: int = 1
    records: list[RequestRecord] | None = None
    limit_record: RequestRecord | None = None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Probe a conservative Ollama Cloud session quota lower bound."
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL for Ollama's authenticated API (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Cloud model name to probe, e.g. glm-4.7:cloud or "
            "minimax-m2.5:cloud."
        ),
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Single user prompt. Ignored when --prompt-file is provided.",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help=(
            "Optional workload file. Plain text: one prompt per non-empty line. "
            "JSONL: each line may contain {'prompt': ...} or {'messages': [...]}."
        ),
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt prepended to every example.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max generated tokens per request (default: 512).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (default: 0.0 for reproducibility).",
    )
    parser.add_argument(
        "--keep-alive",
        default="10m",
        help="Ollama keep_alive value (default: 10m). Use 0 for worst-case cold loads.",
    )
    parser.add_argument(
        "--think",
        default=None,
        help='Optional think mode. Examples: "low", "medium", "high", "true".',
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Sleep between successful requests (default: 0.0).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "Number of concurrent workers. Use 1 to isolate quota from throughput. "
            "Use 3 to saturate Ollama Pro's documented concurrency limit."
        ),
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=DEFAULT_MAX_REQUESTS,
        help=f"Hard stop on the number of requests (default: {DEFAULT_MAX_REQUESTS}).",
    )
    parser.add_argument(
        "--max-duration-minutes",
        type=float,
        default=DEFAULT_MAX_DURATION_MINUTES,
        help=(
            "Hard wall-clock stop in minutes. Keep slightly above 300 to cover a "
            "full 5-hour session window (default: 310)."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"HTTP timeout per request in seconds (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Retries for transient 5xx errors before aborting (default: 3).",
    )
    parser.add_argument(
        "--backoff-seconds",
        type=float,
        default=DEFAULT_BACKOFF_SECONDS,
        help="Base backoff for transient retries (default: 5.0).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for JSONL request logs and summary JSON. Auto-generated if omitted.",
    )
    parser.add_argument(
        "--session-start-pct",
        type=float,
        default=None,
        help="Dashboard 5-hour session usage percentage before the run, e.g. 0.0 or 12.3.",
    )
    parser.add_argument(
        "--session-end-pct",
        type=float,
        default=None,
        help="Dashboard 5-hour session usage percentage after the run.",
    )
    parser.add_argument(
        "--weekly-start-pct",
        type=float,
        default=None,
        help="Dashboard weekly usage percentage before the run.",
    )
    parser.add_argument(
        "--weekly-end-pct",
        type=float,
        default=None,
        help="Dashboard weekly usage percentage after the run.",
    )
    return parser.parse_args()


def _coerce_think(value: str | None) -> bool | str | None:
    """Convert CLI think argument to a valid API payload value."""
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    return value


def _default_output_dir() -> pathlib.Path:
    """Build a timestamped default output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return pathlib.Path(DEFAULT_OUTPUT_ROOT) / f"run_{timestamp}"


def _build_messages(prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
    """Build the standard message list for one prompt."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def load_examples(prompt: str | None, prompt_file: str | None, system_prompt: str | None) -> list[PromptExample]:
    """Load one or more examples from CLI inputs."""
    if prompt_file:
        path = pathlib.Path(prompt_file)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")

        examples: list[PromptExample] = []
        if path.suffix.lower() == ".jsonl":
            with path.open() as handle:
                for index, raw_line in enumerate(handle, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    label = str(payload.get("label") or f"line_{index}")
                    if "messages" in payload:
                        messages = list(payload["messages"])
                        if system_prompt:
                            messages = [{"role": "system", "content": system_prompt}, *messages]
                    elif "prompt" in payload:
                        messages = _build_messages(str(payload["prompt"]), system_prompt)
                    else:
                        raise ValueError(
                            "JSONL prompt entries must contain either 'messages' or 'prompt'."
                        )
                    examples.append(PromptExample(messages=messages, label=label))
        else:
            with path.open() as handle:
                for index, raw_line in enumerate(handle, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    examples.append(
                        PromptExample(
                            messages=_build_messages(line, system_prompt),
                            label=f"line_{index}",
                        )
                    )
        if not examples:
            raise ValueError(f"No prompts loaded from {path}")
        return examples

    if prompt is None:
        prompt = "Write a concise explanation of why latency tail behavior matters in LLM serving."

    return [PromptExample(messages=_build_messages(prompt, system_prompt), label="single_prompt")]


def percentile(values: list[int], pct: float) -> float:
    """Compute a percentile with linear interpolation."""
    if not values:
        return 0.0

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    rank = (len(sorted_values) - 1) * (pct / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(sorted_values[lower])

    weight = rank - lower
    return (
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


def is_limit_error(status: int, error_message: str | None) -> bool:
    """Heuristically identify quota/usage-limit failures."""
    if status == 429:
        return True
    if status in {401, 403} and error_message:
        lowered = error_message.lower()
        return "limit" in lowered or "usage" in lowered or "quota" in lowered
    return False


def _ui_delta_bounds(start_pct: float | None, end_pct: float | None) -> tuple[float | None, float | None]:
    """Return lower/upper bounds on percentage delta under 0.1% dashboard quantization.

    We assume displayed values are rounded to one decimal place. For a displayed
    percentage x, the true value lies in [x - 0.05, x + 0.05], clipped at 0.

    Returns:
        (delta_lower, delta_upper) in percent units, or (None, None) if either
        endpoint is unavailable.
    """
    if start_pct is None or end_pct is None:
        return None, None

    start_low = max(0.0, start_pct - 0.05)
    start_high = max(0.0, start_pct + 0.05)
    end_low = max(0.0, end_pct - 0.05)
    end_high = max(0.0, end_pct + 0.05)

    delta_low = max(0.0, end_low - start_high)
    delta_high = max(0.0, end_high - start_low)
    return delta_low, delta_high


def post_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    keep_alive: str,
    think: bool | str | None,
    timeout_seconds: float,
) -> tuple[int, dict[str, Any] | None, str | None, float]:
    """Send one synchronous non-streaming chat request."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if think is not None:
        payload["think"] = think

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/chat",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )

    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
            elapsed = time.monotonic() - started
            parsed = json.loads(raw) if raw else {}
            return response.getcode(), parsed, None, elapsed
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        elapsed = time.monotonic() - started
        parsed: dict[str, Any] | None = None
        error_message = raw
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                error_message = str(parsed.get("error") or raw)
        except json.JSONDecodeError:
            parsed = None
        return exc.code, parsed, error_message, elapsed
    except urllib.error.URLError as exc:
        elapsed = time.monotonic() - started
        return 0, None, str(exc.reason), elapsed


def make_record(
    *,
    index: int,
    label: str,
    status: int,
    payload: dict[str, Any] | None,
    error: str | None,
    wall_time_seconds: float,
) -> RequestRecord:
    """Convert one API response into a normalized record."""
    payload = payload or {}
    message = payload.get("message") or {}
    content = str(message.get("content") or "")
    total_duration_ns = int(payload.get("total_duration") or 0)
    load_duration_ns = int(payload.get("load_duration") or 0)
    prompt_eval_duration_ns = int(payload.get("prompt_eval_duration") or 0)
    eval_duration_ns = int(payload.get("eval_duration") or 0)
    effective_usage_ns = load_duration_ns + prompt_eval_duration_ns + eval_duration_ns
    effective_usage_source = "decomposed_duration"
    if effective_usage_ns <= 0 and total_duration_ns > 0:
        # Ollama Cloud may omit the duration breakdown while still returning
        # end-to-end server time. Fall back to total_duration as a usage proxy.
        effective_usage_ns = total_duration_ns
        effective_usage_source = "total_duration_fallback"
    limit_hit = is_limit_error(status, error)

    return RequestRecord(
        index=index,
        label=label,
        status=status,
        ok=status == 200,
        limit_hit=limit_hit,
        error=error,
        wall_time_seconds=wall_time_seconds,
        total_duration_ns=total_duration_ns,
        load_duration_ns=load_duration_ns,
        prompt_eval_count=int(payload.get("prompt_eval_count") or 0),
        prompt_eval_duration_ns=prompt_eval_duration_ns,
        eval_count=int(payload.get("eval_count") or 0),
        eval_duration_ns=eval_duration_ns,
        effective_usage_ns=effective_usage_ns,
        effective_usage_source=effective_usage_source,
        response_chars=len(content),
        created_at=payload.get("created_at"),
    )


def print_progress(record: RequestRecord) -> None:
    """Print a concise progress line."""
    if record.ok:
        usage_ms = record.effective_usage_ns / 1_000_000.0
        print(
            f"[{record.index:05d}] OK "
            f"prompt={record.prompt_eval_count} out={record.eval_count} "
            f"usage={usage_ms:.1f}ms wall={record.wall_time_seconds:.2f}s"
        )
        return

    status = record.status if record.status else "ERR"
    suffix = " limit-hit" if record.limit_hit else ""
    print(f"[{record.index:05d}] {status}{suffix} {record.error or ''}".strip())


def summarize(
    records: list[RequestRecord],
    limit_record: RequestRecord | None,
    *,
    session_start_pct: float | None = None,
    session_end_pct: float | None = None,
    weekly_start_pct: float | None = None,
    weekly_end_pct: float | None = None,
) -> dict[str, Any]:
    """Build a summary JSON payload."""
    successes = [record for record in records if record.ok]
    failures = [record for record in records if not record.ok]
    usage_values = [record.effective_usage_ns for record in successes]
    token_values = [record.prompt_eval_count + record.eval_count for record in successes]
    total_usage_ns = sum(usage_values)
    total_tokens = sum(token_values)
    p95_usage_ns = percentile(usage_values, 95.0)
    p99_usage_ns = percentile(usage_values, 99.0)
    p95_tokens = percentile(token_values, 95.0)

    safe_lb_from_p95_usage = (
        int(total_usage_ns // p95_usage_ns) if p95_usage_ns > 0 else len(successes)
    )
    safe_lb_from_p99_usage = (
        int(total_usage_ns // p99_usage_ns) if p99_usage_ns > 0 else len(successes)
    )
    safe_lb_from_p95_tokens = (
        int(total_tokens // p95_tokens) if p95_tokens > 0 else len(successes)
    )

    summary = {
        "success_count": len(successes),
        "failure_count": len(failures),
        "limit_hit": limit_record is not None and limit_record.limit_hit,
        "limit_status": limit_record.status if limit_record else None,
        "limit_error": limit_record.error if limit_record else None,
        "observed_request_lower_bound": len(successes),
        "total_effective_usage_ns": total_usage_ns,
        "total_token_equivalent": total_tokens,
        "avg_effective_usage_ns": (total_usage_ns / len(successes)) if successes else 0.0,
        "p95_effective_usage_ns": p95_usage_ns,
        "p99_effective_usage_ns": p99_usage_ns,
        "p95_token_equivalent": p95_tokens,
        "safe_request_lower_bound_from_p95_usage": safe_lb_from_p95_usage,
        "safe_request_lower_bound_from_p99_usage": safe_lb_from_p99_usage,
        "safe_request_lower_bound_from_p95_tokens": safe_lb_from_p95_tokens,
        "usage_proxy_sources": sorted({record.effective_usage_source for record in successes}),
    }
    summary["dashboard"] = {
        "session_start_pct": session_start_pct,
        "session_end_pct": session_end_pct,
        "weekly_start_pct": weekly_start_pct,
        "weekly_end_pct": weekly_end_pct,
    }

    def _attach_budget_estimates(
        *,
        name: str,
        start_pct: float | None,
        end_pct: float | None,
    ) -> dict[str, Any] | None:
        delta_low, delta_high = _ui_delta_bounds(start_pct, end_pct)
        if delta_low is None or delta_high is None or delta_high <= 0:
            return None

        delta_mid = max(0.0, (end_pct or 0.0) - (start_pct or 0.0))
        total_requests_mid = len(successes) * 100.0 / delta_mid if delta_mid > 0 else None
        total_requests_lb = len(successes) * 100.0 / delta_high
        total_requests_ub = (
            len(successes) * 100.0 / delta_low if delta_low > 0 else None
        )

        total_usage_mid = total_usage_ns * 100.0 / delta_mid if delta_mid > 0 else None
        total_usage_lb = total_usage_ns * 100.0 / delta_high
        total_usage_ub = total_usage_ns * 100.0 / delta_low if delta_low > 0 else None

        total_tokens_mid = total_tokens * 100.0 / delta_mid if delta_mid > 0 else None
        total_tokens_lb = total_tokens * 100.0 / delta_high
        total_tokens_ub = total_tokens * 100.0 / delta_low if delta_low > 0 else None

        safe_requests_from_p95_usage = (
            int(total_usage_lb // p95_usage_ns) if p95_usage_ns > 0 else None
        )
        safe_requests_from_p99_usage = (
            int(total_usage_lb // p99_usage_ns) if p99_usage_ns > 0 else None
        )
        safe_requests_from_p95_tokens = (
            int(total_tokens_lb // p95_tokens) if p95_tokens > 0 else None
        )

        return {
            "displayed_delta_pct": delta_mid,
            "delta_pct_lower_bound": delta_low,
            "delta_pct_upper_bound": delta_high,
            "request_equivalent_budget_mid": total_requests_mid,
            "request_equivalent_budget_lower_bound": total_requests_lb,
            "request_equivalent_budget_upper_bound": total_requests_ub,
            "effective_usage_ns_budget_mid": total_usage_mid,
            "effective_usage_ns_budget_lower_bound": total_usage_lb,
            "effective_usage_ns_budget_upper_bound": total_usage_ub,
            "token_equivalent_budget_mid": total_tokens_mid,
            "token_equivalent_budget_lower_bound": total_tokens_lb,
            "token_equivalent_budget_upper_bound": total_tokens_ub,
            "safe_request_lower_bound_from_p95_usage_budget": safe_requests_from_p95_usage,
            "safe_request_lower_bound_from_p99_usage_budget": safe_requests_from_p99_usage,
            "safe_request_lower_bound_from_p95_token_budget": safe_requests_from_p95_tokens,
            "window": name,
        }

    summary["session_budget_estimate"] = _attach_budget_estimates(
        name="5h_session",
        start_pct=session_start_pct,
        end_pct=session_end_pct,
    )
    summary["weekly_budget_estimate"] = _attach_budget_estimates(
        name="7d_weekly",
        start_pct=weekly_start_pct,
        end_pct=weekly_end_pct,
    )
    return summary


def ensure_output_dir(path: str | None) -> pathlib.Path:
    """Create the output directory if needed."""
    output_dir = pathlib.Path(path) if path else _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def reserve_work(state: SharedState) -> tuple[int, PromptExample] | None:
    """Reserve the next work item for a worker, or return None if finished."""
    with state.lock:
        if state.stop_event.is_set():
            return None

        elapsed_minutes = (time.monotonic() - state.start_time) / 60.0
        if elapsed_minutes >= state.max_duration_minutes:
            state.stop_event.set()
            return None

        if state.next_index > state.max_requests:
            state.stop_event.set()
            return None

        index = state.next_index
        state.next_index += 1
        example = state.examples[(index - 1) % len(state.examples)]
        return index, example


def record_result(
    *,
    state: SharedState,
    record: RequestRecord,
    records_file,
) -> None:
    """Store one completed record and update stop conditions."""
    with state.lock:
        if state.records is None:
            state.records = []
        state.records.append(record)
        records_file.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        records_file.flush()
        print_progress(record)

        if record.limit_hit:
            state.limit_record = record
            state.stop_event.set()
            print("Quota or usage limit appears to have been hit; stopping.")
            return

        if not record.ok:
            state.stop_event.set()
            print("Stopping on a non-retryable non-limit error.", file=sys.stderr)


def worker_loop(
    *,
    worker_id: int,
    state: SharedState,
    args: argparse.Namespace,
    api_key: str,
    think: bool | str | None,
    records_file,
) -> None:
    """One quota-probe worker."""
    while not state.stop_event.is_set():
        work_item = reserve_work(state)
        if work_item is None:
            return

        index, example = work_item
        status = 0
        payload: dict[str, Any] | None = None
        error: str | None = None
        wall_time_seconds = 0.0

        for attempt in range(args.max_retries + 1):
            status, payload, error, wall_time_seconds = post_chat(
                base_url=args.base_url,
                api_key=api_key,
                model=args.model,
                messages=example.messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                keep_alive=args.keep_alive,
                think=think,
                timeout_seconds=args.timeout_seconds,
            )
            if status not in TRANSIENT_STATUSES or attempt >= args.max_retries:
                break
            backoff = args.backoff_seconds * (2**attempt)
            with state.lock:
                print(
                    f"[worker={worker_id} req={index:05d}] transient status={status}, "
                    f"retrying in {backoff:.1f}s",
                    flush=True,
                )
            time.sleep(backoff)

        record = make_record(
            index=index,
            label=example.label,
            status=status,
            payload=payload,
            error=error,
            wall_time_seconds=wall_time_seconds,
        )
        record_result(state=state, record=record, records_file=records_file)

        if args.sleep_seconds > 0 and not state.stop_event.is_set():
            time.sleep(args.sleep_seconds)


def run_probe(args: argparse.Namespace) -> int:
    """Run the quota probing loop."""
    api_key = os.environ.get("OLLAMA_API_KEY") or os.environ.get("LLAMA_API_KEY")
    if not api_key:
        print("Neither OLLAMA_API_KEY nor LLAMA_API_KEY is set.", file=sys.stderr)
        return 2

    examples = load_examples(args.prompt, args.prompt_file, args.system_prompt)
    think = _coerce_think(args.think)
    output_dir = ensure_output_dir(args.output_dir)
    records_path = output_dir / "requests.jsonl"
    summary_path = output_dir / "summary.json"
    config_path = output_dir / "config.json"

    with config_path.open("w") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)

    start_time = time.monotonic()
    state = SharedState(
        start_time=start_time,
        max_requests=args.max_requests,
        max_duration_minutes=args.max_duration_minutes,
        examples=examples,
        lock=threading.Lock(),
        stop_event=threading.Event(),
        records=[],
    )

    with records_path.open("w") as records_file:
        workers = [
            threading.Thread(
                target=worker_loop,
                kwargs={
                    "worker_id": worker_id + 1,
                    "state": state,
                    "args": args,
                    "api_key": api_key,
                    "think": think,
                    "records_file": records_file,
                },
                daemon=True,
            )
            for worker_id in range(max(1, args.concurrency))
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

        if not state.limit_record and (time.monotonic() - start_time) / 60.0 >= args.max_duration_minutes:
            print(
                f"Reached max duration {args.max_duration_minutes:.1f} minutes without a hard limit."
            )

    summary = summarize(
        state.records or [],
        state.limit_record,
        session_start_pct=args.session_start_pct,
        session_end_pct=args.session_end_pct,
        weekly_start_pct=args.weekly_start_pct,
        weekly_end_pct=args.weekly_end_pct,
    )
    summary["model"] = args.model
    summary["base_url"] = args.base_url
    summary["output_dir"] = str(output_dir)
    summary["workload_examples"] = len(examples)
    summary["concurrency"] = args.concurrency
    summary["elapsed_minutes"] = (time.monotonic() - start_time) / 60.0

    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print()
    print("=== Ollama Quota Probe Summary ===")
    print(f"Model: {summary['model']}")
    print(f"Successful requests: {summary['success_count']}")
    print(f"Observed request lower bound: {summary['observed_request_lower_bound']}")
    print(
        "Safe lower bound from p95 effective usage: "
        f"{summary['safe_request_lower_bound_from_p95_usage']}"
    )
    print(
        "Safe lower bound from p99 effective usage: "
        f"{summary['safe_request_lower_bound_from_p99_usage']}"
    )
    print(
        "Safe lower bound from p95 token equivalent: "
        f"{summary['safe_request_lower_bound_from_p95_tokens']}"
    )
    if summary.get("session_budget_estimate"):
        session_est = summary["session_budget_estimate"]
        print(
            "Implied 5h request-equivalent budget: "
            f"{session_est['request_equivalent_budget_mid']:.0f} "
            f"(LB {session_est['request_equivalent_budget_lower_bound']:.0f})"
        )
        print(
            "Safe 5h request lower bound from p95 usage budget: "
            f"{session_est['safe_request_lower_bound_from_p95_usage_budget']}"
        )
    if summary.get("weekly_budget_estimate"):
        week_est = summary["weekly_budget_estimate"]
        print(
            "Implied weekly request-equivalent budget: "
            f"{week_est['request_equivalent_budget_mid']:.0f} "
            f"(LB {week_est['request_equivalent_budget_lower_bound']:.0f})"
        )
        print(
            "Safe weekly request lower bound from p95 usage budget: "
            f"{week_est['safe_request_lower_bound_from_p95_usage_budget']}"
        )
    if summary["limit_hit"]:
        print(f"Limit status: {summary['limit_status']}")
        print(f"Limit error: {summary['limit_error']}")
    else:
        print("Limit hit: no (run was censored before a hard limit was observed)")
    print(f"Artifacts: {output_dir}")

    return 0


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    raise SystemExit(run_probe(args))


if __name__ == "__main__":
    main()
