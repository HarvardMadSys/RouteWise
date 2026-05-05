"""Compose BurstGPT arrivals with reused ShareGPT conversation text.

The output is a simulator-ready JSONL workload. BurstGPT is authoritative for
arrival time, token counts, model label, elapsed time, and session structure.
Each BurstGPT session is deterministically bound to one ShareGPT conversation,
and requests within that session advance through that conversation's turns.

Usage:
    python3 scripts/prepare_workload.py --days 30
    python3 scripts/prepare_workload.py --days 30 --dry-run
    python3 scripts/prepare_workload.py --start-day 10 --days 30

Raw BurstGPT/ShareGPT traces are downloaded into data/.cache/ when missing.

Output schema (one JSON object per line):
    {
        "arrived_at": float,
        "session_id": str,
        "num_prefill_tokens": int,
        "num_decode_tokens": int,
        "model": str,
        "log_type": str,
        "elapsed_time_sec": float,
        "prompt_text": str,
        "response_text": str,
        "sharegpt_conversation_id": str,
        "sharegpt_turn_index": int
    }
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_BURSTGPT = _REPO_ROOT / "data" / ".cache" / "BurstGPT_3.csv"
_DEFAULT_SHAREGPT = (
    _REPO_ROOT / "data" / ".cache" / "ShareGPT_V3_unfiltered_cleaned_split.json"
)
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "data"
_SECONDS_PER_DAY = 86_400
_BURSTGPT_URL = "https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_3.csv"
_SHAREGPT_URL = "https://huggingface.co/datasets/learnanything/sharegpt_v3_unfiltered_cleaned_split/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
_BURSTGPT_SHA256 = "2299986a07388aa303ec2c41d1131e756db650a39ed6ef9dfe7cc3d7f9a43b8f"
_SHAREGPT_SHA256 = "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024


def _text_value(message: dict[str, Any]) -> str:
    value = message.get("value", "")
    return value if isinstance(value, str) else str(value)


def _append_transcript_part(parts: list[str], role: str, text: str) -> None:
    if text:
        parts.append(f"{role}: {text}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_DOWNLOAD_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_sha256: str, *, verify: bool = True) -> None:
    if not verify:
        return

    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Checksum mismatch for {path}\n"
            f"  expected: {expected_sha256}\n"
            f"  actual:   {actual_sha256}"
        )


def download_file(
    name: str,
    url: str,
    path: Path,
    expected_sha256: str,
    *,
    force: bool = False,
    verify: bool = True,
) -> None:
    """Download one raw trace unless a valid local copy already exists."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size > 0 and not force:
        verify_file(path, expected_sha256, verify=verify)
        logger.info("[skip] %s already exists: %s", name, path)
        return

    tmp_path = path.with_name(f"{path.name}.part")
    if tmp_path.exists():
        tmp_path.unlink()

    logger.info("[download] %s", name)
    logger.info("           %s", url)
    logger.info("        -> %s", path)

    request = Request(url, headers={"User-Agent": "RouteWise-workload-prep/1.0"})
    with urlopen(request) as response, tmp_path.open("wb") as handle:
        while chunk := response.read(_DOWNLOAD_CHUNK_SIZE):
            handle.write(chunk)

    verify_file(tmp_path, expected_sha256, verify=verify)
    tmp_path.replace(path)
    logger.info("[ok] %s saved: %s", name, path)


def ensure_raw_workloads(
    *,
    burstgpt_path: Path,
    sharegpt_path: Path,
) -> None:
    """Ensure both raw traces exist under data/.cache/ before composition."""
    download_file(
        "BurstGPT_3",
        os.environ.get("BURSTGPT_URL", _BURSTGPT_URL),
        burstgpt_path,
        _BURSTGPT_SHA256,
    )
    download_file(
        "ShareGPT_V3",
        os.environ.get("SHAREGPT_URL", _SHAREGPT_URL),
        sharegpt_path,
        _SHAREGPT_SHA256,
    )


def load_sharegpt_conversations(path: Path) -> list[dict[str, object]]:
    """Return reusable ShareGPT conversations with cumulative text turns."""
    if not path.exists():
        raise FileNotFoundError(
            f"ShareGPT trace not found: {path}\n"
            "Run: python3 scripts/prepare_workload.py --days 30"
        )

    logger.info("Loading ShareGPT text pool from %s ...", path)
    with path.open() as handle:
        conversations = json.load(handle)

    reusable_conversations: list[dict[str, object]] = []
    human_roles = {"human", "user"}
    assistant_roles = {"gpt", "chatgpt"}

    for conv in conversations:
        conv_id = str(conv.get("id", ""))
        messages = conv.get("conversations", [])
        if not isinstance(messages, list):
            continue

        human_turn_index = 0
        transcript_parts: list[str] = []
        turns: list[dict[str, object]] = []
        for idx, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            role = message.get("from")
            if role in assistant_roles:
                _append_transcript_part(transcript_parts, "Assistant", _text_value(message))
                continue
            if role not in human_roles:
                continue

            prompt_text = _text_value(message)
            response_text = ""
            if idx + 1 < len(messages):
                next_message = messages[idx + 1]
                if (
                    isinstance(next_message, dict)
                    and next_message.get("from") in assistant_roles
                ):
                    response_text = _text_value(next_message)

            if prompt_text:
                _append_transcript_part(transcript_parts, "Human", prompt_text)
                turns.append(
                    {
                        "turn_index": human_turn_index,
                        "prompt_text": "\n\n".join(transcript_parts),
                        "response_text": response_text,
                    }
                )
                human_turn_index += 1

        if turns:
            reusable_conversations.append(
                {
                    "conversation_id": conv_id,
                    "turns": turns,
                }
            )

    if not reusable_conversations:
        raise ValueError(f"No ShareGPT human turns found in {path}")

    turn_count = sum(len(conv["turns"]) for conv in reusable_conversations)
    logger.info(
        "Loaded %d reusable ShareGPT conversations (%d human turns)",
        len(reusable_conversations),
        turn_count,
    )
    return reusable_conversations


def compose_workload(
    burstgpt_path: Path,
    sharegpt_path: Path,
    output_path: Path,
    *,
    days: int = 30,
    start_day: int = 0,
    dry_run: bool = False,
) -> dict[str, object]:
    """Compose a rebased BurstGPT slice with reused ShareGPT text."""
    if days <= 0:
        raise ValueError(f"days must be positive, got {days}")
    if start_day < 0:
        raise ValueError(f"start_day must be non-negative, got {start_day}")
    if not burstgpt_path.exists():
        raise FileNotFoundError(
            f"BurstGPT trace not found: {burstgpt_path}\n"
            "Run: python3 scripts/prepare_workload.py --days 30"
        )

    sharegpt_conversations = load_sharegpt_conversations(sharegpt_path)
    sharegpt_turn_count = sum(len(conv["turns"]) for conv in sharegpt_conversations)

    logger.info("Reading BurstGPT trace from %s ...", burstgpt_path)
    with burstgpt_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        first = next(reader)
        t0 = float(first["Timestamp"])

    window_start = t0 + start_day * _SECONDS_PER_DAY
    window_end = window_start + days * _SECONDS_PER_DAY
    logger.info(
        "Extraction window: day %d -> day %d (%.1f -> %.1f raw seconds)",
        start_day,
        start_day + days,
        window_start,
        window_end,
    )

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output_path.open("w")
    else:
        output_handle = None

    total_requests = 0
    skipped_rows = 0
    api_log_rows = 0
    sessions_seen: set[str] = set()
    session_to_conversation: dict[str, int] = {}
    session_turn_index: dict[str, int] = {}
    token_totals: list[int] = []
    last_arrived_at = 0.0

    try:
        with burstgpt_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row_idx, row in enumerate(reader):
                ts = float(row["Timestamp"])
                if ts < window_start:
                    continue
                if ts >= window_end:
                    break

                try:
                    request_tokens = int(row["Request tokens"])
                    response_tokens = int(row["Response tokens"])
                    total_tokens = int(row["Total tokens"])
                except (KeyError, ValueError):
                    skipped_rows += 1
                    continue

                if total_tokens == 0:
                    skipped_rows += 1
                    continue

                session_id = row.get("Session ID", "").strip()
                if not session_id:
                    session_id = f"api:{row_idx}"
                    api_log_rows += 1
                sessions_seen.add(session_id)

                if session_id not in session_to_conversation:
                    session_to_conversation[session_id] = (
                        len(session_to_conversation) % len(sharegpt_conversations)
                    )
                    session_turn_index[session_id] = 0

                conversation = sharegpt_conversations[
                    session_to_conversation[session_id]
                ]
                turns = conversation["turns"]
                if not isinstance(turns, list):
                    raise TypeError("ShareGPT conversation turns must be a list")
                turn_index = min(session_turn_index[session_id], len(turns) - 1)
                turn = turns[turn_index]
                session_turn_index[session_id] += 1

                arrived_at = ts - window_start
                record = {
                    "arrived_at": arrived_at,
                    "session_id": session_id,
                    "num_prefill_tokens": request_tokens,
                    "num_decode_tokens": response_tokens,
                    "model": row.get("Model", ""),
                    "log_type": row.get("Log Type", ""),
                    "elapsed_time_sec": float(row.get("Elapsed time") or 0.0),
                    "prompt_text": turn["prompt_text"],
                    "response_text": turn["response_text"],
                    "sharegpt_conversation_id": conversation["conversation_id"],
                    "sharegpt_turn_index": turn["turn_index"],
                }

                if output_handle is not None:
                    output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

                total_requests += 1
                token_totals.append(total_tokens)
                last_arrived_at = arrived_at
                if total_requests % 500_000 == 0:
                    logger.info("  ... %d requests composed so far", total_requests)
    finally:
        if output_handle is not None:
            output_handle.close()

    stats: dict[str, object] = {
        "burstgpt_file": str(burstgpt_path),
        "sharegpt_file": str(sharegpt_path),
        "output_file": str(output_path),
        "days_requested": days,
        "start_day": start_day,
        "total_requests": total_requests,
        "unique_sessions": len(sessions_seen),
        "api_log_rows": api_log_rows,
        "skipped_rows": skipped_rows,
        "sharegpt_conversation_pool": len(sharegpt_conversations),
        "sharegpt_turn_pool": sharegpt_turn_count,
        "sharegpt_session_reuse_factor": len(sessions_seen) / len(sharegpt_conversations),
        "sharegpt_turn_reuse_factor": total_requests / sharegpt_turn_count,
        "duration_seconds": last_arrived_at,
        "requests_per_day": total_requests / days,
    }

    if token_totals:
        sorted_tokens = sorted(token_totals)
        stats["median_total_tokens"] = int(sorted_tokens[len(sorted_tokens) // 2])
        stats["mean_total_tokens"] = sum(token_totals) / len(token_totals)

    if output_path.exists() and not dry_run:
        stats["output_size_mb"] = round(output_path.stat().st_size / (1024 * 1024), 1)

    logger.info(
        "Composition complete: %d requests, %d sessions, %.2fx ShareGPT session reuse",
        total_requests,
        len(sessions_seen),
        len(sessions_seen) / len(sharegpt_conversations),
    )
    if dry_run:
        logger.info("Dry run -- no file written.")
    else:
        logger.info("Written %s", output_path)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compose BurstGPT arrivals/tokens with reused ShareGPT text.",
    )
    parser.add_argument(
        "--burstgpt",
        type=Path,
        default=_DEFAULT_BURSTGPT,
        help="Path to BurstGPT_3.csv (default: data/.cache/BurstGPT_3.csv)",
    )
    parser.add_argument(
        "--sharegpt",
        type=Path,
        default=_DEFAULT_SHAREGPT,
        help="Path to ShareGPT V3 JSON (default: data/.cache/ShareGPT_V3_...).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: data/burstgpt_{days}d.jsonl)",
    )
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--start-day", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if args.output is None:
        suffix = f"burstgpt_{args.days}d"
        if args.start_day > 0:
            suffix += f"_from{args.start_day}"
        args.output = _DEFAULT_OUTPUT_DIR / f"{suffix}.jsonl"

    ensure_raw_workloads(
        burstgpt_path=args.burstgpt,
        sharegpt_path=args.sharegpt,
    )

    stats = compose_workload(
        burstgpt_path=args.burstgpt,
        sharegpt_path=args.sharegpt,
        output_path=args.output,
        days=args.days,
        start_day=args.start_day,
        dry_run=args.dry_run,
    )

    print("\n--- Workload Composition Summary ---")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
