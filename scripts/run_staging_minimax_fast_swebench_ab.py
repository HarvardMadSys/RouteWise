#!/usr/bin/env python3
"""Run a staged SWE-bench A/B between minimax-fast and minimax-fast2.

Default mode is a dry run that prints the planned order only. Add ``--execute``
to run the full experiment:

    uv run --python 3.13 --only-group agent-benchmark python \
        scripts/run_staging_minimax_fast_swebench_ab.py --execute

The default plan is 10 SWE-bench Verified tasks, 10 attempts per task, and two
models per attempt:

    routewise -> openai/minimax-fast
    baseline  -> openai/minimax-fast2

Runs are task-blocked, and every attempt uses the fixed order
``routewise -> baseline`` so each pair is measured in a tight window.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from math import sqrt
from pathlib import Path
from statistics import mean, median, stdev
from typing import TYPE_CHECKING, Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.agentic_benchmark.runtime.config import GatewayConfig
from experiments.agentic_benchmark.runtime.gateway import GatewayClient
from experiments.agentic_benchmark.swebench.runner import TaskRun, run_task
from experiments.agentic_benchmark.swebench.tasks import TaskSpec, sample_tasks

if TYPE_CHECKING:
    from experiments.agentic_benchmark.runtime.schemas import LLMRequestRecord

DEFAULT_BASE_URL = "https://staging.freeinference.org"
DEFAULT_POLICIES = (
    ("routewise", "openai/minimax-fast"),
    ("baseline", "openai/minimax-fast2"),
)


@dataclass(frozen=True)
class PlannedRun:
    index: int
    task: TaskSpec
    attempt: int
    policy: str
    model_name: str

    @property
    def session_id(self) -> str:
        return f"swe-{self.policy}-{self.task.instance_id}-{self.attempt}"


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE lines without executing the file as shell."""
    if not path.exists():
        return
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or run the staging minimax-fast vs minimax-fast2 SWE-bench A/B."
    )
    parser.add_argument("--execute", action="store_true", help="Actually run SWE-bench.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--subset", default="verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-tasks", type=int, default=10)
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--instance-id",
        action="append",
        dest="instance_ids",
        help="Explicit SWE-bench instance id. Repeat to pass multiple tasks.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume an existing output directory, skipping completed successful runs.",
    )
    parser.add_argument("--cost-limit", type=float, default=0.0)
    parser.add_argument("--pull-timeout", type=int, default=1800)
    parser.add_argument("--env-cmd-timeout", type=int, default=300)
    parser.add_argument("--task-timeout", type=int, default=3600)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=3,
        help="Abort the batch after this many consecutive failed runs (0 disables).",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the admin export preflight check before running.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip pre-pulling task Docker images before the measured runs.",
    )
    return parser.parse_args()


def choose_tasks(args: argparse.Namespace) -> list[TaskSpec]:
    if args.instance_ids:
        return [
            TaskSpec(instance_id=instance_id, subset=args.subset, split=args.split)
            for instance_id in args.instance_ids
        ]
    return sample_tasks(
        subset=args.subset,
        split=args.split,
        n=args.n_tasks,
        seed=args.seed,
    )


def build_plan(tasks: list[TaskSpec], attempts: int) -> list[PlannedRun]:
    """Build task-blocked RouteWise/baseline pairs."""
    plan: list[PlannedRun] = []
    for task in tasks:
        for attempt in range(1, attempts + 1):
            for policy, model_name in DEFAULT_POLICIES:
                plan.append(
                    PlannedRun(
                        index=len(plan) + 1,
                        task=task,
                        attempt=attempt,
                        policy=policy,
                        model_name=model_name,
                    )
                )
    return plan


def load_plan(path: Path) -> list[PlannedRun]:
    data = json.loads(path.read_text(errors="ignore"))
    runs = data.get("runs")
    if not isinstance(runs, list):
        raise RuntimeError(f"Invalid run plan: {path}")
    plan: list[PlannedRun] = []
    for item in runs:
        task_data = item["task"]
        plan.append(
            PlannedRun(
                index=int(item["index"]),
                task=TaskSpec(
                    instance_id=task_data["instance_id"],
                    subset=task_data["subset"],
                    split=task_data["split"],
                ),
                attempt=int(item["attempt"]),
                policy=item["policy"],
                model_name=item["model_name"],
            )
        )
    return plan


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("outputs/agentic_benchmark") / f"staging_minimax_fast_ab_10x10_{stamp}"


def print_plan(plan: list[PlannedRun], output_dir: Path) -> None:
    print(f"output_dir: {output_dir}")
    print(f"planned_runs: {len(plan)}")
    counts = Counter((run.policy, run.model_name) for run in plan)
    for (policy, model_name), count in sorted(counts.items()):
        print(f"  {policy:9s} {model_name:24s} {count}")
    print("\nfirst 24 runs:")
    for run in plan[:24]:
        print(
            f"  {run.index:03d} attempt={run.attempt:02d} "
            f"task={run.task.instance_id} policy={run.policy} model={run.model_name}"
        )
    if len(plan) > 24:
        print(f"  ... {len(plan) - 24} more")


def record_to_public_dict(record: LLMRequestRecord) -> dict[str, Any]:
    """Drop prompt/completion content while preserving routing metrics."""
    routewise = record.routewise or {}
    return {
        "request_id": record.request_id,
        "model_id": record.model_id,
        "provider": record.provider,
        "status_code": record.status_code,
        "latency_ms": record.latency_ms,
        "ttft_ms": record.ttft_ms,
        "prompt_tokens": record.prompt_tokens,
        "completion_tokens": record.completion_tokens,
        "total_tokens": record.total_tokens,
        "cost_usd": record.cost_usd,
        "error": record.error,
        "session_id": record.session_id,
        "routewise": {
            "policy": routewise.get("policy"),
            "lp_status": routewise.get("lp_status"),
            "selected_endpoint": routewise.get("selected_endpoint"),
            "initial_selected_endpoint": routewise.get("initial_selected_endpoint"),
            "final_endpoint": routewise.get("final_endpoint"),
            "selected_provider_type": routewise.get("selected_provider_type"),
            "final_provider_type": routewise.get("final_provider_type"),
            "fallback_attempts": routewise.get("fallback_attempts"),
            "hedged": routewise.get("hedged"),
            "hedge_triggered": routewise.get("hedge_triggered"),
            "hedge_winner": routewise.get("hedge_winner"),
            "backup_provider": routewise.get("backup_provider"),
            "backup_won": routewise.get("backup_won"),
            "quota_committed": routewise.get("quota_committed"),
            "quota_pool": routewise.get("quota_pool"),
            "quota_remaining": routewise.get("quota_remaining"),
            "quota_used_fraction": routewise.get("quota_used_fraction"),
            "sc_committed": routewise.get("sc_committed"),
            "concurrency_pool": routewise.get("concurrency_pool"),
        }
        if routewise
        else None,
    }


def summarize_records(records: list[LLMRequestRecord]) -> dict[str, Any]:
    routewise = [record.routewise or {} for record in records if record.routewise]
    return {
        "providers": dict(Counter(record.provider for record in records)),
        "status_codes": dict(Counter(str(record.status_code) for record in records)),
        "routewise": {
            "policies": dict(Counter(str(rw.get("policy")) for rw in routewise)),
            "lp_status": dict(Counter(str(rw.get("lp_status")) for rw in routewise)),
            "selected_endpoint": dict(
                Counter(str(rw.get("selected_endpoint")) for rw in routewise)
            ),
            "final_endpoint": dict(Counter(str(rw.get("final_endpoint")) for rw in routewise)),
            "final_provider_type": dict(
                Counter(str(rw.get("final_provider_type")) for rw in routewise)
            ),
            "fallback_attempts": dict(
                Counter(str(rw.get("fallback_attempts")) for rw in routewise)
            ),
            "hedged": dict(Counter(str(rw.get("hedged")) for rw in routewise)),
            "hedge_winner": dict(Counter(str(rw.get("hedge_winner")) for rw in routewise)),
            "quota_committed": dict(Counter(str(rw.get("quota_committed")) for rw in routewise)),
        },
    }


def trajectory_exit_status(path: str) -> str | None:
    try:
        data = json.loads(Path(path).read_text(errors="ignore"))
    except Exception:
        return None
    info = data.get("info") if isinstance(data, dict) else None
    if not isinstance(info, dict):
        return None
    status = info.get("exit_status")
    return str(status) if status is not None else None


def run_one(
    planned: PlannedRun,
    *,
    config: GatewayConfig,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = datetime.now(timezone.utc)
    result: TaskRun | None = None
    error: str | None = None
    try:
        result = run_task(
            planned.task,
            policy=planned.policy,
            model_name=planned.model_name,
            config=config,
            output_dir=output_dir,
            attempt=planned.attempt,
            textbased=True,
            cost_limit=args.cost_limit,
            pull_timeout=args.pull_timeout,
            env_cmd_timeout=args.env_cmd_timeout,
            timeout_sec=args.task_timeout,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finished = datetime.now(timezone.utc)

    if result is None:
        return (
            {
                "index": planned.index,
                "task": asdict(planned.task),
                "attempt": planned.attempt,
                "policy": planned.policy,
                "model_name": planned.model_name,
                "session_id": planned.session_id,
                "started_at": started.isoformat(),
                "finished_at": finished.isoformat(),
                "returncode": None,
                "wall_sec": (finished - started).total_seconds(),
                "n_requests": 0,
                "total_cost_usd": 0.0,
                "mean_ttft_ms": None,
                "total_llm_latency_ms": None,
                "mean_llm_latency_ms": None,
                "trajectory_path": None,
                "exit_status": None,
                "request_summary": {},
                "records_missing": True,
                "export_error": None,
                "error": error,
            },
            [],
        )

    request_rows = [
        {
            "run_index": planned.index,
            "task_id": planned.task.instance_id,
            "attempt": planned.attempt,
            "policy": planned.policy,
            "model_name": planned.model_name,
            **record_to_public_dict(record),
        }
        for record in result.records
    ]

    return (
        {
            "index": planned.index,
            "task": asdict(planned.task),
            "attempt": planned.attempt,
            "policy": planned.policy,
            "model_name": planned.model_name,
            "session_id": result.session_id,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "returncode": result.returncode,
            "wall_sec": result.wall_sec,
            "n_requests": result.n_requests,
            "total_cost_usd": result.total_cost_usd,
            "mean_ttft_ms": result.mean_ttft_ms,
            "total_llm_latency_ms": result.total_llm_latency_ms,
            "mean_llm_latency_ms": result.mean_llm_latency_ms,
            "trajectory_path": result.trajectory_path,
            "exit_status": trajectory_exit_status(result.trajectory_path),
            "request_summary": summarize_records(result.records),
            # an empty export on a successful run means the records were not
            # persisted yet (or the export missed them), not that none exist
            "records_missing": bool(result.export_error) or not result.records,
            "export_error": result.export_error,
            "error": error,
        },
        request_rows,
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def rewrite_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def run_key_from_parts(task_id: str, attempt: int, policy: str) -> tuple[str, int, str]:
    return (task_id, int(attempt), policy)


def run_key(row: dict[str, Any]) -> tuple[str, int, str]:
    task = row.get("task") or {}
    return run_key_from_parts(str(task.get("instance_id")), int(row["attempt"]), row["policy"])


def planned_key(run: PlannedRun) -> tuple[str, int, str]:
    return run_key_from_parts(run.task.instance_id, run.attempt, run.policy)


def is_successful_run(row: dict[str, Any]) -> bool:
    return row.get("error") is None and row.get("returncode") == 0


def has_request_records(row: dict[str, Any]) -> bool:
    return is_successful_run(row) and not row.get("records_missing")


def load_previous_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(errors="ignore").splitlines():
        if not raw.strip():
            continue
        rows.append(json.loads(raw))
    return rows


def latest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        by_key[run_key(row)] = row
    return list(by_key.values())


def preflight_admin_export(config: GatewayConfig) -> None:
    GatewayClient(config).export_requests(
        start_time=datetime.now(timezone.utc) - timedelta(minutes=1),
        include_content=False,
    )


def preflight_policy_chat(config: GatewayConfig, plan: list[PlannedRun]) -> None:
    """Send one tiny chat per arm and assert its exported routing metadata.

    Catches a dead API key or a mis-wired model alias before any measured run:
    the routewise arm must carry routewise metadata, the baseline must not.
    """
    stamp = int(datetime.now(timezone.utc).timestamp())
    for policy, model_name in sorted({(run.policy, run.model_name) for run in plan}):
        model_id = model_name.split("/")[-1]
        client = GatewayClient(replace(config, model=model_id))
        session_id = f"preflight-{policy}-{stamp}"
        window_start = datetime.now(timezone.utc) - timedelta(seconds=10)
        result = client.chat_completion(
            [{"role": "user", "content": "Say ok"}], session_id=session_id, max_tokens=8
        )
        if result.status_code != 200:
            raise RuntimeError(
                f"preflight chat for {policy} ({model_name}) returned "
                f"{result.status_code}: {str(result.raw)[:200]}"
            )
        records = []
        # the gateway logs rows async after responding; give the insert a moment
        for _ in range(6):
            records = client.export_by_session(
                session_id, start_time=window_start, model_id=model_id
            )
            if records:
                break
            time.sleep(2)
        if not records:
            raise RuntimeError(
                f"preflight: chat succeeded but no exported record for {session_id}; "
                "fix the export pipeline before burning runs"
            )
        routewise_meta = records[0].routewise or {}
        if policy == "routewise":
            if (
                routewise_meta.get("policy") != "routewise"
                or routewise_meta.get("lp_status") is None
                or not (
                    routewise_meta.get("final_endpoint") or routewise_meta.get("selected_endpoint")
                )
            ):
                raise RuntimeError(
                    f"preflight: {model_name} did not report routewise routing "
                    f"metadata (got {routewise_meta or None}); staging alias mis-wired?"
                )
        elif routewise_meta.get("policy") == "routewise":
            raise RuntimeError(
                f"preflight: baseline {model_name} is routed by routewise; "
                "the A/B would compare routewise against itself"
            )
        print(f"preflight: {policy} ({model_name}) ok", flush=True)


def warmup_task_images(plan: list[PlannedRun], *, pull_timeout: int) -> None:
    """Pull every pending task's Docker image before any measured run.

    The first run of a task otherwise pays the multi-GB image pull inside its
    timed window, and build_plan schedules the routewise arm first for every
    attempt, so the cold-pull cost would land on one arm only.
    """
    from datasets import load_dataset
    from minisweagent.run.benchmarks.swebench import (
        DATASET_MAPPING,
        get_swebench_docker_image_name,
    )

    tasks = {run.task.instance_id: run.task for run in plan}
    if not tasks:
        return
    instances: dict[str, dict[str, Any]] = {}
    for subset, split in sorted({(task.subset, task.split) for task in tasks.values()}):
        dataset = load_dataset(DATASET_MAPPING.get(subset, subset), split=split)
        wanted = {
            instance_id
            for instance_id, task in tasks.items()
            if (task.subset, task.split) == (subset, split)
        }
        for instance in dataset:
            if instance["instance_id"] in wanted:
                instances[instance["instance_id"]] = instance
    for instance_id in sorted(tasks):
        instance = instances.get(instance_id)
        if instance is None:
            print(
                f"warmup: no dataset row for {instance_id}, leaving the pull to its run",
                flush=True,
            )
            continue
        image = get_swebench_docker_image_name(instance)
        cached = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if cached.returncode == 0:
            print(f"warmup: cached {image}", flush=True)
            continue
        print(f"warmup: pulling {image}", flush=True)
        last_error: str | None = None
        for pull_attempt in (1, 2):
            try:
                # SWE-bench images are amd64-only; be explicit so arm64 hosts pull them too.
                pulled = subprocess.run(
                    ["docker", "pull", "--platform", "linux/amd64", image],
                    capture_output=True,
                    text=True,
                    timeout=pull_timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                last_error = f"timed out after {pull_timeout}s"
            else:
                if pulled.returncode == 0:
                    last_error = None
                    break
                detail = (pulled.stderr or pulled.stdout or "").strip().splitlines()
                last_error = detail[-1] if detail else "unknown error"
            if pull_attempt == 1:
                print(f"warmup: retrying {image} ({last_error})", flush=True)
                time.sleep(5)
        if last_error is not None:
            raise RuntimeError(f"warmup pull failed for {image}: {last_error}")


def backfill_missing_records(
    rows: list[dict[str, Any]],
    *,
    config: GatewayConfig,
    request_rows_path: Path,
) -> None:
    """Retry request-log export for successful task runs that missed records."""
    client = GatewayClient(config)
    for row in rows:
        if not is_successful_run(row) or not row.get("records_missing"):
            continue
        try:
            started_at = datetime.fromisoformat(row["started_at"])
            records = client.export_by_session(
                row["session_id"],
                start_time=started_at - timedelta(seconds=10),
                model_id=row["model_name"].split("/")[-1],
            )
        except Exception as exc:
            row["export_error"] = f"{type(exc).__name__}: {exc}"
            continue
        if not records:
            continue
        row["n_requests"] = len(records)
        row["total_cost_usd"] = sum(record.cost_usd or 0.0 for record in records)
        ttfts = [record.ttft_ms for record in records if record.ttft_ms is not None]
        row["mean_ttft_ms"] = sum(ttfts) / len(ttfts) if ttfts else None
        latencies = [record.latency_ms for record in records if record.latency_ms is not None]
        row["total_llm_latency_ms"] = sum(latencies) if latencies else None
        row["mean_llm_latency_ms"] = sum(latencies) / len(latencies) if latencies else None
        row["request_summary"] = summarize_records(records)
        row["records_missing"] = False
        row["export_error"] = None
        task = row["task"]
        for record in records:
            append_jsonl(
                request_rows_path,
                {
                    "run_index": row["index"],
                    "task_id": task["instance_id"],
                    "attempt": row["attempt"],
                    "policy": row["policy"],
                    "model_name": row["model_name"],
                    **record_to_public_dict(record),
                },
            )


def rebuild_missing_llm_fields(rows: list[dict[str, Any]], *, request_rows_path: Path) -> None:
    """Fill LLM-latency fields on checkpoint rows written before those fields existed.

    Rebuilds from request_rows.jsonl so a resumed batch reports every arm over
    the same run subset instead of only the reruns.
    """
    targets = [
        row
        for row in rows
        if has_request_records(row)
        and ("total_llm_latency_ms" not in row or "mean_llm_latency_ms" not in row)
    ]
    if not targets or not request_rows_path.exists():
        return
    latencies_by_run: dict[tuple[str, int, str], dict[str, float]] = defaultdict(dict)
    for line_no, raw in enumerate(request_rows_path.read_text(errors="ignore").splitlines()):
        if not raw.strip():
            continue
        request_row = json.loads(raw)
        latency = request_row.get("latency_ms")
        if latency is None:
            continue
        key = run_key_from_parts(
            str(request_row.get("task_id")),
            int(request_row.get("attempt") or 0),
            str(request_row.get("policy")),
        )
        # backfill may have appended duplicates; keep one value per request id
        request_id = request_row.get("request_id") or f"line-{line_no}"
        latencies_by_run[key][str(request_id)] = float(latency)
    for row in targets:
        values = list(latencies_by_run.get(run_key(row), {}).values())
        row["total_llm_latency_ms"] = sum(values) if values else None
        row["mean_llm_latency_ms"] = sum(values) / len(values) if values else None


# Two-tailed 95% Student-t critical values by degrees of freedom (>30 ~ normal).
_T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def task_level_ci(paired_deltas: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    """95% t-interval over per-task mean deltas.

    Attempts of the same task are strongly correlated, so each task contributes
    one observation (its mean delta across attempts) rather than every attempt.
    """
    by_task: dict[str, list[float]] = defaultdict(list)
    for pair in paired_deltas:
        if pair.get(key) is not None:
            by_task[pair["task_id"]].append(float(pair[key]))
    if not by_task:
        return None
    per_task_mean = {task: mean(vals) for task, vals in sorted(by_task.items())}
    values = list(per_task_mean.values())
    result: dict[str, Any] = {
        "n_tasks": len(values),
        "mean": mean(values),
        "per_task_mean": per_task_mean,
        "ci95": None,
    }
    if len(values) >= 2:
        half_width = _T_CRIT_95.get(len(values) - 1, 1.96) * stdev(values) / sqrt(len(values))
        result["ci95"] = [result["mean"] - half_width, result["mean"] + half_width]
    return result


def aggregate_summary(run_rows: list[dict[str, Any]], plan: list[PlannedRun]) -> dict[str, Any]:
    run_rows = latest_rows(run_rows)
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        by_policy[row["policy"]].append(row)

    def numeric(rows: list[dict[str, Any]], key: str) -> list[float]:
        return [
            float(row[key])
            for row in rows
            if isinstance(row.get(key), int | float) and row.get(key) is not None
        ]

    policy_summary: dict[str, Any] = {}
    for policy, rows in sorted(by_policy.items()):
        success_rows = [row for row in rows if is_successful_run(row)]
        record_rows = [row for row in success_rows if has_request_records(row)]
        policy_summary[policy] = {
            "runs": len(rows),
            "successful_runs": len(success_rows),
            "recorded_runs": len(record_rows),
            "submitted": sum(1 for row in success_rows if row.get("exit_status") == "Submitted"),
            "observed_total_cost_usd": sum(numeric(record_rows, "total_cost_usd")),
            "mean_wall_sec": mean(numeric(success_rows, "wall_sec"))
            if numeric(success_rows, "wall_sec")
            else None,
            "median_wall_sec": median(numeric(success_rows, "wall_sec"))
            if numeric(success_rows, "wall_sec")
            else None,
            "mean_requests": mean(numeric(record_rows, "n_requests"))
            if numeric(record_rows, "n_requests")
            else None,
            "mean_ttft_ms": mean(numeric(record_rows, "mean_ttft_ms"))
            if numeric(record_rows, "mean_ttft_ms")
            else None,
            "mean_total_llm_latency_ms": mean(numeric(record_rows, "total_llm_latency_ms"))
            if numeric(record_rows, "total_llm_latency_ms")
            else None,
            "median_total_llm_latency_ms": median(numeric(record_rows, "total_llm_latency_ms"))
            if numeric(record_rows, "total_llm_latency_ms")
            else None,
            "mean_llm_latency_ms": mean(numeric(record_rows, "mean_llm_latency_ms"))
            if numeric(record_rows, "mean_llm_latency_ms")
            else None,
            "llm_recorded_runs": len(numeric(record_rows, "total_llm_latency_ms")),
            "errors": sum(1 for row in rows if row.get("error")),
            "nonzero_returncodes": sum(1 for row in rows if row.get("returncode") not in (None, 0)),
            "export_errors": sum(1 for row in rows if row.get("export_error")),
        }

    rows_by_pair = {run_key(row): row for row in run_rows}
    paired_deltas: list[dict[str, Any]] = []
    pair_keys = sorted({(run.task.instance_id, run.attempt) for run in plan})
    for task_id, attempt in pair_keys:
        routewise = rows_by_pair.get((task_id, attempt, "routewise"))
        baseline = rows_by_pair.get((task_id, attempt, "baseline"))
        if not routewise or not baseline:
            continue
        both_successful = is_successful_run(routewise) and is_successful_run(baseline)
        both_recorded = has_request_records(routewise) and has_request_records(baseline)
        if not both_successful:
            continue
        paired_deltas.append(
            {
                "task_id": task_id,
                "attempt": attempt,
                "wall_sec_delta": routewise["wall_sec"] - baseline["wall_sec"],
                "cost_usd_delta": None
                if not both_recorded
                else routewise["total_cost_usd"] - baseline["total_cost_usd"],
                "total_llm_latency_ms_delta": None
                if (
                    not both_recorded
                    or routewise.get("total_llm_latency_ms") is None
                    or baseline.get("total_llm_latency_ms") is None
                )
                else routewise["total_llm_latency_ms"] - baseline["total_llm_latency_ms"],
                "mean_llm_latency_ms_delta": None
                if (
                    not both_recorded
                    or routewise.get("mean_llm_latency_ms") is None
                    or baseline.get("mean_llm_latency_ms") is None
                )
                else routewise["mean_llm_latency_ms"] - baseline["mean_llm_latency_ms"],
                "requests_delta": None
                if not both_recorded
                else routewise["n_requests"] - baseline["n_requests"],
                "mean_ttft_ms_delta": None
                if (
                    not both_recorded
                    or routewise["mean_ttft_ms"] is None
                    or baseline["mean_ttft_ms"] is None
                )
                else routewise["mean_ttft_ms"] - baseline["mean_ttft_ms"],
            }
        )
    record_paired = [p for p in paired_deltas if p["cost_usd_delta"] is not None]
    ttft_paired = [p for p in paired_deltas if p["mean_ttft_ms_delta"] is not None]
    llm_paired = [p for p in paired_deltas if p["total_llm_latency_ms_delta"] is not None]
    mean_llm_paired = [p for p in paired_deltas if p["mean_llm_latency_ms_delta"] is not None]

    return {
        "planned_runs": len(plan),
        "completed_runs": len(run_rows),
        "policy_summary": policy_summary,
        "paired": {
            "wall_pairs": len(paired_deltas),
            "record_pairs": len(record_paired),
            "llm_pairs": len(llm_paired),
            "mean_wall_sec_delta": mean([p["wall_sec_delta"] for p in paired_deltas])
            if paired_deltas
            else None,
            "mean_cost_usd_delta": mean([p["cost_usd_delta"] for p in record_paired])
            if record_paired
            else None,
            "mean_requests_delta": mean([p["requests_delta"] for p in record_paired])
            if record_paired
            else None,
            "mean_ttft_ms_delta": mean([p["mean_ttft_ms_delta"] for p in ttft_paired])
            if ttft_paired
            else None,
            "mean_total_llm_latency_ms_delta": mean(
                [p["total_llm_latency_ms_delta"] for p in llm_paired]
            )
            if llm_paired
            else None,
            "mean_llm_latency_ms_delta": mean(
                [p["mean_llm_latency_ms_delta"] for p in mean_llm_paired]
            )
            if mean_llm_paired
            else None,
            "task_level": {
                "wall_sec_delta": task_level_ci(paired_deltas, "wall_sec_delta"),
                "total_llm_latency_ms_delta": task_level_ci(
                    paired_deltas, "total_llm_latency_ms_delta"
                ),
                "mean_llm_latency_ms_delta": task_level_ci(
                    paired_deltas, "mean_llm_latency_ms_delta"
                ),
                "cost_usd_delta": task_level_ci(paired_deltas, "cost_usd_delta"),
            },
        },
    }


def run_plan(plan: list[PlannedRun], args: argparse.Namespace, output_dir: Path) -> int:
    config = GatewayConfig.from_env(base_url=args.base_url)

    if not config.api_key:
        raise RuntimeError(
            "No public FreeInference API key configured. Set "
            "FREEINFERENCE_STAGE_API_KEY or FreeInference_Stage_API_Key in .env."
        )
    if not config.admin_api_key:
        raise RuntimeError(
            "No admin token configured. Set ADMIN_TOKEN or FREEINFERENCE_ADMIN_API_KEY in .env."
        )
    if not args.skip_preflight:
        preflight_admin_export(config)
        preflight_policy_chat(config, plan)

    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "run_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "base_url": args.base_url,
                "runs": [
                    {
                        "index": run.index,
                        "task": asdict(run.task),
                        "attempt": run.attempt,
                        "policy": run.policy,
                        "model_name": run.model_name,
                        "session_id": run.session_id,
                    }
                    for run in plan
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )

    task_runs_path = output_dir / "task_runs.jsonl"
    request_rows_path = output_dir / "request_rows.jsonl"
    previous_rows = load_previous_rows(task_runs_path) if args.resume else []
    run_rows: list[dict[str, Any]] = latest_rows(previous_rows)
    completed_keys = {run_key(row) for row in run_rows if is_successful_run(row)}
    if args.resume:
        print(
            f"resume: loaded {len(previous_rows)} checkpoint rows, "
            f"skipping {len(completed_keys)} successful runs",
            flush=True,
        )

    if not args.skip_warmup:
        pending = [run for run in plan if planned_key(run) not in completed_keys]
        warmup_task_images(pending, pull_timeout=args.pull_timeout)

    consecutive_failures = 0
    for planned in plan:
        if planned_key(planned) in completed_keys:
            print(
                f"[{planned.index}/{len(plan)}] skip completed "
                f"task={planned.task.instance_id} attempt={planned.attempt} "
                f"policy={planned.policy}",
                flush=True,
            )
            continue
        print(
            f"[{planned.index}/{len(plan)}] task={planned.task.instance_id} "
            f"attempt={planned.attempt} policy={planned.policy} "
            f"model={planned.model_name}",
            flush=True,
        )
        row, request_rows = run_one(planned, config=config, output_dir=output_dir, args=args)
        run_rows.append(row)
        append_jsonl(task_runs_path, row)
        for request_row in request_rows:
            append_jsonl(request_rows_path, request_row)
        print(
            f"  returncode={row['returncode']} wall={row['wall_sec']:.1f}s "
            f"requests={row['n_requests']} cost=${row['total_cost_usd']:.6f} "
            f"llm_ms={row['total_llm_latency_ms']} ttft={row['mean_ttft_ms']}",
            flush=True,
        )
        if row.get("error"):
            print(f"  ERROR: {row['error']}", flush=True)
        elif row.get("export_error"):
            print(f"  EXPORT WARNING: {row['export_error']}", flush=True)
        failed = row.get("error") is not None or row.get("returncode") != 0
        consecutive_failures = consecutive_failures + 1 if failed else 0
        if failed and args.stop_on_error:
            break
        if args.max_consecutive_failures and consecutive_failures >= args.max_consecutive_failures:
            print(
                f"aborting: {consecutive_failures} consecutive failed runs "
                "(tune with --max-consecutive-failures)",
                flush=True,
            )
            break

    run_rows = latest_rows(run_rows)
    backfill_missing_records(run_rows, config=config, request_rows_path=request_rows_path)
    rebuild_missing_llm_fields(run_rows, request_rows_path=request_rows_path)
    run_rows = sorted(run_rows, key=lambda row: int(row.get("index") or 0))
    rewrite_jsonl(task_runs_path, run_rows)
    summary = aggregate_summary(run_rows, plan)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nwrote {task_runs_path}")
    print(f"wrote {summary_path}")
    return 0


def main() -> int:
    load_dotenv(Path(".env"))
    args = parse_args()
    if args.resume and args.output_dir and args.resume.resolve() != args.output_dir.resolve():
        raise RuntimeError("--resume and --output-dir point to different directories.")
    output_dir = args.resume or args.output_dir or default_output_dir()
    plan_path = output_dir / "run_plan.json"
    if args.resume and plan_path.exists():
        plan = load_plan(plan_path)
    else:
        tasks = choose_tasks(args)
        plan = build_plan(tasks, args.attempts)
    print_plan(plan, output_dir)
    if not args.execute:
        print("\nDry run only. Add --execute to run these SWE-bench jobs.")
        return 0
    return run_plan(plan, args, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
