"""Drive SWE-bench instances one at a time through mini-SWE-agent + the gateway.

Each instance runs in mini-SWE-agent's Docker sandbox; every LLM call carries the
per-task ``X-Session-ID`` so its cost/latency can be pulled from the gateway log
afterwards. The gateway routing is chosen by ``model_name`` (minimax-m2.5 = fixed
baseline, minimax-fast = RouteWise); ``policy`` is just a label for the run.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import minisweagent

from ..runtime.gateway import GatewayClient
from ..runtime.miniswe_config import write_model_config

if TYPE_CHECKING:
    from ..runtime.config import GatewayConfig
    from ..runtime.schemas import LLMRequestRecord
    from .tasks import TaskSpec


_TEXTBASED_MODEL_CLASS = "minisweagent.models.litellm_textbased_model.LitellmTextbasedModel"


def _bundled_config(textbased: bool) -> str:
    """Path to mini-SWE-agent's bundled SWE-bench base config.

    Textbased uses the backticks variant (plain-text ReAct, no function-calling),
    which avoids the content=None assistant messages the gateway currently mishandles.
    """
    name = "swebench_backticks.yaml" if textbased else "swebench.yaml"
    return str(Path(minisweagent.__file__).parent / "config" / "benchmarks" / name)


def _mini_extra_bin() -> str:
    """The ``mini-extra`` CLI living next to the running interpreter."""
    return str(Path(sys.executable).parent / "mini-extra")


@dataclass
class TaskRun:
    """Result of running one SWE-bench instance, with its gateway-logged requests."""

    instance_id: str
    policy: str
    model_name: str
    session_id: str
    returncode: int
    started_at: float
    finished_at: float
    trajectory_path: str
    records: list[LLMRequestRecord] = field(default_factory=list)
    export_error: str | None = None

    @property
    def wall_sec(self) -> float:
        """Wall-clock seconds for the whole agent run."""
        return self.finished_at - self.started_at

    @property
    def n_requests(self) -> int:
        """Number of gateway requests this task made."""
        return len(self.records)

    @property
    def total_cost_usd(self) -> float:
        """Summed gateway-billed cost across the task's requests."""
        return sum(r.cost_usd or 0.0 for r in self.records)

    @property
    def mean_ttft_ms(self) -> float | None:
        """Mean TTFT over requests that reported one."""
        vals = [r.ttft_ms for r in self.records if r.ttft_ms is not None]
        return sum(vals) / len(vals) if vals else None

    @property
    def total_llm_latency_ms(self) -> float | None:
        """Summed gateway-reported request latency, excluding sandbox/tool time."""
        vals = [r.latency_ms for r in self.records if r.latency_ms is not None]
        return sum(vals) if vals else None


def run_task(
    task: TaskSpec,
    *,
    policy: str,
    model_name: str,
    config: GatewayConfig,
    output_dir: str | Path,
    attempt: int = 1,
    textbased: bool = True,
    cost_limit: float = 0.0,
    pull_timeout: int = 1800,
    env_cmd_timeout: int = 300,
    timeout_sec: int = 3600,
) -> TaskRun:
    """Run one SWE-bench instance and read its per-request cost/latency back."""
    output_dir = Path(output_dir)
    session_id = f"swe-{policy}-{task.instance_id}-{attempt}"
    cfg_path = write_model_config(
        output_dir / "configs" / f"{session_id}.yaml",
        session_id=session_id,
        api_base=config.base_url,
        model_name=model_name,
    )
    traj_path = output_dir / "traj" / f"{session_id}.json"
    log_path = output_dir / f"{session_id}.log"
    traj_path.parent.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "OPENAI_API_KEY": config.require_api_key(),
        "MSWEA_COST_TRACKING": "ignore_errors",
    }
    cmd = [
        _mini_extra_bin(),
        "swebench-single",
        "-c",
        _bundled_config(textbased),
        "-c",
        str(cfg_path),
        # emulation on arm64 needs longer image-pull and per-command timeouts
        "-c",
        f"environment.pull_timeout={pull_timeout}",
        "-c",
        f"environment.timeout={env_cmd_timeout}",
    ]
    if textbased:
        cmd += ["--model-class", _TEXTBASED_MODEL_CLASS]
    cmd += [
        "--subset",
        task.subset,
        "--split",
        task.split,
        "-i",
        task.instance_id,
        "-y",
        "--exit-immediately",
        "--cost-limit",
        str(cost_limit),
        "-o",
        str(traj_path),
    ]

    window_start = datetime.now(timezone.utc) - timedelta(seconds=10)
    started = datetime.now(timezone.utc).timestamp()
    with log_path.open("w") as log:
        proc = subprocess.run(
            cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_sec
        )
    finished = datetime.now(timezone.utc).timestamp()

    export_error = None
    try:
        records = GatewayClient(config).export_by_session(
            session_id, start_time=window_start, model_id=model_name.split("/")[-1]
        )
    except Exception as exc:
        # The agent run has already completed at this point. Keep the task result
        # and let the batch script backfill request rows later instead of
        # reporting the whole SWE-bench run as failed.
        records = []
        export_error = f"{type(exc).__name__}: {exc}"
    return TaskRun(
        instance_id=task.instance_id,
        policy=policy,
        model_name=model_name,
        session_id=session_id,
        returncode=proc.returncode,
        started_at=started,
        finished_at=finished,
        trajectory_path=str(traj_path),
        records=records,
        export_error=export_error,
    )
