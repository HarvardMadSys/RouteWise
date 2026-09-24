"""Generate per-task mini-SWE-agent model config pointing at the gateway.

mini-SWE-agent selects the gateway routing purely by model name: `minimax-m2.5`
is the fixed-routing baseline, `minimax-fast` is the same model under RouteWise.
Each task gets a unique `X-Session-ID` header so its requests can be read back
from the gateway log and aggregated per task. The API key is left to litellm's
`OPENAI_API_KEY` env var so it never lands in a config file on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_AGENT_MODEL = "openai/minimax-m2.5"


def build_model_config(
    *,
    session_id: str,
    api_base: str,
    model_name: str = DEFAULT_AGENT_MODEL,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int | None = 1,
    seed: int | None = None,
) -> dict[str, Any]:
    """Build the mini-SWE-agent ``model`` config block for one task.

    Args:
        session_id: Stamped as the ``X-Session-ID`` header; the gateway logs it,
            so a task's requests are retrievable via the admin export endpoint.
        api_base: Gateway root (with or without a trailing ``/v1``).
        model_name: litellm model name with provider prefix, e.g.
            ``openai/minimax-m2.5`` (baseline) or ``openai/minimax-fast`` (RouteWise).
        temperature: Sampling temperature; pinned to 0 for reproducibility.
        top_p: Nucleus sampling cap. Keep at 1.0 when top_k is used.
        top_k: Candidate cap. Sent through OpenAI ``extra_body`` because LiteLLM's
            OpenAI provider does not treat ``top_k`` as a standard parameter.
        seed: Provider seed when the upstream supports it.
    """
    model_kwargs: dict[str, Any] = {
        "api_base": api_base.rstrip("/").removesuffix("/v1") + "/v1",
        "temperature": temperature,
        "top_p": top_p,
        "extra_headers": {"X-Session-ID": session_id},
    }
    if seed is not None:
        model_kwargs["seed"] = seed
    if top_k is not None:
        model_kwargs["extra_body"] = {"top_k": top_k}
    return {
        "model": {
            "model_name": model_name,
            "model_kwargs": model_kwargs,
            "cost_tracking": "ignore_errors",
        }
    }


def write_model_config(path: str | Path, **kwargs: Any) -> Path:
    """Write a per-task model-config YAML to merge over the bundled swebench config.

    Used as ``mini-extra swebench-single -c swebench.yaml -c <path> -i <id>``.
    """
    path = Path(path)
    cfg = build_model_config(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path
