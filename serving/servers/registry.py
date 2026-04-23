"""Model registry and configuration loader.

This module builds provider adapters from configuration and registers them on a
``RouteExecutor``. It supports both environment-based and YAML-based
configuration. Prefer YAML (``config/models.yaml``) for reproducibility.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import yaml

from serving.adapters import (
    ClaudeAdapter,
    DeepSeekAdapter,
    GeminiAdapter,
    LlamaAdapter,
    ModelConfig,
    OpenAIAdapter,
    OpenAICompatAdapter,
    ZhipuAdapter,
)

if TYPE_CHECKING:
    from pathlib import Path

    from routing.executor import RouteExecutor


def _make_adapter(kind: str, cfg: dict[str, Any]):
    """Construct a provider adapter from a kind string and model config.

    Args:
        kind: Adapter kind (``"vllm"``, ``"sglang"``, ``"claude"``, ``"deepseek"``, ``"gemini"``, ``"llama"``, ``"openai"``, ``"zhipu"``,
              ``"chutes"``, ``"featherless"``, ``"openai_compat"``).
        cfg: ``ModelConfig`` keyword arguments.

    Returns:
        A concrete adapter instance.

    Raises:
        ValueError: When ``kind`` is unknown.
    """
    model_cfg = ModelConfig(**cfg)

    # All OpenAI-compatible services use the same adapter
    if kind in ("vllm", "sglang", "chutes", "featherless", "openai_compat"):
        return OpenAICompatAdapter(model_cfg)

    if kind == "claude":
        return ClaudeAdapter(model_cfg)
    if kind == "deepseek":
        return DeepSeekAdapter(model_cfg)
    if kind == "gemini":
        return GeminiAdapter(model_cfg)
    if kind == "llama":
        return LlamaAdapter(model_cfg)
    if kind == "openai":
        return OpenAIAdapter(model_cfg)
    if kind == "zhipu":
        return ZhipuAdapter(model_cfg)

    raise ValueError(f"Unknown adapter kind: {kind}")


def register_from_models_yaml(router: RouteExecutor, path: Path) -> int:
    """Register models and routes from a YAML configuration file.

    Example schema::

        models:
          - id: llama-3.3-70b-instruct
            name: Llama 3.3 70B Instruct
            provider: llama
            base_url: ${LLAMA_BASE_URL}
            api_key: ${LLAMA_API_KEY}
            context_length: 131072
            max_output_length: 8192
            supports_tools: true
            supports_structured_output: true
            supported_params: [temperature, top_p, top_k, min_p, max_tokens, stop, seed]
            aliases: ["llama-3.3-70b-instruct"]
            route:
              - kind: llama
                weight: 1.0
                base_url: ${LLAMA_BASE_URL}
                api_key: ${LLAMA_API_KEY}

    Args:
        router: Executor to receive registered routes.
        path: Path to the YAML configuration file.

    Returns:
        int: Number of registered route identifiers (including aliases).
    """
    if not path.exists():
        return 0
    data = yaml.safe_load(path.read_text()) or {}
    models: list[dict[str, Any]] = data.get("models", [])
    count = 0
    for m in models:
        # Environment expansion for base_url/api_key in both top-level and route entries
        def expand_env(val: str | None) -> str | None:
            if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
                return os.getenv(val[2:-1])
            return val

        # Build primary config
        top_cfg = {
            k: m.get(k)
            for k in (
                "id",
                "name",
                "provider",
                "base_url",
                "api_key",
                "aliases",
                "provider_model_id",
                "quantization",
                "input_modalities",
                "output_modalities",
                "context_length",
                "max_output_length",
                "supports_tools",
                "supports_structured_output",
                "supported_params",
                "pricing",
            )
        }
        if top_cfg.get("base_url"):
            top_cfg["base_url"] = expand_env(top_cfg["base_url"])  # type: ignore
        if top_cfg.get("api_key"):
            top_cfg["api_key"] = expand_env(top_cfg["api_key"])  # type: ignore
        if top_cfg.get("provider_model_id"):
            top_cfg["provider_model_id"] = expand_env(top_cfg["provider_model_id"])  # type: ignore

        # If no explicit route list, use a single route targeting the primary config
        routes = m.get("route") or [
            {
                "kind": top_cfg.get("provider"),
                "weight": 1.0,
                **{k: top_cfg.get(k) for k in ("base_url", "api_key")},
            }
        ]

        adapters_with_weights = []
        for r in routes:
            kind = r.get("kind") or top_cfg.get("provider")
            base_url = expand_env(r.get("base_url") or top_cfg.get("base_url"))
            api_key = expand_env(r.get("api_key") or top_cfg.get("api_key"))
            weight = float(r.get("weight", 1.0))

            # Adapter config inherits from top-level model config
            adapter_cfg = dict(top_cfg)
            adapter_cfg["base_url"] = base_url
            adapter_cfg["api_key"] = api_key
            adapter_cfg["provider"] = kind

            route_provider_model_id = r.get("provider_model_id")
            if route_provider_model_id is not None:
                adapter_cfg["provider_model_id"] = expand_env(route_provider_model_id)

            # Route-level pricing override (key for cost-aware routing in Phase 2)
            if "pricing" in r:
                adapter_cfg["pricing"] = r["pricing"]

            adapter = _make_adapter(kind, adapter_cfg)
            adapters_with_weights.append((adapter, weight))

        # Register canonical id and aliases
        model_id = str(top_cfg["id"])  # type: ignore
        aliases = (top_cfg.get("aliases") or []) or []
        for alias in [model_id, *aliases]:
            router.register_route(alias, adapters_with_weights)
            count += 1

    return count
