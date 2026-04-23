"""OpenRouter-compatible models listing routes.

This module exposes endpoints that return a consolidated list of models in an
OpenRouter/OpenAI-compatible schema:

- `/models`
- `/openrouter/models`
- `/v1/models`

The router aggregates multiple backend adapters per logical model and reports
conservative capabilities (e.g., minimum context length, intersection of
sampling parameters). The response is deterministic across concurrent requests;
for example, the `created` field is frozen at import time.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from serving.schemas import ModelItem, ModelList
from serving.servers.deps import get_router

router = APIRouter()

# Fixed creation timestamp captured at import time to ensure
# deterministic responses across concurrent requests.
CREATED_TS = int(time.time())


@router.get("/models")
@router.get("/openrouter/models")
@router.get("/v1/models", response_model=ModelList)
async def list_models(
    router_exec=Depends(get_router),
) -> ModelList:
    """List available models with metadata similar to OpenRouter schema.

    When multiple adapters are registered for a model, the server advertises
    conservative limits (minimum across adapters) to ensure compatibility
    regardless of the routed backend.
    """
    models: list[ModelItem] = []
    emitted_ids: set[str] = set()

    for model_id, route in router_exec.routes.items():
        configs = [adapter.config for adapter, _ in route.adapters]
        if not configs:
            continue

        # Conservative limits across all adapters for this model
        context_length = min(cfg.context_length for cfg in configs)
        max_output_length = min(cfg.max_output_length for cfg in configs)

        # Intersection of supported sampling params across adapters
        supported_params_sets = [set(cfg.supported_params) for cfg in configs]
        if supported_params_sets:
            supported_sampling_parameters = sorted(set.intersection(*supported_params_sets))
        else:
            supported_sampling_parameters = []

        # Supported features per OpenRouter provider doc
        supported_features: list[str] = []
        if any(cfg.supports_tools for cfg in configs):
            supported_features.append("tools")
        if any(cfg.supports_structured_output for cfg in configs):
            supported_features.append("json_mode")
            supported_features.append("structured_outputs")

        # Use the first config for display name/provider/pricing as canonical
        primary_cfg = configs[0]
        canonical_id = primary_cfg.id
        if canonical_id in emitted_ids:
            # Skip aliases; only emit one entry per canonical model id
            continue
        emitted_ids.add(canonical_id)

        model_entry = ModelItem(
            id=canonical_id,
            name=primary_cfg.name,
            created=CREATED_TS,
            owned_by=primary_cfg.provider,
            input_modalities=primary_cfg.input_modalities,
            output_modalities=primary_cfg.output_modalities,
            quantization=primary_cfg.quantization,
            context_length=context_length,
            max_output_length=max_output_length,
            pricing=primary_cfg.pricing,
            supported_sampling_parameters=supported_sampling_parameters,
            supported_features=supported_features,
        )
        # Optional OpenRouter-specific metadata
        if model_id != canonical_id:
            model_entry.openrouter = {"slug": model_id}
        models.append(model_entry)

    return ModelList(data=models)
