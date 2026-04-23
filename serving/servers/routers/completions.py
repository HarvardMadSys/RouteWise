"""OpenAI-compatible chat completions endpoint with routing and auth."""

from __future__ import annotations

import json
import time
from contextlib import suppress
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from serving.observability.metrics import (
    API_TOKEN_ANOMALIES,
    API_TOKENS,
    normalize_model_label,
    normalize_provider_label,
)
from serving.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from serving.servers.auth import verify_api_key
from serving.servers.deps import get_db_logger, get_rate_limiter, get_router
from serving.servers.rate_limiter import TokenCounter
from serving.utils.logging import get_logger
from serving.utils.token_utils import normalize_usage

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        404: {"model": ErrorResponse, "description": "Model Not Found"},
        429: {"model": ErrorResponse, "description": "Rate Limit Exceeded"},
        500: {"model": ErrorResponse, "description": "Server Error"},
    },
)
async def chat_completions(
    request: Request,
    authorization: str | None = Header(None),
    user_ctx: dict = Depends(verify_api_key),
    router_exec=Depends(get_router),
    rate_limiter=Depends(get_rate_limiter),
    db_logger=Depends(get_db_logger),
) -> dict[str, Any]:
    """Handle chat completion requests with routing and fallback.

    Parses the request body using Pydantic for validation, then routes
    to the appropriate adapter. Streaming and non-streaming flows are
    both supported.
    """
    try:
        body = await request.json()
        payload = ChatCompletionRequest.model_validate(body)
    except Exception as e:
        raise HTTPException(400, "Invalid JSON or schema in request body") from e

    model = payload.model
    messages = [m.model_dump() for m in payload.messages]

    # Debug-only: log inbound message roles to verify client behavior.
    # Note: We intentionally avoid logging message contents to protect privacy.
    try:
        roles = [msg.get("role") for msg in messages]
        tool_count = sum(1 for msg in messages if msg.get("role") == "tool")
        logger.debug(
            f"Inbound roles: model={model}, roles={roles}, tool_messages={tool_count}, total={len(messages)}"
        )
    except Exception:
        # Swallow any logging issues to avoid impacting request handling.
        pass

    # Check if model has routing configured
    if model not in router_exec.routes:
        raise HTTPException(404, f"Model '{model}' not found")

    # Extract parameters
    params: dict[str, Any] = {}
    if payload.temperature is not None:
        params["temperature"] = payload.temperature
    if payload.top_p is not None:
        params["top_p"] = payload.top_p
    if payload.top_k is not None:
        params["top_k"] = payload.top_k
    if payload.min_p is not None:
        params["min_p"] = payload.min_p
    if payload.max_tokens is not None:
        params["max_tokens"] = payload.max_tokens
    if payload.stop is not None:
        params["stop"] = payload.stop
    if payload.seed is not None:
        params["seed"] = payload.seed
    if payload.frequency_penalty is not None:
        params["frequency_penalty"] = payload.frequency_penalty
    if payload.presence_penalty is not None:
        params["presence_penalty"] = payload.presence_penalty
    if payload.tools is not None:
        params["tools"] = payload.tools
    if payload.tool_choice is not None:
        params["tool_choice"] = payload.tool_choice
    if payload.response_format is not None:
        params["response_format"] = payload.response_format.model_dump(by_alias=True)
    # Always record whether this request is streaming for DB analytics
    params["stream"] = bool(payload.stream)

    # Rate limit check with advanced features
    if rate_limiter:
        # Higher priority for authenticated requests (via Authorization or X-API-Key)
        priority = 1 if user_ctx.get("authenticated") else 0
        success, meta = await rate_limiter.acquire_tokens(
            model_id=model,
            messages=messages,
            max_tokens=params.get("max_tokens"),
            priority=priority,
            timeout=30.0,
        )

        if not success:
            error_detail: dict[str, Any] = {
                "error": {
                    "type": "rate_limit_exceeded",
                    "message": meta.get("error", "Rate limit exceeded"),
                    "model": model,
                    "retry_after": meta.get("retry_after", 60),
                }
            }
            if "tokens_requested" in meta:
                error_detail["error"]["tokens_requested"] = meta["tokens_requested"]
            if "queue_size" in meta:
                error_detail["error"]["queue_size"] = meta["queue_size"]

            headers: dict[str, str] = {
                "X-RateLimit-RetryAfter": str(meta.get("retry_after", 60)),
                "X-RateLimit-Model": model,
            }
            status = rate_limiter.get_status(model)
            if status.get("configured"):
                headers.update(
                    {
                        "X-RateLimit-Limit": str(status.get("capacity")),
                        "X-RateLimit-Remaining": str(int(status.get("tokens_available", 0))),
                        "X-RateLimit-Window": str(int(status.get("window_seconds", 0))),
                    }
                )

            raise HTTPException(status_code=429, detail=error_detail, headers=headers)

    # Generate request ID and metadata
    request_id = f"req_{int(time.time() * 1000000)}"
    start_time = time.time()
    is_authenticated = bool(user_ctx.get("authenticated"))
    # Initialize provider early to avoid UnboundLocalError in exception handlers
    provider = "router"
    # Extract a stable session identifier from a single, canonical header.
    # Clients are expected to send X-Session-ID. Starlette headers are case-insensitive.
    session_id = request.headers.get("X-Session-ID")

    metadata = {
        "user_agent": request.headers.get("user-agent"),
        "ip": request.client.host if request.client else None,
        # Preserve legacy field but treat either auth header as authenticated
        "authorization": bool(authorization) or is_authenticated,
        "authenticated": is_authenticated,
        "user_id": user_ctx.get("user_id"),
    }
    if session_id:
        metadata["session_id"] = session_id

    # Helper function to get pricing for a specific provider
    def get_pricing_for_provider(
        provider_name: str, base_url: str | None = None
    ) -> dict[str, str] | None:
        """Find pricing from the actual adapter used (by provider + base_url)."""
        if model not in router_exec.routes:
            return None
        route_config = router_exec.routes[model]

        # Match adapter by provider and optionally base_url
        for adapter, _ in route_config.adapters:
            if not hasattr(adapter, "config"):
                continue
            if adapter.config.provider == provider_name:
                # If base_url provided, match it too (for same provider, different endpoints)
                if (
                    base_url
                    and hasattr(adapter.config, "base_url")
                    and adapter.config.base_url != base_url
                ):
                    continue
                # Found matching adapter
                if hasattr(adapter.config, "pricing"):
                    return adapter.config.pricing
        return None

    def get_adapter_config_for_provider(provider_name: str, base_url: str | None = None) -> Any:
        """Return the adapter config object for the provider/base_url used."""
        if model not in router_exec.routes:
            return None
        route_config = router_exec.routes[model]
        for adapter, _ in route_config.adapters:
            cfg = getattr(adapter, "config", None)
            if not cfg:
                continue
            if cfg.provider != provider_name:
                continue
            if base_url and getattr(cfg, "base_url", None) != base_url:
                continue
            return cfg
        return None

    # Streaming path
    if payload.stream:

        async def stream_generator():
            usage_data = None
            routing_info = None
            chunk_count = 0
            # Accumulate streamed content for DB logging
            final_text = ""
            finish_reason_for_db = "stop"
            # Properly handle tool_calls delta merging by index
            tool_calls_map: dict[int, dict[str, Any]] = {}
            # Track TTFT: time to first token
            ttft_ms: int | None = None
            try:
                # Emit initial assistant role chunk for client compatibility (e.g., Cursor)
                from serving.stream import make_role_chunk

                role_chunk = make_role_chunk(model=model)
                logger.debug(f"Yielding initial role chunk: {role_chunk[:150]}")
                yield role_chunk

                logger.debug(f"Starting to consume adapter stream for model: {model}")
                async for chunk in router_exec.stream_chat_completion(model, messages, **params):
                    chunk_count += 1
                    # Forward adapter SSE chunks with sanitization. Adapters may emit final usage chunk.
                    if chunk_count <= 10 or chunk_count % 10 == 0:
                        logger.debug(f"Chunk {chunk_count} received from adapter: {chunk[:200]}")

                    # Extract usage and routing info from chunks; sanitize before yielding
                    if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
                        try:
                            chunk_json = json.loads(chunk[6:])
                            if chunk_json.get("usage"):
                                usage_data = chunk_json["usage"]
                                logger.debug(
                                    f"Extracted usage from chunk {chunk_count}: {usage_data}"
                                )
                            # Streaming adapters may also include _routing in final chunk
                            if "_routing" in chunk_json:
                                routing_info = chunk_json.get("_routing")
                                logger.debug(
                                    f"Extracted routing from chunk {chunk_count}: {routing_info}"
                                )
                                # Never leak routing info to clients
                                with suppress(Exception):
                                    del chunk_json["_routing"]

                            # Record TTFT at the first meaningful delta (content or tool_calls)
                            if ttft_ms is None:
                                try:
                                    choices_local = chunk_json.get("choices", [])
                                    if choices_local:
                                        delta_local = choices_local[0].get("delta", {})
                                        has_content = bool(delta_local.get("content"))
                                        has_tool_calls = bool(delta_local.get("tool_calls"))
                                        if has_content or has_tool_calls:
                                            ttft_ms = int((time.time() - start_time) * 1000)
                                            logger.debug(
                                                f"TTFT recorded (first delta): {ttft_ms}ms"
                                            )
                                except Exception:
                                    # Best effort only; do not impact streaming on errors.
                                    pass

                            # Accumulate content and finish_reason for DB logging
                            choices = chunk_json.get("choices", [])
                            if choices:
                                choice = choices[0]
                                delta = choice.get("delta", {})

                                # Accumulate content
                                content_piece = delta.get("content")
                                if content_piece:
                                    # Record TTFT at the first actual content token
                                    if ttft_ms is None:
                                        ttft_ms = int((time.time() - start_time) * 1000)
                                        logger.debug(f"TTFT recorded: {ttft_ms}ms")
                                    final_text += content_piece

                                # Handle tool_calls delta merging
                                tool_calls_delta = delta.get("tool_calls")
                                if tool_calls_delta:
                                    for tc_delta in tool_calls_delta:
                                        idx = tc_delta.get("index", 0)
                                        if idx not in tool_calls_map:
                                            tool_calls_map[idx] = {
                                                "index": idx,
                                                "id": tc_delta.get("id", ""),
                                                "type": tc_delta.get("type", "function"),
                                                "function": {"name": "", "arguments": ""},
                                            }

                                        # Merge id if present
                                        if "id" in tc_delta:
                                            tool_calls_map[idx]["id"] = tc_delta["id"]

                                        # Merge type if present
                                        if "type" in tc_delta:
                                            tool_calls_map[idx]["type"] = tc_delta["type"]

                                        # Merge function delta
                                        if "function" in tc_delta:
                                            fn_delta = tc_delta["function"]
                                            if "name" in fn_delta:
                                                tool_calls_map[idx]["function"]["name"] = fn_delta[
                                                    "name"
                                                ]
                                            if "arguments" in fn_delta:
                                                # Arguments are streamed incrementally
                                                tool_calls_map[idx]["function"]["arguments"] += (
                                                    fn_delta["arguments"]
                                                )

                                # Update finish_reason if present
                                fr = choice.get("finish_reason")
                                if fr:
                                    finish_reason_for_db = fr

                            # Yield sanitized chunk to client
                            sanitized_chunk = f"data: {json.dumps(chunk_json)}\n\n"
                            logger.debug(
                                f"Yielding sanitized chunk {chunk_count} to client: {sanitized_chunk[:150]}"
                            )
                            yield sanitized_chunk
                            continue
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"Failed to parse chunk {chunk_count}: {e}")
                            # Fall through to yield original chunk unmodified

                    # Non-JSON or [DONE] chunks pass through
                    logger.debug(f"Yielding chunk {chunk_count} to client: {chunk[:150]}")
                    yield chunk

                logger.info(f"Stream complete: total_chunks={chunk_count}")

                # Reconstruct a complete response object for DB logging
                response_for_db: dict[str, Any] = {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": final_text if final_text else None,
                            },
                            "finish_reason": finish_reason_for_db,
                        }
                    ],
                }

                # Add tool_calls if any were accumulated
                if tool_calls_map:
                    # Convert map to list, sorted by index
                    tool_calls_list = [tc for _, tc in sorted(tool_calls_map.items())]
                    response_for_db["choices"][0]["message"]["tool_calls"] = tool_calls_list

                # Add usage if available
                if usage_data:
                    response_for_db["usage"] = normalize_usage(usage_data) or usage_data

                # Get pricing from actual provider used
                provider = "router"
                pricing = None
                if routing_info:
                    provider = routing_info.get("provider", "router")
                    base_url = routing_info.get("base_url")
                    pricing = get_pricing_for_provider(provider, base_url)
                    if routing_info:
                        metadata.update(routing_info)

                if db_logger:
                    await db_logger.log_request(
                        request_id=request_id,
                        model_id=model,
                        provider=provider,
                        prompt=messages,
                        response=response_for_db,
                        usage=response_for_db.get("usage") if response_for_db else usage_data,
                        latency_ms=int((time.time() - start_time) * 1000),
                        status_code=200,
                        params=(
                            (
                                lambda p: (
                                    p.update(
                                        {
                                            "max_tokens": p.get("max_tokens")
                                            if p.get("max_tokens") is not None
                                            else (
                                                getattr(
                                                    get_adapter_config_for_provider(
                                                        provider,
                                                        routing_info.get("base_url")
                                                        if routing_info
                                                        else None,
                                                    ),
                                                    "max_output_length",
                                                    None,
                                                )
                                            )
                                        }
                                    )
                                    or p
                                )
                            )(dict(params))
                        ),
                        metadata=metadata,
                        ttft_ms=ttft_ms,
                        pricing=pricing,
                    )
            except Exception as exc:
                if db_logger:
                    await db_logger.log_request(
                        request_id=request_id,
                        model_id=model,
                        provider="router",
                        prompt=messages,
                        response=None,
                        usage=None,
                        latency_ms=int((time.time() - start_time) * 1000),
                        status_code=500,
                        error=str(exc),
                        params=params,
                        metadata=metadata,
                        pricing=None,  # Error case - no pricing available
                    )
                if rate_limiter:
                    estimated_tokens = TokenCounter.estimate_tokens(
                        messages, params.get("max_tokens")
                    )
                    await rate_limiter.release_tokens(model, estimated_tokens)

                error_chunk = {"error": {"message": str(exc), "type": "server_error", "code": 500}}
                error_msg = f"data: {json.dumps(error_chunk)}\n\n"
                logger.error(f"Yielding error chunk: {error_msg}")
                yield error_msg

        logger.debug(f"Creating StreamingResponse for model: {model}")
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming path
    try:
        response = await router_exec.chat_completion(model, messages, **params)

        # Always sanitize internal routing metadata from response to client
        provider = "router"
        base_url = None
        if isinstance(response, dict) and "_routing" in response:
            try:
                provider = response["_routing"].get("provider", "router")
                base_url = response["_routing"].get("base_url")
                # Enrich metadata for analytics; safe to skip if no DB logger
                metadata.update(response["_routing"])  # type: ignore[arg-type]
            except Exception:
                # Do not let metadata processing impact client response
                pass
            # Never leak internal routing details to clients
            with suppress(Exception):
                del response["_routing"]

        if db_logger:
            pricing = get_pricing_for_provider(provider, base_url)

            # Normalize usage to extract reasoning_tokens from nested locations
            normalized_usage = normalize_usage(response.get("usage"))

            await db_logger.log_request(
                request_id=request_id,
                model_id=model,
                provider=provider,
                prompt=messages,
                response=response,
                usage=normalized_usage,
                latency_ms=int((time.time() - start_time) * 1000),
                status_code=200,
                params=(
                    (
                        lambda p: (
                            p.update(
                                {
                                    "max_tokens": p.get("max_tokens")
                                    if p.get("max_tokens") is not None
                                    else (
                                        getattr(
                                            get_adapter_config_for_provider(provider, base_url),
                                            "max_output_length",
                                            None,
                                        )
                                    )
                                }
                            )
                            or p
                        )
                    )(dict(params))
                ),
                metadata=metadata,
                pricing=pricing,
            )

        # Emit token counters when usage is available, with anomaly checks
        # Normalize usage to extract reasoning_tokens from nested locations
        raw_usage = response.get("usage", {}) if isinstance(response, dict) else {}
        usage = normalize_usage(raw_usage) or {}
        if usage:
            prompt_tokens_raw = usage.get("prompt_tokens")
            completion_tokens_raw = usage.get("completion_tokens")
            total_tokens_raw = usage.get("total_tokens")
            reasoning_tokens_raw = usage.get("reasoning_tokens")

            try:
                prompt_tokens = int(prompt_tokens_raw or 0)
                completion_tokens = int(completion_tokens_raw or 0)
                reasoning_tokens = int(reasoning_tokens_raw or 0)
                total_tokens = int(
                    total_tokens_raw or (prompt_tokens + completion_tokens + reasoning_tokens)
                )
            except Exception:
                API_TOKEN_ANOMALIES.labels(
                    model=normalize_model_label(model),
                    provider=normalize_provider_label(provider),
                    reason="non_integer",
                ).inc()
                logger.warning(f"Invalid token usage types for {model}/{provider}: {usage}")
                prompt_tokens = completion_tokens = reasoning_tokens = total_tokens = 0

            # Basic sanity: non-negative, totals consistent, and not absurdly large
            max_tokens_cap = 10_000_000
            sane = (
                0 <= prompt_tokens < max_tokens_cap
                and 0 <= completion_tokens < max_tokens_cap
                and 0 <= reasoning_tokens < max_tokens_cap
                and 0 <= total_tokens < max_tokens_cap
                and total_tokens >= prompt_tokens + completion_tokens + reasoning_tokens
            )
            if not sane:
                API_TOKEN_ANOMALIES.labels(
                    model=normalize_model_label(model),
                    provider=normalize_provider_label(provider),
                    reason="invalid_values",
                ).inc()
                logger.warning(f"Token usage anomaly for {model}/{provider}: {usage}")
            else:
                if prompt_tokens:
                    API_TOKENS.labels(
                        model=normalize_model_label(model),
                        provider=normalize_provider_label(provider),
                        direction="prompt",
                    ).inc(prompt_tokens)
                if completion_tokens:
                    API_TOKENS.labels(
                        model=normalize_model_label(model),
                        provider=normalize_provider_label(provider),
                        direction="completion",
                    ).inc(completion_tokens)
                if reasoning_tokens:
                    API_TOKENS.labels(
                        model=normalize_model_label(model),
                        provider=normalize_provider_label(provider),
                        direction="reasoning",
                    ).inc(reasoning_tokens)

        if rate_limiter:
            actual_tokens = response.get("usage", {}).get("total_tokens")
            if actual_tokens:
                estimated = TokenCounter.estimate_tokens(messages, params.get("max_tokens"))
                if abs(actual_tokens - estimated) > estimated * 0.2:
                    logger.warning(
                        f"Token estimation variance for {model}: estimated {estimated}, actual {actual_tokens}"
                    )

        return response

    except Exception as exc:
        if rate_limiter:
            estimated_tokens = TokenCounter.estimate_tokens(messages, params.get("max_tokens"))
            await rate_limiter.release_tokens(model, estimated_tokens)
        if db_logger:
            await db_logger.log_request(
                request_id=request_id,
                model_id=model,
                provider="router",
                prompt=messages,
                response=None,
                usage=None,
                latency_ms=int((time.time() - start_time) * 1000),
                status_code=500,
                error=str(exc),
                params=params,
                metadata=metadata,
                pricing=None,  # Error case - no pricing available
            )
        raise HTTPException(500, str(exc)) from exc
