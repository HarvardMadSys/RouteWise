"""Google Gemini API adapter with cache and thinking token tracking."""

import json
import time
from collections.abc import AsyncGenerator
from typing import Any

from serving.utils.tokens import estimate_prompt_tokens, estimate_text_tokens

from .base import BaseAdapter, UsageInfo


class GeminiAdapter(BaseAdapter):
    """Adapter for Google Gemini 2.0/2.5 models.

    This adapter translates OpenAI-style chat messages to Google
    Generative Language API payloads. It is defensive about Cursor/
    OpenAI rich message payloads by extracting only textual content
    for ``parts[].text`` and skipping empty/None values to avoid
    400 Bad Request errors from the Gemini endpoints.
    """

    def _convert_messages_to_gemini(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Convert OpenAI-style messages into Gemini request body.

        - Flattens Cursor/OpenAI rich content into plain text.
        - Skips empty/None parts to satisfy Gemini validation.
        - Maps roles: user -> "user", assistant -> "model".
        - Collects a systemInstruction with an explicit system role.
        """

        def _extract_text(content: Any) -> str:
            """Extract a best-effort text string from mixed content.

            Handles cases where content is a list of blocks (OpenAI style),
            a single mapping, or a primitive. Non-textual blocks (images,
            tool call metadata) are ignored. Dicts without a clear text
            field are JSON-serialized to preserve debugging signal.
            """
            if content is None:
                return ""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts: list[str] = []
                for item in content:
                    if isinstance(item, str):
                        texts.append(item)
                    elif isinstance(item, dict):
                        # OpenAI format blocks
                        t = item.get("text") if "text" in item else None
                        # Some clients nest as {"type": "text", "text": "..."}
                        if t is None and item.get("type") == "text":
                            t = item.get("text")
                        if isinstance(t, str) and t:
                            texts.append(t)
                        # Ignore image_url/inputs; Gemini needs inline/file refs
                return "\n".join([s for s in texts if s])
            if isinstance(content, dict):
                if isinstance(content.get("text"), str):
                    return content["text"]
                # Fallback: JSON-serialize for debuggability
                try:
                    return json.dumps(content, ensure_ascii=False)
                except Exception:
                    return str(content)
            return str(content)

        system_instruction: str | None = None
        contents: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")
            raw_content = msg.get("content")

            if role == "system":
                text = _extract_text(raw_content)
                if text:
                    system_instruction = text
                continue

            if role == "user":
                text = _extract_text(raw_content)
                if text:
                    contents.append({"role": "user", "parts": [{"text": text}]})
                continue

            if role == "assistant":
                text = _extract_text(raw_content)
                if text:
                    contents.append({"role": "model", "parts": [{"text": text}]})
                continue

            if role == "tool":
                # Map OpenAI tool result to Gemini functionResponse part.
                name = msg.get("name") or "tool"
                # Try to parse JSON content; fall back to string
                response_payload: Any
                if isinstance(raw_content, dict | list):
                    response_payload = raw_content
                else:
                    try:
                        response_payload = json.loads(raw_content or "{}")
                    except Exception:
                        response_payload = {
                            "output": str(raw_content) if raw_content is not None else ""
                        }
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {"functionResponse": {"name": name, "response": response_payload}}
                        ],
                    }
                )
                continue

        request_body: dict[str, Any] = {"contents": contents}
        if system_instruction:
            request_body["systemInstruction"] = {
                "role": "system",
                "parts": [{"text": system_instruction}],
            }

        return request_body

    def _apply_generation_config(self, params: dict[str, Any]) -> dict[str, Any]:
        config = {}

        if "max_tokens" in params:
            config["maxOutputTokens"] = params["max_tokens"]

        if "temperature" in params:
            config["temperature"] = params["temperature"]

        if "top_p" in params:
            config["topP"] = params["top_p"]

        if "top_k" in params:
            # Keep within Gemini's accepted range if set.
            try:
                tk = int(params["top_k"])  # defensive cast
                if tk >= 1:
                    config["topK"] = tk
            except Exception:
                pass

        if "stop" in params:
            config["stopSequences"] = (
                params["stop"] if isinstance(params["stop"], list) else [params["stop"]]
            )

        if params.get("response_format", {}).get("type") == "json_object":
            config["responseMimeType"] = "application/json"

        return config

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI tool definitions to Gemini functionDeclarations.

        - Consolidates all functions into a single Tool object.
        - Ensures array schemas include an ``items`` field to satisfy
          Gemini validation (avoids 400 INVALID_ARGUMENT on arrays).
        - Recursively maps nested object/array schemas and preserves
          useful keywords like ``enum``, ``description``, ``default``.
        """

        def _to_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
            """Recursively normalize OpenAI JSON schema to Gemini-compatible."""
            if not isinstance(schema, dict):
                return {"type": "string"}

            out: dict[str, Any] = {}

            # Basic type mapping with fallback
            typ = schema.get("type")
            if isinstance(typ, list):
                # Simplify union types by preferring first; Gemini schema is
                # conservative. If "null" is included, ignore it.
                typ = next((t for t in typ if t != "null"), typ[0] if typ else None)
            if not isinstance(typ, str):
                typ = "string"
            out["type"] = typ

            # Preserve common annotations
            if "description" in schema:
                out["description"] = schema["description"]
            if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
                out["enum"] = schema["enum"]
            if "default" in schema:
                out["default"] = schema["default"]
            for k in ("minimum", "maximum", "minLength", "maxLength", "pattern"):
                if k in schema:
                    out[k] = schema[k]

            # Nested structures
            if typ == "object":
                props = schema.get("properties") or {}
                out_props: dict[str, Any] = {}
                for name, subschema in props.items():
                    out_props[name] = _to_gemini_schema(subschema)
                out["properties"] = out_props
                req = schema.get("required") or []
                if req:
                    out["required"] = req
            elif typ == "array":
                items = schema.get("items")
                # Gemini requires items; default to string if absent
                out["items"] = (
                    _to_gemini_schema(items) if isinstance(items, dict) else {"type": "string"}
                )
                # Optional: minItems/maxItems
                for k in ("minItems", "maxItems", "uniqueItems"):
                    if k in schema:
                        out[k] = schema[k]

            return out

        declarations: list[dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") != "function":
                continue
            func = tool.get("function", {})
            if not func or not func.get("name"):
                continue
            gemini_func: dict[str, Any] = {
                "name": func["name"],
                "description": func.get("description", ""),
            }

            if "parameters" in func and isinstance(func["parameters"], dict):
                params_schema = func["parameters"] or {}
                gemini_func["parameters"] = _to_gemini_schema(params_schema) or {
                    "type": "object",
                    "properties": {},
                }

            declarations.append(gemini_func)

        return [{"functionDeclarations": declarations}] if declarations else []

    def _apply_tool_config(self, params: dict[str, Any]) -> dict[str, Any]:
        """Translate OpenAI tool_choice to Gemini toolConfig when applicable."""
        tool_choice = params.get("tool_choice")
        if not tool_choice:
            return {}

        cfg: dict[str, Any] = {}
        if isinstance(tool_choice, str):
            if tool_choice.lower() == "none":
                cfg = {"functionCallingConfig": {"mode": "NONE"}}
            # "auto" and other strings map to default behavior; skip
        elif isinstance(tool_choice, dict):
            # Expect {"type":"function","function":{"name":"..."}}
            fn = tool_choice.get("function", {}) if tool_choice.get("type") == "function" else {}
            name = fn.get("name")
            if name:
                cfg = {
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": [name],
                    }
                }

        return {"toolConfig": cfg} if cfg else {}

    async def chat_completion(self, messages: list[dict[str, Any]], **params) -> dict[str, Any]:
        """Execute non-streaming chat completion with thinking and cache token extraction."""
        request_body = self._convert_messages_to_gemini(messages)

        generation_config = self._apply_generation_config(params)
        if generation_config:
            request_body["generationConfig"] = generation_config

        if params.get("tools"):
            request_body["tools"] = self._convert_tools(params["tools"])
        # Optional tool choice mapping
        tool_cfg = self._apply_tool_config(params)
        if tool_cfg:
            request_body.update(tool_cfg)

        # Use provider_model_id if available, otherwise fall back to config.id
        model_name = getattr(self.config, "provider_model_id", None) or self.config.id
        url = (
            f"{self.config.base_url}/models/{model_name}:generateContent?key={self.config.api_key}"
        )

        data = await self.http.json_post_with_retry(url, json=request_body)

        if "candidates" not in data or not data["candidates"]:
            raise ValueError(f"Unexpected Gemini response structure: {json.dumps(data)}")

        candidate = data["candidates"][0]

        # Handle case where content or parts might be missing
        # This can happen when model hits MAX_TOKENS during thinking phase
        if "content" not in candidate:
            raise ValueError(f"No content in candidate: {json.dumps(candidate)}")

        content = candidate.get("content", {})
        content_parts = content.get("parts", [])

        # If no parts, the model likely hit token limit during thinking
        # or content was filtered. Return empty response.

        text_content = ""
        tool_calls = []

        for part in content_parts:
            if "text" in part:
                text_content += part["text"]
            elif "functionCall" in part:
                func_call = part["functionCall"]
                tool_calls.append(
                    {
                        "id": f"call_{int(time.time() * 1000)}",
                        "type": "function",
                        "function": {
                            "name": func_call["name"],
                            "arguments": json.dumps(func_call.get("args", {})),
                        },
                    }
                )

        if "usageMetadata" in data:
            usage_meta = data["usageMetadata"]
            # Handle different field names in different API versions
            prompt_tokens = usage_meta.get("promptTokenCount", 0)
            total_tokens = usage_meta.get("totalTokenCount", 0)
            cached_tokens = usage_meta.get("cachedContentTokenCount", 0)

            # Preview models may not include candidatesTokenCount
            # In that case, calculate it from total - prompt - thoughts
            completion_tokens = usage_meta.get("candidatesTokenCount")
            reasoning_tokens = usage_meta.get("thoughtsTokenCount", 0)
            if completion_tokens is None:
                # Calculate completion tokens: total - prompt - thoughts
                calculated = total_tokens - prompt_tokens - reasoning_tokens
                # Validate calculation: fallback to estimation if invalid or zero with content
                if calculated > 0:
                    completion_tokens = calculated
                elif calculated == 0 and text_content:
                    # Edge case: totalTokenCount=0 but model returned text
                    # This means usage metadata is incomplete, estimate from response
                    completion_tokens = estimate_text_tokens(text_content)
                    total_tokens = prompt_tokens + int(completion_tokens) + reasoning_tokens
                elif calculated < 0:
                    # totalTokenCount missing or incomplete, estimate from response
                    completion_tokens = estimate_text_tokens(text_content)
                    # Recalculate total if it was missing
                    if total_tokens == 0:
                        total_tokens = prompt_tokens + int(completion_tokens) + reasoning_tokens
                else:
                    # calculated == 0 and no text_content, legitimately empty
                    completion_tokens = 0

            usage = UsageInfo(
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                total_tokens=int(total_tokens),
                reasoning_tokens=int(reasoning_tokens),
                cache_read_tokens=int(cached_tokens),  # Gemini: cached tokens are read from cache
            )
        else:
            prompt_tokens = estimate_prompt_tokens(messages)
            completion_tokens = estimate_text_tokens(text_content)
            usage = UsageInfo(
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                total_tokens=int(prompt_tokens + completion_tokens),
            )

        finish_reason_map = {
            "STOP": "stop",
            "MAX_TOKENS": "length",
            "SAFETY": "content_filter",
            "OTHER": "stop",
        }
        finish_reason = finish_reason_map.get(candidate.get("finishReason", "STOP"), "stop")

        return self.format_response(
            content=text_content,
            model=self.config.id,
            usage=usage,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=finish_reason,
        )

    async def stream_chat_completion(
        self, messages: list[dict[str, Any]], **params
    ) -> AsyncGenerator[str, None]:
        """Execute streaming chat completion with thinking and cache token extraction."""
        request_body = self._convert_messages_to_gemini(messages)

        generation_config = self._apply_generation_config(params)
        if generation_config:
            request_body["generationConfig"] = generation_config

        if params.get("tools"):
            request_body["tools"] = self._convert_tools(params["tools"])
        # Optional tool choice mapping
        tool_cfg = self._apply_tool_config(params)
        if tool_cfg:
            request_body.update(tool_cfg)

        # Use provider_model_id if available, otherwise fall back to config.id
        model_name = getattr(self.config, "provider_model_id", None) or self.config.id
        # Prefer SSE for reliable incremental tokens with Gemini; add alt=sse.
        url = (
            f"{self.config.base_url}/models/{model_name}:streamGenerateContent"
            f"?key={self.config.api_key}&alt=sse"
        )

        # Debug: log request body when tool messages present
        from serving.utils.logging import get_logger

        logger = get_logger(__name__)
        if any(msg.get("role") == "tool" for msg in messages):
            logger.debug(
                f"Sending request with tool results: {json.dumps(request_body, indent=2)[:2000]}"
            )

        total_content = ""
        prompt_tokens = 0
        reasoning_tokens = 0
        cached_tokens = 0
        completion_tokens_upstream = 0
        total_tokens_upstream = 0
        finish_reason_raw = "STOP"
        has_function_call = False

        # Auto-detect upstream streaming format (SSE vs NDJSON) and parse both.
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        async for line in self.http.stream_post(
            url, json=request_body, headers=headers, mode="auto"
        ):
            if not line:
                continue
            try:
                # Accept either raw JSON line or SSE 'data: ' prefixed line.
                raw = line[6:] if line.startswith("data: ") else line
                if raw.strip() == "[DONE]":
                    break
                data = json.loads(raw)

                emitted_any = False
                if "candidates" in data:
                    candidate = data["candidates"][0]

                    # Track finish reason from upstream
                    if candidate.get("finishReason"):
                        finish_reason_raw = candidate["finishReason"]

                    # Handle case where content or parts might be missing
                    if "content" not in candidate:
                        # Some events might carry top-level incremental text
                        # without nesting into content/parts. Fall back below.
                        content_parts = []
                    else:
                        content_parts = candidate["content"].get("parts", [])

                    for part in content_parts:
                        if "text" in part:
                            text = part["text"]
                            total_content += text
                            yield self.format_stream_chunk(text, self.config.id)
                            emitted_any = True
                        elif "functionCall" in part:
                            # Mark that we detected a function call
                            has_function_call = True

                            func_call = part["functionCall"]
                            name = func_call.get("name")
                            args_obj = func_call.get("args", {})
                            try:
                                args_str = json.dumps(args_obj)
                            except Exception:
                                args_str = "{}"
                            # Emit OpenAI-compatible tool_calls delta
                            # Note: Gemini returns complete functionCall in one chunk,
                            # unlike incremental streaming. We still follow OpenAI spec.
                            chunk = {
                                "id": f"chatcmpl-{int(time.time() * 1000)}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": self.config.id,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "tool_calls": [
                                                {
                                                    "index": 0,  # Required by OpenAI spec
                                                    "id": f"call_{int(time.time() * 1000)}",
                                                    "type": "function",
                                                    "function": {
                                                        "name": name or "",
                                                        "arguments": args_str,
                                                    },
                                                }
                                            ]
                                        },
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                            emitted_any = True

                # Fallbacks: some frames surface incremental text at top-level
                if not emitted_any:
                    for key in ("text", "deltaText", "partialText"):
                        if isinstance(data.get(key), str) and data[key]:
                            t = data[key]
                            total_content += t
                            yield self.format_stream_chunk(t, self.config.id)
                            emitted_any = True
                            break

                # Handle empty content case (e.g., after tool execution)
                # When Gemini returns finishReason with empty content, we should still
                # emit at least one chunk to signal the response has started/completed
                if not emitted_any and candidate.get("finishReason"):
                    # Emit an empty content delta to satisfy client expectations
                    yield self.format_stream_chunk("", self.config.id)
                    emitted_any = True

                # Extract all usage metadata from each chunk
                if "usageMetadata" in data:
                    usage_meta = data["usageMetadata"]
                    prompt_tokens = usage_meta.get("promptTokenCount", prompt_tokens)
                    reasoning_tokens = usage_meta.get("thoughtsTokenCount", reasoning_tokens)
                    cached_tokens = usage_meta.get("cachedContentTokenCount", cached_tokens)
                    completion_tokens_upstream = usage_meta.get(
                        "candidatesTokenCount", completion_tokens_upstream
                    )
                    total_tokens_upstream = usage_meta.get("totalTokenCount", total_tokens_upstream)

            except json.JSONDecodeError:
                continue

        # Prefer upstream precise values, fall back to estimation
        final_prompt_tokens = prompt_tokens or estimate_prompt_tokens(messages)

        if completion_tokens_upstream > 0:
            final_completion_tokens = completion_tokens_upstream
        else:
            final_completion_tokens = estimate_text_tokens(total_content)

        if total_tokens_upstream > 0:
            final_total_tokens = total_tokens_upstream
        else:
            # Calculate: prompt + completion + reasoning
            final_total_tokens = final_prompt_tokens + final_completion_tokens + reasoning_tokens

        # Map finish_reason from Gemini to OpenAI format
        # Note: Gemini returns "STOP" for successful tool calls, but OpenAI protocol
        # requires finish_reason="tool_calls" when tool_calls are present in the response.
        finish_reason_map = {
            "STOP": "stop",
            "MAX_TOKENS": "length",
            "SAFETY": "content_filter",
            "OTHER": "stop",
        }
        mapped_finish_reason = finish_reason_map.get(finish_reason_raw, "stop")

        # Override finish_reason to "tool_calls" if we emitted tool_calls in this response
        # This is required by OpenAI protocol for consistency
        if has_function_call:
            mapped_finish_reason = "tool_calls"

        # Build final usage chunk
        usage_chunk = {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.config.id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": mapped_finish_reason}],
            "usage": {
                "prompt_tokens": int(final_prompt_tokens),
                "completion_tokens": int(final_completion_tokens),
                "total_tokens": int(final_total_tokens),
            },
        }

        # Add optional fields for OpenAI compatibility
        if reasoning_tokens > 0:
            usage_chunk["usage"]["reasoning_tokens"] = int(reasoning_tokens)
        if cached_tokens > 0:
            usage_chunk["usage"]["cache_read_tokens"] = int(cached_tokens)

        yield f"data: {json.dumps(usage_chunk)}\n\n"
        yield "data: [DONE]\n\n"
