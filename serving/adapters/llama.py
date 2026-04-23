"""Llama API adapter for OpenAI-compatible interface."""

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from contextlib import suppress
from typing import Any

from serving.stream import done_sentinel, make_final_usage_chunk
from serving.utils.tokens import estimate_prompt_tokens, estimate_text_tokens

from .base import BaseAdapter, UsageInfo


class LlamaAdapter(BaseAdapter):  # type: ignore[no-any-unimported]
    """Adapter for Llama API using OpenAI-compatible endpoints."""

    def _extract_text(self, content: Any) -> str:
        """Extract plain text from OpenAI/Cursor rich content.

        - Lists: join all text blocks.
        - Dicts: prefer 'text', otherwise JSON-serialize as fallback.
        - Primitives: cast to str.
        - None: empty string.
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    if item:
                        parts.append(item)
                elif isinstance(item, dict):
                    t = item.get("text")
                    if t is None and item.get("type") == "text":
                        t = item.get("text")
                    if isinstance(t, str) and t:
                        parts.append(t)
            return "\n".join(parts)
        if isinstance(content, dict):
            if isinstance(content.get("text"), str):
                return content["text"]
            try:
                return json.dumps(content, ensure_ascii=False)
            except Exception:
                return str(content)
        return str(content)

    def _normalize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize messages to satisfy stricter OpenAI-compatible servers.

        - Flattens content to plain strings.
        - Skips empty/None content entries.
        - Preserves roles and tool-related fields.
        """
        out: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            text = self._extract_text(m.get("content"))

            # For assistant messages with tool_calls but no content, use None instead of ""
            # Llama API requires null (not empty string) per OpenAI spec
            if role == "assistant" and not text and m.get("tool_calls") is not None:
                msg: dict[str, Any] = {"role": role, "content": None}
            else:
                msg: dict[str, Any] = {"role": role, "content": text}

            # Preserve tool-calls context for assistant/tool messages when present
            if m.get("tool_calls") is not None:
                msg["tool_calls"] = m["tool_calls"]
            if m.get("tool_call_id") is not None:
                msg["tool_call_id"] = m["tool_call_id"]
            if m.get("name") is not None:
                msg["name"] = m["name"]
            # Skip messages that end up empty to reduce 400 risk
            if role in ("user", "assistant", "system", "developer", "tool"):
                # Allow empty content if message has tool_calls or is system/developer/tool role
                if (
                    text
                    or role in ("system", "developer", "tool")
                    or m.get("tool_calls") is not None
                ):
                    out.append(msg)
            else:
                # Unknown role: keep as-is but only if text present
                if text:
                    out.append(msg)
        return out

    def _normalize_json_schema(self, schema: dict[str, Any] | None) -> dict[str, Any]:
        """Normalize OpenAI JSON schema for tool parameters for stricter servers.

        - Ensures array types include an ``items`` schema (defaults to string).
        - Recursively converts nested object/array schemas and preserves common
          annotations like ``description``, ``enum``, ``default``, and numeric
          bounds.
        """
        if not isinstance(schema, dict):
            return {"type": "object", "properties": {}}

        def _norm(s: dict[str, Any]) -> dict[str, Any]:
            out: dict[str, Any] = {}
            typ = s.get("type")
            if isinstance(typ, list):
                typ = next((t for t in typ if t != "null"), typ[0] if typ else None)
            if not isinstance(typ, str):
                typ = "string"
            out["type"] = typ

            # Preserve useful hints
            for k in (
                "description",
                "default",
                "pattern",
                "minimum",
                "maximum",
                "minLength",
                "maxLength",
            ):
                if k in s:
                    out[k] = s[k]
            if "enum" in s and isinstance(s["enum"], list) and s["enum"]:
                out["enum"] = s["enum"]

            if typ == "object":
                props = s.get("properties") or {}
                out_props: dict[str, Any] = {}
                for name, sub in props.items():
                    out_props[name] = _norm(sub) if isinstance(sub, dict) else {"type": "string"}
                out["properties"] = out_props
                req = s.get("required") or []
                if req:
                    out["required"] = req
            elif typ == "array":
                items = s.get("items")
                out["items"] = _norm(items) if isinstance(items, dict) else {"type": "string"}
                for k in ("minItems", "maxItems", "uniqueItems"):
                    if k in s:
                        out[k] = s[k]
            return out

        return _norm(schema)

    def _normalize_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Ensure tool parameter schemas are acceptable by strict OpenAI compat servers."""
        if not tools:
            return tools
        normalized: list[dict[str, Any]] = []
        for t in tools:
            if t.get("type") != "function":
                normalized.append(t)
                continue
            fn = dict(t.get("function", {}))
            if not fn.get("name"):
                normalized.append(t)
                continue
            params = fn.get("parameters")
            if isinstance(params, dict):
                fn["parameters"] = self._normalize_json_schema(params)
            normalized.append({"type": "function", "function": fn})
        return normalized

    async def chat_completion(
        self, messages: list[dict[str, Any]], **params: Any
    ) -> dict[str, Any]:
        """Send chat completion request to Llama API."""
        validated_params = self.validate_params(params)
        from serving.utils.logging import get_logger

        logger = get_logger(__name__)

        # Use provider_model_id if specified, otherwise fall back to id
        model_id = self.config.provider_model_id or self.config.id

        # Normalize message content to plain text to avoid 400 on rich blocks
        llama_messages = self._normalize_messages(messages)

        payload = {"model": model_id, "messages": llama_messages, **validated_params}

        # Llama API specific parameters
        if "top_k" in params and "top_k" in self.config.supported_params:
            payload["top_k"] = params["top_k"]

        if "min_p" in params and "min_p" in self.config.supported_params:
            payload["min_p"] = params["min_p"]

        if "frequency_penalty" in params and "frequency_penalty" in self.config.supported_params:
            payload["frequency_penalty"] = params["frequency_penalty"]

        if "presence_penalty" in params and "presence_penalty" in self.config.supported_params:
            payload["presence_penalty"] = params["presence_penalty"]

        # Tool calling support for Llama 3.3
        if params.get("tools") and self.config.supports_tools:
            payload["tools"] = self._normalize_tools(params["tools"])  # normalize array schemas
            default_tool_choice = os.getenv("LLAMA_TOOL_CHOICE_DEFAULT")
            if params.get("tool_choice") is not None:
                payload["tool_choice"] = params["tool_choice"]
            elif default_tool_choice:
                payload["tool_choice"] = default_tool_choice
            with suppress(Exception):
                logger.debug(
                    f"Non-stream: tools={len(payload['tools'])}, tool_choice={payload.get('tool_choice')}"
                )

        # Structured output support
        if params.get("response_format") and self.config.supports_structured_output:
            payload["response_format"] = params["response_format"]

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            # Strip whitespace and newlines from API key to avoid header validation errors
            api_key = (
                self.config.api_key.strip()
                if isinstance(self.config.api_key, str)
                else self.config.api_key
            )
            headers["Authorization"] = f"Bearer {api_key}"

        # Use the /inference endpoint for Llama API
        # Use standard OpenAI-compatible endpoint for Llama
        url = f"{self.config.base_url}/chat/completions"

        data = await self.http.json_post_with_retry(url, json=payload, headers=headers)

        # Check if usage is provided, otherwise estimate
        usage = None
        if "usage" in data:
            usage = UsageInfo(
                prompt_tokens=data["usage"].get("prompt_tokens", 0),
                completion_tokens=data["usage"].get("completion_tokens", 0),
                total_tokens=data["usage"].get("total_tokens", 0),
            )
        else:
            # Estimate tokens if not provided
            # Use normalized messages for more accurate estimation
            content = data.get("content", "")
            prompt_tokens = estimate_prompt_tokens(llama_messages)
            completion_tokens = estimate_text_tokens(content)
            usage = UsageInfo(
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                total_tokens=int(prompt_tokens + completion_tokens),
            )

        # Extract tool calls if present
        tool_calls = None
        # Prefer tool_calls inside the choice.message as per OpenAI schema
        if (
            isinstance(data.get("choices"), list)
            and data["choices"]
            and isinstance(data["choices"][0].get("message"), dict)
        ):
            msg0 = data["choices"][0]["message"]
            if msg0.get("tool_calls"):
                tool_calls = msg0.get("tool_calls")
            elif msg0.get("function_call"):
                # Normalize legacy function_call to tool_calls
                fc = msg0.get("function_call") or {}
                name = fc.get("name") or ""
                arguments = fc.get("arguments") or "{}"
                tool_calls = [
                    {
                        "id": f"call_{int(__import__('time').time() * 1000)}",
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                ]
        elif "tool_calls" in data:
            tool_calls = data["tool_calls"]

        # Extract content from OpenAI-compatible response format
        content = ""
        if "choices" in data and data["choices"] and "message" in data["choices"][0]:
            content = data["choices"][0]["message"].get("content", "")
        else:
            # Fallback to direct content field for compatibility
            content = data.get("content", "")

        # Extract finish_reason from choices (OpenAI-compatible format)
        finish_reason = "stop"
        if (
            isinstance(data.get("choices"), list)
            and data["choices"]
            and "finish_reason" in data["choices"][0]
        ):
            finish_reason = data["choices"][0].get("finish_reason", "stop")

        # Format response to OpenAI standard
        response = self.format_response(
            content=content,
            model=self.config.id,
            usage=usage,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )
        return response  # type: ignore[no-any-return]

    async def stream_chat_completion(
        self, messages: list[dict[str, Any]], **params: Any
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion response from Llama API.

        This implementation aims to be robust against upstream quirks where the
        provider does not emit a terminal ``[DONE]`` sentinel and also keeps the
        connection open. In such cases, we enforce an idle timeout to end the
        stream gracefully and still emit a final usage packet + ``[DONE]`` to
        the client in OpenAI-compatible format.
        """
        validated_params = self.validate_params(params)
        from serving.utils.logging import get_logger

        logger = get_logger(__name__)

        # Use provider_model_id if specified, otherwise fall back to id
        model_id = self.config.provider_model_id or self.config.id

        # Normalize message content to plain text to avoid 400 on rich blocks
        llama_messages = self._normalize_messages(messages)

        payload = {
            "model": model_id,
            "messages": llama_messages,
            "stream": True,
            **validated_params,
        }

        # Add Llama-specific parameters
        for param in ["top_k", "min_p", "frequency_penalty", "presence_penalty"]:
            if param in params and param in self.config.supported_params:
                payload[param] = params[param]

        if params.get("tools") and self.config.supports_tools:
            payload["tools"] = self._normalize_tools(params["tools"])  # normalize array schemas
            default_tool_choice = os.getenv("LLAMA_TOOL_CHOICE_DEFAULT")
            if params.get("tool_choice") is not None:
                payload["tool_choice"] = params["tool_choice"]
            elif default_tool_choice:
                payload["tool_choice"] = default_tool_choice
            try:
                logger.debug(
                    f"Stream: tools={len(payload['tools'])}, tool_choice={payload.get('tool_choice')}"
                )
                # Debug: log first tool definition
                if payload["tools"]:
                    logger.debug(f"First tool: {json.dumps(payload['tools'][0], indent=2)}")
                # Debug: log messages
                logger.debug(f"Message count: {len(payload['messages'])}")
                for i, msg in enumerate(payload["messages"][-3:]):  # Log last 3 messages
                    logger.debug(
                        f"Message {i}: role={msg.get('role')}, has_content={bool(msg.get('content'))}, has_tool_calls={bool(msg.get('tool_calls'))}"
                    )
            except Exception as e:
                logger.debug(f"Debug logging failed: {e}")
                pass

        if params.get("response_format") and self.config.supports_structured_output:
            payload["response_format"] = params["response_format"]

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            # Strip whitespace and newlines from API key to avoid header validation errors
            api_key = (
                self.config.api_key.strip()
                if isinstance(self.config.api_key, str)
                else self.config.api_key
            )
            headers["Authorization"] = f"Bearer {api_key}"

        # Use standard OpenAI-compatible endpoint for Llama
        url = f"{self.config.base_url}/chat/completions"

        total_content = ""
        prompt_tokens_override: int | None = None
        finish_reason = "stop"
        has_tool_calls = False  # Track if we've seen any tool_calls (for finish_reason)

        # Ask for SSE explicitly for best compatibility
        headers = {**headers, "Accept": "text/event-stream"}

        # Enforce an idle timeout to avoid hanging forever if upstream never
        # sends [DONE] nor closes the connection (observed with many complex tools).
        # Default: 15s, overridable via env LLAMA_STREAM_IDLE_TIMEOUT_SECS.
        try:
            idle_timeout = float(os.getenv("LLAMA_STREAM_IDLE_TIMEOUT_SECS", "15"))
        except Exception:
            idle_timeout = 15.0

        stream_iter = self.http.stream_post(url, json=payload, headers=headers)
        try:
            while True:
                try:
                    line = await asyncio.wait_for(stream_iter.__anext__(), timeout=idle_timeout)
                except asyncio.TimeoutError:
                    # Gracefully stop on idle to prevent deadlock with clients.
                    break
                except StopAsyncIteration:
                    break

                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    break
                try:
                    chunk_data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                if "usage" in chunk_data:
                    prompt_tokens_override = chunk_data["usage"].get(
                        "prompt_tokens", prompt_tokens_override
                    )

                choices = chunk_data.get("choices") or []
                if choices:
                    choice = choices[0]
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta") or {}

                    # Handle content chunks (following OpenAI/Gemini adapter pattern)
                    # Note: Router already sends an initial role chunk, so we skip role deltas
                    content = delta.get("content")
                    if content:
                        total_content += content
                        yield self.format_stream_chunk(content, self.config.id)

                    # Forward tool_calls delta if present (following OpenAI adapter pattern)
                    tool_calls_delta = delta.get("tool_calls")
                    if tool_calls_delta:
                        has_tool_calls = True
                        # Remove any role field from delta to avoid duplicate role chunks
                        # since the router already emits an initial role-only chunk.
                        try:
                            chunk_copy = chunk_data.copy()
                            chunk_copy["model"] = self.config.id
                            # Deep copy minimal path we mutate to keep performance reasonable
                            ch0 = chunk_copy.get("choices", [{}])[0]
                            d0 = dict(ch0.get("delta") or {})
                            d0.pop("role", None)
                            ch0["delta"] = d0
                            chunk_copy["choices"][0] = ch0
                        except Exception:
                            chunk_copy = chunk_data
                            chunk_copy["model"] = self.config.id
                        yield f"data: {json.dumps(chunk_copy)}\n\n"

                    # Normalize legacy function_call delta to tool_calls
                    function_call = delta.get("function_call")
                    if isinstance(function_call, dict):
                        name = function_call.get("name") or ""
                        args = function_call.get("arguments") or "{}"
                        # Keep arguments as-is (string) per OpenAI stream format
                        chunk_copy = {
                            "id": chunk_data.get("id")
                            or f"chatcmpl-{int(__import__('time').time() * 1000)}",
                            "object": "chat.completion.chunk",
                            "created": chunk_data.get("created") or int(__import__("time").time()),
                            "model": self.config.id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "id": f"call_{int(__import__('time').time() * 1000)}",
                                                "type": "function",
                                                "function": {"name": name, "arguments": args},
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk_copy)}\n\n"
                    # Continue consuming stream
                    continue
        finally:
            # Best-effort: close/cleanup the underlying iterator if needed
            try:
                aclose = getattr(stream_iter, "aclose", None)
                if callable(aclose):
                    await aclose()
            except Exception:
                pass

        # After loop ends (either [DONE] or stream closed), send final chunks
        # If tool calling occurred, finish_reason should be "tool_calls" not "stop"
        if has_tool_calls:
            finish_reason = "tool_calls"

        yield make_final_usage_chunk(
            model=self.config.id,
            messages=messages,
            total_content=total_content,
            prompt_tokens_override=prompt_tokens_override,
            finish_reason=finish_reason,
            provider=self.config.provider,
            base_url=self.config.base_url,
        )
        yield done_sentinel()
