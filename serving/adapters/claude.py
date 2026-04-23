"""Claude (Anthropic) adapter for Google Vertex AI API.

This adapter interfaces with Claude models through Google Vertex
hybrid API which uses custom endpoints:
- Non-streaming: /v1:rawPredict
- Streaming: /v1:streamRawPredict

Key differences from standard Anthropic API:
- Uses 'api-key' header instead of 'x-api-key'
- Requires 'anthropic_version' parameter
- Different endpoint paths (Google Vertex AI style)
"""

from __future__ import annotations

import json
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from serving.stream import done_sentinel, make_final_usage_chunk

from .base import BaseAdapter, UsageInfo


class ClaudeAdapter(BaseAdapter):
    """Adapter for Claude models via Google Vertex API."""

    # Anthropic API version
    ANTHROPIC_VERSION = "vertex-2023-10-16"

    async def chat_completion(
        self, messages: list[dict[str, Any]], **params: Any
    ) -> dict[str, Any]:
        """Execute a non-streaming chat completion request.

        Args:
            messages: Chat history in OpenAI-compatible format.
            **params: Parameters including system, max_tokens, temperature, etc.

        Returns:
            OpenAI-compatible response dictionary.
        """
        validated_params = self.validate_params(params)

        # Build Claude-specific payload
        # Note: API uses Vertex AI style - model is in URL, not payload
        payload: dict[str, Any] = {
            "anthropic_version": self.ANTHROPIC_VERSION,
            "messages": self._convert_messages(messages),
            # Claude Vertex expects message blocks; stream flag left to endpoint semantics
        }

        # Extract system prompt if present; otherwise derive from messages
        if params.get("system"):
            payload["system"] = params["system"]
        else:
            sys_text = self._extract_system(messages)
            if sys_text:
                payload["system"] = sys_text

        # Add validated parameters
        payload["max_tokens"] = validated_params.get("max_tokens", self.config.max_output_length)

        if "temperature" in validated_params:
            payload["temperature"] = validated_params["temperature"]

        if "top_p" in validated_params:
            payload["top_p"] = validated_params["top_p"]

        if "top_k" in validated_params and "top_k" in self.config.supported_params:
            payload["top_k"] = validated_params["top_k"]

        if "stop" in validated_params:
            # Claude calls these "stop_sequences"
            payload["stop_sequences"] = validated_params["stop"]

        # Tools support
        if params.get("tools") and self.config.supports_tools:
            converted_tools = self._convert_tools(params["tools"])
            if converted_tools:
                payload["tools"] = converted_tools

                # Convert tool_choice from OpenAI format to Claude format
                if params.get("tool_choice"):
                    tool_choice = params["tool_choice"]
                    # OpenAI: "auto", "required", "none", or {"type": "function", "function": {"name": "..."}}
                    # Claude: {"type": "auto"}, {"type": "any"}, {"type": "tool", "name": "..."}
                    if isinstance(tool_choice, str):
                        if tool_choice == "auto":
                            # "auto" is default, don't send it
                            pass
                        elif tool_choice == "required" or tool_choice == "any":
                            payload["tool_choice"] = {"type": "any"}
                        elif tool_choice == "none":
                            # Don't use tools - remove them
                            del payload["tools"]
                    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                        # Specific function selection
                        func_name = tool_choice.get("function", {}).get("name")
                        if func_name:
                            payload["tool_choice"] = {"type": "tool", "name": func_name}

        # Build endpoint URL (non-streaming)
        endpoint = f"{self.config.base_url.rstrip('/')}/v1:rawPredict"

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "api-key": self.config.api_key,
        }

        data = await self.http.json_post_with_retry(
            endpoint, json=payload, headers=headers, timeout=None, retries=3
        )

        # Extract content from Claude response format
        content_blocks = data.get("content", [])
        content = ""
        tool_calls = None

        # Build tool_calls with contiguous indices starting from 0
        tool_index = 0
        for block in content_blocks:
            if block.get("type") == "text":
                content += block.get("text", "")
            elif block.get("type") == "tool_use":
                # Convert Claude tool_use to OpenAI tool_calls format
                # Ensure indices are contiguous per OpenAI schema
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    {
                        "index": tool_index,
                        "id": block.get("id"),
                        "type": "function",
                        "function": {
                            "name": block.get("name"),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    }
                )
                tool_index += 1

        # Extract usage information
        usage_data = data.get("usage", {})
        # Separate non-cached prompt tokens from cached reads for accurate billing
        input_tokens = int(usage_data.get("input_tokens", 0) or 0)
        cache_read_input_tokens = int(usage_data.get("cache_read_input_tokens", 0) or 0)
        cache_creation_input_tokens = int(usage_data.get("cache_creation_input_tokens", 0) or 0)
        output_tokens = int(usage_data.get("output_tokens", 0) or 0)
        thinking_tokens = int(usage_data.get("thinking_tokens", 0) or 0)

        usage = UsageInfo(
            prompt_tokens=max(0, input_tokens - cache_read_input_tokens),
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            # Claude includes cache tokens
            cache_read_tokens=cache_read_input_tokens,
            cache_write_tokens=cache_creation_input_tokens,
            # Forward extended thinking tokens as reasoning tokens for cost tracking
            reasoning_tokens=thinking_tokens,
        )

        # Map Claude's stop_reason to OpenAI's finish_reason
        stop_reason = data.get("stop_reason", "end_turn")
        finish_reason = self._map_stop_reason(stop_reason)

        return self.format_response(
            content=content,
            model=self.config.id,
            usage=usage,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

    async def stream_chat_completion(
        self, messages: list[dict[str, Any]], **params: Any
    ) -> AsyncGenerator[str, None]:
        """Execute a streaming chat completion request.

        Yields SSE-formatted chunks compatible with OpenAI clients.
        """
        validated_params = self.validate_params(params)

        # Build Claude-specific payload
        # Note: API uses Vertex AI style - model is in URL, not payload
        # Also: streaming is determined by endpoint, not "stream" parameter
        converted_msgs = self._convert_messages(messages)
        payload: dict[str, Any] = {
            "anthropic_version": self.ANTHROPIC_VERSION,
            "messages": converted_msgs,
        }

        # Extract system prompt if present; otherwise derive from messages
        if params.get("system"):
            payload["system"] = params["system"]
        else:
            sys_text = self._extract_system(messages)
            if sys_text:
                payload["system"] = sys_text

        # Add validated parameters
        payload["max_tokens"] = validated_params.get("max_tokens", self.config.max_output_length)

        if "temperature" in validated_params:
            payload["temperature"] = validated_params["temperature"]

        if "top_p" in validated_params:
            payload["top_p"] = validated_params["top_p"]

        if "top_k" in validated_params and "top_k" in self.config.supported_params:
            payload["top_k"] = validated_params["top_k"]

        if "stop" in validated_params:
            payload["stop_sequences"] = validated_params["stop"]

        # Tools support
        if params.get("tools") and self.config.supports_tools:
            converted_tools = self._convert_tools(params["tools"])
            if converted_tools:
                payload["tools"] = converted_tools

                # Convert tool_choice from OpenAI format to Claude format
                if params.get("tool_choice"):
                    tool_choice = params["tool_choice"]
                    # OpenAI: "auto", "required", "none", or {"type": "function", "function": {"name": "..."}}
                    # Claude: {"type": "auto"}, {"type": "any"}, {"type": "tool", "name": "..."}
                    if isinstance(tool_choice, str):
                        if tool_choice == "auto":
                            # "auto" is default, don't send it
                            pass
                        elif tool_choice == "required" or tool_choice == "any":
                            payload["tool_choice"] = {"type": "any"}
                        elif tool_choice == "none":
                            # Don't use tools - remove them
                            del payload["tools"]
                    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                        # Specific function selection
                        func_name = tool_choice.get("function", {}).get("name")
                        if func_name:
                            payload["tool_choice"] = {"type": "tool", "name": func_name}

        # Build endpoint URL (streaming)
        endpoint = f"{self.config.base_url.rstrip('/')}/v1:streamRawPredict"

        headers = {
            "Content-Type": "application/json",
            # Allow server to choose SSE or NDJSON; HTTP client will auto-detect
            "Accept": "*/*",
            "api-key": self.config.api_key,
        }

        total_content = ""
        input_tokens = 0
        output_tokens = 0
        cache_read_input_tokens = 0
        cache_creation_input_tokens = 0
        finish_reason = "stop"
        line_count = 0

        # Track tool use for streaming
        current_tool_use: dict[str, Any] | None = None
        tool_input_buffer = ""
        current_tool_index = 0  # Track tool call index for OpenAI streaming format
        completed_tool_calls: list[dict[str, Any]] = []  # Collect all tool calls before sending

        from serving.utils.logging import get_logger

        logger = get_logger(__name__)
        logger.debug(f"Starting stream to: {endpoint}")
        logger.debug(f"Payload summary: {json.dumps(self._summarize_messages(converted_msgs))}")

        try:
            # Google Vertex API may return non-streaming JSON instead of SSE
            # Try to use ndjson mode which is more tolerant
            async for line in self.http.stream_post(
                endpoint, json=payload, headers=headers, mode="auto", timeout=None
            ):
                line_count += 1
                if not line.strip():
                    continue

                # Parse JSON line
                try:
                    chunk_data = json.loads(line)
                    if line_count <= 10:
                        logger.debug(f"Chunk {line_count}: {json.dumps(chunk_data)[:300]}")
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON at line {line_count}: {line[:100]}")
                    continue

                # Check for upstream API errors
                if "Code" in chunk_data and "Error" in chunk_data:
                    error_code = chunk_data.get("Code")
                    error_msg = chunk_data.get("Error")
                    logger.error(f"[CLAUDE UPSTREAM ERROR] Code: {error_code}, Error: {error_msg}")
                    logger.error(
                        "[CLAUDE UPSTREAM ERROR] This is likely an issue with the API, not our code."
                    )
                    logger.error(
                        "[CLAUDE UPSTREAM ERROR] Falling back to non-streaming endpoint..."
                    )
                    # Raise exception to trigger fallback
                    raise RuntimeError(f"Upstream API error: {error_msg}")

                chunk_type = chunk_data.get("type")
                if line_count <= 10:
                    logger.debug(f"Chunk type at line {line_count}: {chunk_type}")

                # Google Vertex API sometimes returns a complete "message" object instead of streaming chunks
                if chunk_type == "message":
                    # Extract content blocks - both text and tool_use
                    content_blocks = chunk_data.get("content", [])
                    full_text = ""
                    tool_calls_list = []

                    # Router emits initial role chunk; no need to track here.
                    for block in content_blocks:
                        block_type = block.get("type")
                        if block_type == "text":
                            full_text += block.get("text", "")
                        elif block_type == "tool_use":
                            # Convert Claude tool_use to OpenAI streaming format
                            # CRITICAL: Must include "index" field for streaming!
                            tool_call = {
                                "index": len(tool_calls_list),  # contiguous index within tool_calls
                                "id": block.get("id"),
                                "type": "function",
                                "function": {
                                    "name": block.get("name"),
                                    "arguments": json.dumps(block.get("input", {})),
                                },
                            }
                            tool_calls_list.append(tool_call)
                            logger.debug(
                                f"Found tool_use: {block.get('name')}, id: {block.get('id')}, index: {tool_call['index']}"
                            )

                    # Extract usage
                    usage_data = chunk_data.get("usage", {})
                    input_tokens = int(usage_data.get("input_tokens", 0) or 0)
                    output_tokens = int(usage_data.get("output_tokens", 0) or 0)
                    cache_read_input_tokens = int(usage_data.get("cache_read_input_tokens", 0) or 0)
                    cache_creation_input_tokens = int(
                        usage_data.get("cache_creation_input_tokens", 0) or 0
                    )
                    thinking_tokens = int(usage_data.get("thinking_tokens", 0) or 0)

                    # Yield content chunk if present
                    if full_text:
                        total_content = full_text
                        # Log full text length and preview for debugging
                        logger.debug(
                            f"Full text length: {len(full_text)}, preview: {full_text[:500]}"
                        )
                        logger.debug(f"Full text end: ...{full_text[-200:]}")
                        # Don't include role - router layer handles initial role chunk
                        yield self.format_stream_chunk(full_text, self.config.id)

                    # Map stop_reason
                    stop_reason = chunk_data.get("stop_reason", "end_turn")
                    finish_reason = self._map_stop_reason(stop_reason)

                    # If tools present
                    if tool_calls_list:
                        # Forward tool_calls to client per OpenAI streaming protocol
                        # Router layer handles initial role chunk, we just send tools
                        tool_chunk = self.format_tool_chunk(tool_calls_list, self.config.id)
                        logger.debug(f"Yielding {len(tool_calls_list)} tool calls")
                        yield tool_chunk

                        # Ensure finish_reason reflects tool_calls for OpenAI clients
                        finish_reason = "tool_calls"
                        finish_chunk = self.format_stream_chunk(
                            "", self.config.id, finish_reason=finish_reason
                        )
                        logger.debug(f"Yielding finish_reason={finish_reason}")
                        yield finish_chunk

                        # Send usage even for tool_calls to enable proper logging and billing
                        usage_data = chunk_data.get("usage", {})
                        input_tokens_msg = usage_data.get("input_tokens", 0)
                        cache_read_msg = usage_data.get("cache_read_input_tokens", 0)
                        cache_creation_msg = usage_data.get("cache_creation_input_tokens", 0)
                        output_tokens_msg = usage_data.get("output_tokens", 0)

                        yield make_final_usage_chunk(
                            model=self.config.id,
                            messages=messages,
                            total_content=total_content,
                            prompt_tokens_override=max(0, input_tokens_msg - cache_read_msg)
                            if input_tokens_msg > 0
                            else None,
                            completion_tokens_override=output_tokens_msg
                            if output_tokens_msg > 0
                            else None,
                            finish_reason=finish_reason,
                            provider=self.config.provider,
                            base_url=self.config.base_url,
                            cache_read_tokens=cache_read_msg,
                            cache_write_tokens=cache_creation_msg,
                            reasoning_tokens=thinking_tokens,
                        )
                        yield done_sentinel()
                        return
                    else:
                        # For text responses, yield role in the first content chunk
                        if full_text:
                            # Role already included in format_stream_chunk if content exists
                            pass

                    # Yield final usage chunk
                    yield make_final_usage_chunk(
                        model=self.config.id,
                        messages=messages,
                        total_content=total_content,
                        prompt_tokens_override=max(0, input_tokens - cache_read_input_tokens)
                        if input_tokens > 0
                        else None,
                        completion_tokens_override=output_tokens if output_tokens > 0 else None,
                        finish_reason=finish_reason if not tool_calls_list else None,
                        provider=self.config.provider,
                        base_url=self.config.base_url,
                        cache_read_tokens=cache_read_input_tokens,
                        cache_write_tokens=cache_creation_input_tokens,
                        reasoning_tokens=thinking_tokens,
                    )
                    yield done_sentinel()
                    return

                # Content block start (tool_use or text)
                if chunk_type == "content_block_start":
                    block = chunk_data.get("content_block", {})
                    block_type = block.get("type")
                    logger.debug(f"Block start at line {line_count}: {block_type}")
                    if block_type == "tool_use":
                        tool_id = block.get("id")
                        tool_name = block.get("name")
                        logger.debug(
                            f"Tool start at line {line_count}: {tool_name}, ID: {tool_id}, Index: {current_tool_index}"
                        )

                        # Start accumulating tool use with OpenAI streaming format
                        current_tool_use = {
                            "index": current_tool_index,  # OpenAI streaming requires index
                            "id": tool_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": "",  # Will be built up from deltas
                            },
                        }
                        tool_input_buffer = ""

                # Content delta
                elif chunk_type == "content_block_delta":
                    delta = chunk_data.get("delta", {})
                    delta_type = delta.get("type")
                    if line_count <= 10:
                        logger.debug(
                            f"Delta at line {line_count}: type={delta_type}, delta={delta}"
                        )

                    if delta_type == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            total_content += text
                            if line_count <= 5:
                                logger.debug(f"Yielding text at line {line_count}: {text[:100]}")
                            yield self.format_stream_chunk(text, self.config.id)

                    elif delta_type == "input_json_delta":
                        # Tool use - JSON input is being streamed
                        partial_json = delta.get("partial_json", "")
                        logger.debug(f"Tool input delta at line {line_count}: {partial_json[:200]}")

                        # Accumulate the JSON input
                        tool_input_buffer += partial_json

                # Content block stop - collect tool call (don't send yet, may be multiple)
                elif chunk_type == "content_block_stop":
                    if current_tool_use is not None:
                        # Finalize the tool call arguments
                        current_tool_use["function"]["arguments"] = tool_input_buffer

                        # Add to completed tools list (don't send immediately)
                        completed_tool_calls.append(current_tool_use)
                        logger.debug(
                            f"Collected tool call #{len(completed_tool_calls)}: {current_tool_use.get('function', {}).get('name')}"
                        )

                        # Reset for next tool
                        current_tool_use = None
                        tool_input_buffer = ""
                        current_tool_index += 1
                        # Continue processing - don't return yet (may have more tools)

                # Message delta (usage info)
                elif chunk_type == "message_delta":
                    usage_delta = chunk_data.get("usage", {})
                    if usage_delta:
                        output_tokens = int(usage_delta.get("output_tokens", output_tokens) or 0)
                    stop_reason = chunk_data.get("delta", {}).get("stop_reason")
                    if stop_reason:
                        finish_reason = self._map_stop_reason(stop_reason)

                # Message start (input tokens)
                elif chunk_type == "message_start":
                    message = chunk_data.get("message", {})
                    usage_data = message.get("usage", {})
                    if usage_data:
                        input_tokens = int(usage_data.get("input_tokens", 0) or 0)

                # Message stop
                elif chunk_type == "message_stop":
                    # If we collected tool calls, send them now
                    if completed_tool_calls:
                        logger.debug(
                            f"Sending {len(completed_tool_calls)} collected tool calls at message stop"
                        )
                        # Send all tool calls at once
                        tool_chunk = self.format_tool_chunk(completed_tool_calls, self.config.id)
                        yield tool_chunk

                        # Ensure finish_reason reflects tool_calls for OpenAI streaming clients
                        finish_reason = "tool_calls"
                        # Build final usage chunk from upstream usage with cache separation
                        non_cached_prompt = max(0, input_tokens - cache_read_input_tokens)
                        usage_obj = {
                            "prompt_tokens": non_cached_prompt,
                            "completion_tokens": int(output_tokens),
                            "total_tokens": int(input_tokens + output_tokens),
                        }
                        if cache_read_input_tokens > 0:
                            usage_obj["cache_read_tokens"] = int(cache_read_input_tokens)
                        if cache_creation_input_tokens > 0:
                            usage_obj["cache_write_tokens"] = int(cache_creation_input_tokens)

                        final_chunk = {
                            "id": f"chatcmpl-{int(time.time() * 1000)}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": self.config.id,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                            "usage": usage_obj,
                            "_routing": {
                                "provider": self.config.provider,
                                "base_url": self.config.base_url,
                            },
                        }
                        yield f"data: {json.dumps(final_chunk)}\n\n"
                        yield done_sentinel()
                    else:
                        # No tools - normal text response, send usage from upstream
                        non_cached_prompt = max(0, input_tokens - cache_read_input_tokens)
                        usage_obj = {
                            "prompt_tokens": non_cached_prompt,
                            "completion_tokens": int(output_tokens),
                            "total_tokens": int(input_tokens + output_tokens),
                        }
                        if cache_read_input_tokens > 0:
                            usage_obj["cache_read_tokens"] = int(cache_read_input_tokens)
                        if cache_creation_input_tokens > 0:
                            usage_obj["cache_write_tokens"] = int(cache_creation_input_tokens)

                        final_chunk = {
                            "id": f"chatcmpl-{int(time.time() * 1000)}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": self.config.id,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                            "usage": usage_obj,
                            "_routing": {
                                "provider": self.config.provider,
                                "base_url": self.config.base_url,
                            },
                        }
                        yield f"data: {json.dumps(final_chunk)}\n\n"
                        yield done_sentinel()
                    break
        except Exception as e:
            import aiohttp

            logger.error(f"[CLAUDE STREAM ERROR] {type(e).__name__}: {e}")
            if isinstance(e, aiohttp.ClientResponseError):
                logger.error(f"Response status: {e.status}, message: {e.message}")
                logger.error(f"Request info: {e.request_info}")
                # Try to get detailed error message
                try:
                    error_body = (
                        await e.response.text() if hasattr(e, "response") else "No response body"
                    )
                    logger.error(f"Response body: {error_body[:500]}")
                except Exception:
                    pass
            # Fail-fast: propagate error immediately instead of fallback
            raise

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-format messages to Claude format.

        Claude doesn't use a separate 'system' role in messages array.
        System messages should be extracted and passed as 'system' parameter.
        """
        converted: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # Skip system messages (handled separately)
            if role == "system":
                continue

            # Map user role and normalize content to Claude blocks
            if role == "user":
                if isinstance(content, str):
                    content_blocks = [{"type": "text", "text": content}]
                elif isinstance(content, list):
                    content_blocks = content
                else:
                    content_blocks = [{"type": "text", "text": json.dumps(content or "")}]
                converted.append({"role": "user", "content": content_blocks})

            # Assistant role: either tool_calls (OpenAI) -> tool_use (Claude), or plain text
            elif role == "assistant":
                blocks: list[dict[str, Any]] = []
                # Preserve assistant text content if present
                if isinstance(content, str) and content.strip():
                    blocks.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    # Pass through structured blocks (e.g., text/image)
                    blocks.extend([b for b in content if isinstance(b, dict | str)])
                elif content:
                    # Fallback: dump unknown content to text
                    with suppress(Exception):
                        blocks.append({"type": "text", "text": json.dumps(content)})

                # Append tool_use blocks if tool_calls provided
                tool_calls = msg.get("tool_calls")
                if isinstance(tool_calls, list) and tool_calls:
                    for tc in tool_calls:
                        func = (tc or {}).get("function", {})
                        name = func.get("name")
                        args_raw = func.get("arguments") or "{}"
                        try:
                            args_obj = json.loads(args_raw)
                        except Exception:
                            args_obj = {}
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.get("id"),
                                "name": name,
                                "input": args_obj,
                            }
                        )

                if blocks:
                    converted.append({"role": "assistant", "content": blocks})
            elif role == "tool":
                # Convert OpenAI-style tool result to Claude's tool_result block.
                tool_use_id = msg.get("tool_call_id") or msg.get("id")
                tool_result_block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "is_error": False,
                }
                if isinstance(content, str):
                    tool_result_block["content"] = [{"type": "text", "text": content}]
                elif isinstance(content, list):
                    tool_result_block["content"] = content
                else:
                    tool_result_block["content"] = [
                        {"type": "text", "text": json.dumps(content or "")}
                    ]

                converted.append(
                    {
                        "role": "user",
                        "content": [tool_result_block],
                    }
                )

        return converted

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI tool format to Claude tool format.

        OpenAI format:
        {
            "type": "function",
            "function": {"name": "...", "description": "...", "parameters": {...}}
        }

        Claude format:
        {
            "name": "...",
            "description": "...",
            "input_schema": {...}
        }
        """
        claude_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                if func.get("name"):
                    claude_tools.append(
                        {
                            "name": func.get("name"),
                            "description": func.get("description", ""),
                            "input_schema": func.get("parameters", {}),
                        }
                    )
        return claude_tools

    def _map_stop_reason(self, claude_stop_reason: str) -> str:
        """Map Claude's stop_reason to OpenAI's finish_reason."""
        mapping = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
        }
        return mapping.get(claude_stop_reason, "stop")

    def _extract_system(self, messages: list[dict[str, Any]]) -> str | None:
        """Extract system text from input messages if present.

        Concatenates all system message contents into a single string. Converts
        structured content to plain text conservatively by JSON-dumping.
        """
        parts: list[str] = []
        for msg in messages:
            if msg.get("role") != "system":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(str(block.get("text", "")))
                    else:
                        with suppress(Exception):
                            parts.append(json.dumps(block))
            elif content is not None:
                with suppress(Exception):
                    parts.append(json.dumps(content))
        joined = "\n\n".join([p for p in parts if p])
        return joined if joined else None

    def _summarize_messages(self, msgs: list[dict[str, Any]]) -> dict[str, Any]:
        """Summarize message roles and block types for safe logging."""
        summary: list[dict[str, Any]] = []
        for m in msgs:
            role = m.get("role")
            content = m.get("content")
            kinds: list[str] = []
            if isinstance(content, list):
                for b in content[:10]:  # cap for safety
                    if isinstance(b, dict):
                        kinds.append(str(b.get("type", "?")))
                    else:
                        kinds.append(type(b).__name__)
            else:
                kinds.append("text")
            summary.append({"role": role, "blocks": kinds[:10]})
        return {"count": len(msgs), "messages": summary}
