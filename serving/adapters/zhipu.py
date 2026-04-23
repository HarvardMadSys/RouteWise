"""Zhipu AI (GLM) adapter for OpenAI-compatible interface.

This adapter talks to Zhipu's OpenAI-compatible endpoints without
injecting non-standard roles (e.g., "developer"). It mirrors the
behavior of other adapters in this codebase, using the shared HTTP
client, standardized response formatting, and OpenAI-style SSE chunks.
"""

import json
from collections.abc import AsyncGenerator
from typing import Any

from serving.stream import done_sentinel, make_final_usage_chunk
from serving.utils.tokens import estimate_prompt_tokens, estimate_text_tokens

from .base import BaseAdapter, UsageInfo


def _clean_message(message: dict[str, Any]) -> dict[str, Any]:
    """Remove None values from message dict to ensure API compatibility.

    Some APIs reject messages with null/None fields,
    so we strip them before sending.
    """
    return {k: v for k, v in message.items() if v is not None}


class ZhipuAdapter(BaseAdapter):  # type: ignore[no-any-unimported]
    """Adapter for Zhipu AI GLM models using OpenAI-compatible endpoints."""

    async def chat_completion(
        self, messages: list[dict[str, Any]], **params: Any
    ) -> dict[str, Any]:
        """Send chat completion request to Zhipu API."""
        validated_params = self.validate_params(params)

        # Use provider_model_id if specified, otherwise fall back to id
        model_id = self.config.provider_model_id or self.config.id

        # Clean messages to remove None fields (some APIs reject them)
        cleaned_messages = [_clean_message(msg) for msg in messages]

        # Zhipu uses standard OpenAI format without special roles
        payload = {"model": model_id, "messages": cleaned_messages, **validated_params}

        # Additional parameters if supported
        if "tools" in params and self.config.supports_tools:
            payload["tools"] = params["tools"]
            if "tool_choice" in params:
                payload["tool_choice"] = params["tool_choice"]

        if "response_format" in params and self.config.supports_structured_output:
            payload["response_format"] = params["response_format"]

        if "top_k" in params and "top_k" in self.config.supported_params:
            payload["top_k"] = params["top_k"]

        if "frequency_penalty" in params and "frequency_penalty" in self.config.supported_params:
            payload["frequency_penalty"] = params["frequency_penalty"]

        if "presence_penalty" in params and "presence_penalty" in self.config.supported_params:
            payload["presence_penalty"] = params["presence_penalty"]

        # GLM-4.6: optional tool streaming and thinking parameters
        if "tool_stream" in params and "tool_stream" in self.config.supported_params:
            payload["tool_stream"] = params["tool_stream"]
        if "thinking" in params and "thinking" in self.config.supported_params:
            payload["thinking"] = params["thinking"]

        endpoint = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Use existing HTTP client
        data = await self.http.json_post_with_retry(
            endpoint,
            json=payload,
            headers=headers,
            timeout=None,
            retries=3,
        )

        # Extract response components
        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content", "")
            tool_calls = message.get("tool_calls")
            finish_reason = choices[0].get("finish_reason", "stop")
        else:
            content = ""
            tool_calls = None
            finish_reason = "stop"

        # Get or estimate usage
        usage_payload = data.get("usage") or {}
        if usage_payload:
            usage = UsageInfo(
                prompt_tokens=usage_payload.get("prompt_tokens", 0),
                completion_tokens=usage_payload.get("completion_tokens", 0),
                total_tokens=usage_payload.get("total_tokens", 0),
            )
        else:
            prompt_tokens = estimate_prompt_tokens(messages)
            completion_tokens = estimate_text_tokens(content)
            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

        # Format standardized response
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
        """Stream chat completion responses from Zhipu API."""
        validated_params = self.validate_params(params)

        # Use provider_model_id if specified
        model_id = self.config.provider_model_id or self.config.id

        # Clean messages to remove None fields (some APIs reject them)
        cleaned_messages = [_clean_message(msg) for msg in messages]

        # Build payload
        payload = {
            "model": model_id,
            "messages": cleaned_messages,
            "stream": True,
            **validated_params,
        }

        # Additional parameters if supported
        if "tools" in params and self.config.supports_tools:
            payload["tools"] = params["tools"]
            if "tool_choice" in params:
                payload["tool_choice"] = params["tool_choice"]

        if "response_format" in params and self.config.supports_structured_output:
            payload["response_format"] = params["response_format"]

        if "top_k" in params and "top_k" in self.config.supported_params:
            payload["top_k"] = params["top_k"]

        if "frequency_penalty" in params and "frequency_penalty" in self.config.supported_params:
            payload["frequency_penalty"] = params["frequency_penalty"]

        if "presence_penalty" in params and "presence_penalty" in self.config.supported_params:
            payload["presence_penalty"] = params["presence_penalty"]

        # GLM-4.6: optional tool streaming and thinking parameters (streaming)
        if "tool_stream" in params and "tool_stream" in self.config.supported_params:
            payload["tool_stream"] = params["tool_stream"]
        if "thinking" in params and "thinking" in self.config.supported_params:
            payload["thinking"] = params["thinking"]

        endpoint = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        from serving.utils.logging import get_logger

        logger = get_logger(__name__)
        logger.debug(f"Starting stream to: {endpoint}")
        logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

        total_content = ""
        finish_reason = "stop"
        # Note: tool_calls are forwarded inline; no accumulator needed here.
        line_count = 0

        # Use existing HTTP client stream_post (returns SSE lines)
        async for line in self.http.stream_post(endpoint, json=payload, headers=headers):
            line_count += 1
            # Lines are already in "data: ..." format from stream_post
            if not line.startswith("data: "):
                if line_count <= 5:
                    logger.debug(f"Skipping non-data line at {line_count}: {line[:100]}")
                continue

            # Check for stream end
            if line == "data: [DONE]":
                logger.debug(f"Received [DONE] at line {line_count}")
                # Send final usage chunk
                yield make_final_usage_chunk(
                    model=self.config.id,
                    messages=messages,
                    total_content=total_content,
                    finish_reason=finish_reason,
                    provider=self.config.provider,
                    base_url=self.config.base_url,
                )
                yield done_sentinel()
                break

            # Parse the JSON chunk
            try:
                chunk_data = json.loads(line[6:])
                if line_count <= 10:
                    logger.debug(f"Chunk {line_count}: {json.dumps(chunk_data)[:300]}")

                choices = chunk_data.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}

                    if line_count <= 10:
                        logger.debug(f"Delta at line {line_count}: {delta}")

                    # Handle content
                    content = delta.get("content")
                    if content:
                        total_content += content
                        # Use format_stream_chunk for consistency
                        if line_count <= 5:
                            logger.debug(f"Yielding content at line {line_count}: {content[:100]}")
                        yield self.format_stream_chunk(content, self.config.id)

                    # Handle tool_calls - forward the entire chunk to preserve streaming format
                    tool_calls_delta = delta.get("tool_calls")
                    if tool_calls_delta:
                        logger.debug(
                            f"Tool calls delta at line {line_count}: {json.dumps(tool_calls_delta)[:300]}"
                        )

                        # Pass through the chunk but replace model ID with our logical model ID
                        # This preserves the OpenAI streaming format with incremental tool_calls deltas
                        chunk_copy = chunk_data.copy()
                        chunk_copy["model"] = self.config.id
                        tool_chunk = f"data: {json.dumps(chunk_copy)}\n\n"
                        logger.debug(
                            f"Yielding tool chunk at line {line_count}: {tool_chunk[:200]}"
                        )
                        yield tool_chunk

                    fr = choices[0].get("finish_reason")
                    if fr:
                        finish_reason = fr
                        logger.debug(f"Finish reason: {fr}")
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error at line {line_count}: {e}, line: {line[:100]}")
                continue

        logger.info(f"Stream complete: lines={line_count}, content_chars={len(total_content)}")
