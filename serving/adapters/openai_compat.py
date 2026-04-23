"""Generic adapter for OpenAI-compatible APIs (gateways, local VLLM, etc.)."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from serving.stream import done_sentinel
from serving.utils.logging import get_logger
from serving.utils.tokens import estimate_prompt_tokens, estimate_text_tokens

from .base import BaseAdapter, UsageInfo

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = get_logger(__name__)


def _clean_message(message: dict[str, Any]) -> dict[str, Any]:
    """Remove None values from message dict to ensure API compatibility.

    Some APIs (like Featherless) reject messages with null/None fields,
    so we strip them before sending.
    """
    return {k: v for k, v in message.items() if v is not None}


class OpenAICompatAdapter(BaseAdapter):
    """Generic adapter for OpenAI-compatible APIs.

    Works with:
    - API gateways (Chutes, Featherless, OpenRouter)
    - Local deployments (VLLM, sglang)
    - Any OpenAI-compatible service

    Configuration is fully driven by ModelConfig fields.
    Optional overrides are applied when present on the config:
    - chat_path (default: /v1/chat/completions)
    - auth_header_name (default: Authorization)
    - auth_format (default: Bearer {api_key})
    - extra_headers / extra_query (dict[str,str])
    """

    def __init__(self, config):
        super().__init__(config)

        logger.info(f"[OpenAICompat] Initialized for {config.id} at {config.base_url}")

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for request."""
        headers = {"Content-Type": "application/json"}

        # Add standard OpenAI authentication
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        # Custom header formatting (if provided)
        auth_name = getattr(self.config, "auth_header_name", None)
        auth_format = getattr(self.config, "auth_format", None)
        if auth_name and auth_format and self.config.api_key:
            headers[auth_name] = auth_format.format(api_key=self.config.api_key)

        # Merge any extra headers
        extra_headers = getattr(self.config, "extra_headers", None)
        if isinstance(extra_headers, dict):
            headers.update(extra_headers)

        return headers

    def _build_url(self) -> str:
        """Build full endpoint URL (standard OpenAI path)."""
        base = (self.config.base_url or "").rstrip("/")

        # If base_url already includes /v1, just append /chat/completions
        if base.endswith("/v1"):
            return f"{base}/chat/completions"

        # Otherwise, use the full path
        chat_path = getattr(self.config, "chat_path", None) or "/v1/chat/completions"
        if not chat_path.startswith("/"):
            chat_path = f"/{chat_path}"
        return f"{base}{chat_path}"

    def _get_model_identifier(self) -> str:
        """Get model ID to send to upstream API."""
        return self.config.provider_model_id or self.config.id

    async def chat_completion(
        self, messages: list[dict[str, Any]], **params: Any
    ) -> dict[str, Any]:
        """Execute non-streaming chat completion.

        Args:
            messages: Chat messages in OpenAI format
            **params: Optional parameters (temperature, max_tokens, tools, etc.)

        Returns:
            OpenAI-compatible response dict
        """
        validated = self.validate_params(params)

        # Clean messages to remove None fields (some APIs reject them)
        cleaned_messages = [_clean_message(msg) for msg in messages]

        # Build request payload
        payload = {
            "model": self._get_model_identifier(),
            "messages": cleaned_messages,
            **validated,
        }

        # Add optional features
        if params.get("tools") and self.config.supports_tools:
            payload["tools"] = params["tools"]
            if params.get("tool_choice") is not None:
                payload["tool_choice"] = params["tool_choice"]

        if params.get("response_format") and self.config.supports_structured_output:
            payload["response_format"] = params["response_format"]

        # Make request
        url = self._build_url()
        headers = self._build_headers()

        logger.debug(f"[OpenAICompat] POST {url} model={payload['model']}")
        logger.debug(f"[OpenAICompat] Payload: {payload}")
        logger.debug(f"[OpenAICompat] Headers: {headers}")

        response = await self.http.json_post_with_retry(
            url=url,
            json=payload,
            headers=headers,
            timeout=120,
            retries=2,
        )

        # Parse response
        return self._parse_completion_response(response)

    async def stream_chat_completion(
        self, messages: list[dict[str, Any]], **params: Any
    ) -> AsyncGenerator[str, None]:
        """Execute streaming chat completion.

        Args:
            messages: Chat messages in OpenAI format
            **params: Optional parameters

        Yields:
            SSE-formatted chunks
        """
        validated = self.validate_params(params)

        # Clean messages to remove None fields (some APIs reject them)
        cleaned_messages = [_clean_message(msg) for msg in messages]

        payload = {
            "model": self._get_model_identifier(),
            "messages": cleaned_messages,
            "stream": True,
            **validated,
        }

        # Add optional features
        if params.get("tools") and self.config.supports_tools:
            payload["tools"] = params["tools"]
            if params.get("tool_choice") is not None:
                payload["tool_choice"] = params["tool_choice"]

        if params.get("response_format") and self.config.supports_structured_output:
            payload["response_format"] = params["response_format"]
            # vLLM-specific: guided_json parameter for structured output
            if schema := params["response_format"].get("schema"):
                payload["guided_json"] = schema

        url = self._build_url()
        headers = self._build_headers()

        total_content = ""
        finish_reason = "stop"
        upstream_usage: dict[str, Any] | None = None
        prompt_tokens_override: int | None = None

        # Stream response
        async for chunk in self.http.stream_post(url=url, json=payload, headers=headers):
            if not chunk.strip():
                continue

            if chunk.startswith("data: "):
                data_str = chunk[6:]

                if data_str.strip() == "[DONE]":
                    final_usage = upstream_usage or self._build_fallback_usage(
                        messages=messages,
                        total_content=total_content,
                        prompt_tokens_override=prompt_tokens_override,
                    )
                    final_chunk = self._build_final_chunk(
                        usage=final_usage,
                        finish_reason=finish_reason,
                    )
                    yield final_chunk
                    yield done_sentinel()
                    break

                try:
                    data = json.loads(data_str)
                    choices = data.get("choices") or []

                    # Capture usage payload when provided (usually final chunk before [DONE])
                    if data.get("usage"):
                        upstream_usage = data["usage"]
                        pt = upstream_usage.get("prompt_tokens")
                        if isinstance(pt, int):
                            prompt_tokens_override = pt

                    if not choices:
                        # Nothing to emit (likely a usage-only frame); continue consuming.
                        continue

                    delta = choices[0].get("delta", {})

                    # Handle content chunks
                    if "content" in delta:
                        content = delta["content"]
                        if isinstance(content, str):
                            total_content += content
                            if content:
                                yield self.format_stream_chunk(
                                    content=content,
                                    model=self.config.id,
                                )

                    # Handle tool calls
                    if "tool_calls" in delta and isinstance(delta.get("tool_calls"), list):
                        yield self.format_tool_chunk(
                            tool_calls=delta["tool_calls"],
                            model=self.config.id,
                        )

                    # Track finish reason
                    fr = data["choices"][0].get("finish_reason")
                    if fr:
                        finish_reason = fr

                except json.JSONDecodeError:
                    logger.warning(f"[OpenAICompat] Failed to parse chunk: {data_str[:100]}")

    def _parse_completion_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Parse response into OpenAI-compatible format."""
        choice = response["choices"][0]
        message = choice["message"]

        usage = self._parse_usage(response.get("usage", {}))

        return self.format_response(
            content=message.get("content", ""),
            model=self.config.id,
            usage=usage,
            tool_calls=message.get("tool_calls"),
            finish_reason=choice.get("finish_reason", "stop"),
        )

    def _parse_usage(self, usage_data: dict[str, Any]) -> UsageInfo:
        """Parse usage information from response."""
        return UsageInfo(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
            reasoning_tokens=usage_data.get("reasoning_tokens", 0),
            cache_read_tokens=usage_data.get("cache_read_input_tokens", 0)
            or usage_data.get("cache_read_tokens", 0),
            cache_write_tokens=usage_data.get("cache_creation_input_tokens", 0)
            or usage_data.get("cache_write_tokens", 0),
        )

    def _build_fallback_usage(
        self,
        *,
        messages: list[dict[str, Any]],
        total_content: str,
        prompt_tokens_override: int | None,
    ) -> dict[str, int]:
        prompt_tokens = (
            int(prompt_tokens_override)
            if prompt_tokens_override is not None and prompt_tokens_override > 0
            else int(estimate_prompt_tokens(messages))
        )
        completion_tokens = int(estimate_text_tokens(total_content))
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _build_final_chunk(
        self,
        *,
        usage: dict[str, Any],
        finish_reason: str,
    ) -> str:
        chunk = {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.config.id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            "usage": usage,
            "_routing": {
                "provider": self.config.provider,
                "base_url": self.config.base_url,
            },
        }
        return f"data: {json.dumps(chunk)}\n\n"
