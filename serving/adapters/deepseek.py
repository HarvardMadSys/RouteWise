"""DeepSeek API adapter with cache token tracking."""

import json
from collections.abc import AsyncGenerator
from typing import Any

from serving.stream import done_sentinel, make_final_usage_chunk
from serving.utils.tokens import estimate_prompt_tokens, estimate_text_tokens

from .base import BaseAdapter, UsageInfo


class DeepSeekAdapter(BaseAdapter):
    """Adapter for DeepSeek R1 and compatible models."""

    async def chat_completion(self, messages: list[dict[str, Any]], **params) -> dict[str, Any]:
        """Execute non-streaming chat completion with cache token extraction."""
        validated_params = self.validate_params(params)

        payload = {"model": "deepseek-chat", "messages": messages, **validated_params}

        if params.get("tools"):
            payload["tools"] = params["tools"]
            if params.get("tool_choice"):
                payload["tool_choice"] = params["tool_choice"]

        if params.get("response_format") and params["response_format"].get("type") == "json_object":
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

        data = await self.http.json_post_with_retry(
            f"{self.config.base_url}/chat/completions",
            json=payload,
            headers=headers,
        )

        # Extract cache information from DeepSeek response
        usage_data = data.get("usage", {})
        cache_hit_tokens = usage_data.get("prompt_cache_hit_tokens", 0)
        cache_miss_tokens = usage_data.get("prompt_cache_miss_tokens", 0)

        # DeepSeek returns prompt_tokens = cache_hit + cache_miss
        # For cost calculation, we need to separate them
        prompt_tokens = usage_data.get("prompt_tokens", 0)

        # If cache fields are present, use them; otherwise all tokens are "miss" (full price)
        if cache_hit_tokens > 0 or cache_miss_tokens > 0:
            # Cache info available
            actual_cache_read = cache_hit_tokens
            actual_prompt_tokens = cache_miss_tokens  # Only non-cached tokens count as "prompt"
        else:
            # No cache info, all tokens charged at regular prompt price
            actual_cache_read = 0
            actual_prompt_tokens = prompt_tokens

        usage = UsageInfo(
            prompt_tokens=actual_prompt_tokens,  # Non-cached prompt tokens
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
            reasoning_tokens=usage_data.get("reasoning_tokens", 0),
            cache_read_tokens=actual_cache_read,  # Cache hit tokens
        )
        if usage.total_tokens == 0:
            # Fallback estimate when provider omits usage
            content = data["choices"][0]["message"].get("content", "")
            prompt_tokens = estimate_prompt_tokens(messages)
            completion_tokens = estimate_text_tokens(content)
            usage = UsageInfo(
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                total_tokens=int(prompt_tokens + completion_tokens),
            )

        tool_calls = None
        if "tool_calls" in data["choices"][0]["message"]:
            tool_calls = data["choices"][0]["message"]["tool_calls"]

        return self.format_response(
            content=data["choices"][0]["message"]["content"],
            model=self.config.id,
            usage=usage,
            tool_calls=tool_calls,
            finish_reason=data["choices"][0]["finish_reason"],
        )

    async def stream_chat_completion(
        self, messages: list[dict[str, Any]], **params
    ) -> AsyncGenerator[str, None]:
        """Execute streaming chat completion with cache token extraction."""
        validated_params = self.validate_params(params)

        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "stream": True,
            **validated_params,
        }

        if params.get("tools"):
            payload["tools"] = params["tools"]
            if params.get("tool_choice"):
                payload["tool_choice"] = params["tool_choice"]

        if params.get("response_format") and params["response_format"].get("type") == "json_object":
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

        total_content = ""

        # Track all usage-related tokens from the final chunk
        prompt_tokens = 0
        completion_tokens = 0
        cache_hit_tokens = 0
        cache_miss_tokens = 0
        reasoning_tokens = 0

        async for line in self.http.stream_post(
            f"{self.config.base_url}/chat/completions",
            json=payload,
            headers=headers,
        ):
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                # Calculate actual prompt tokens (non-cached only)
                # DeepSeek returns prompt_tokens = cache_hit + cache_miss
                if cache_hit_tokens > 0 or cache_miss_tokens > 0:
                    # Cache info available, use cache_miss for billing
                    actual_prompt_tokens = cache_miss_tokens
                    actual_cache_read = cache_hit_tokens
                else:
                    # No cache info, all tokens charged at regular price
                    actual_prompt_tokens = prompt_tokens
                    actual_cache_read = 0

                yield make_final_usage_chunk(
                    model=self.config.id,
                    messages=messages,
                    total_content=total_content,
                    prompt_tokens_override=actual_prompt_tokens or None,
                    completion_tokens_override=completion_tokens or None,
                    finish_reason="stop",
                    provider=self.config.provider,
                    base_url=self.config.base_url,
                    reasoning_tokens=reasoning_tokens,
                    cache_read_tokens=actual_cache_read,
                )
                yield done_sentinel()
                break

            try:
                chunk_data = json.loads(line[6:])

                # Extract usage information from chunks (usually in the last chunk)
                if chunk_data.get("usage"):
                    usage_data = chunk_data["usage"]
                    prompt_tokens = usage_data.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage_data.get("completion_tokens", completion_tokens)

                    # Extract cache-related tokens
                    cache_hit_tokens = usage_data.get("prompt_cache_hit_tokens", cache_hit_tokens)
                    cache_miss_tokens = usage_data.get(
                        "prompt_cache_miss_tokens", cache_miss_tokens
                    )

                    # Extract reasoning tokens (for DeepSeek-R1)
                    reasoning_tokens = usage_data.get("reasoning_tokens", reasoning_tokens)

                # Extract content from delta
                if chunk_data.get("choices") and len(chunk_data["choices"]) > 0:
                    delta = chunk_data["choices"][0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        total_content += content
                        yield self.format_stream_chunk(content, self.config.id)
            except json.JSONDecodeError:
                continue
