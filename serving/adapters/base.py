"""Base adapter interface and shared utilities for LLM providers."""

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from serving.http import AsyncHTTPClient
from serving.stream import make_stream_chunk


@dataclass
class UsageInfo:
    """Token usage statistics with cache and reasoning token support."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    # Cache tokens for cost calculation
    cache_read_tokens: int = 0  # Tokens read from cache (cheaper)
    cache_write_tokens: int = 0  # Tokens written to cache (may have cost)

    def to_dict(self) -> dict[str, int]:
        """Convert usage info to OpenAI-compatible dict format."""
        result = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
        # Include reasoning tokens if present (for models like DeepSeek-R1)
        if self.reasoning_tokens > 0:
            result["reasoning_tokens"] = self.reasoning_tokens
        # Include cache tokens if present (for transparency)
        if self.cache_read_tokens > 0:
            result["cache_read_tokens"] = self.cache_read_tokens
        if self.cache_write_tokens > 0:
            result["cache_write_tokens"] = self.cache_write_tokens
        return result


@dataclass
class ModelConfig:
    """Configuration for a model and its upstream provider."""

    id: str
    name: str
    provider: str
    base_url: str
    api_key: str | None = None
    # Public aliases that should also route to this adapter configuration.
    aliases: list[str] = field(default_factory=list)
    # Provider-specific model identifier to send to upstream. If not set,
    # `id` is used.
    provider_model_id: str | None = None
    quantization: str = "bf16"
    input_modalities: list[str] = field(default_factory=lambda: ["text"])
    output_modalities: list[str] = field(default_factory=lambda: ["text"])
    context_length: int = 8192
    max_output_length: int = 4096
    supports_tools: bool = False
    supports_structured_output: bool = False
    supported_params: list[str] = field(
        default_factory=lambda: ["temperature", "top_p", "max_tokens"]
    )
    pricing: dict[str, str] = field(
        default_factory=lambda: {
            "prompt": "0",
            "completion": "0",
            "image": "0",
            "request": "0",
            "input_cache_reads": "0",
            "input_cache_writes": "0",
        }
    )


class BaseAdapter(ABC):
    """Abstract base class for LLM provider adapters."""

    def __init__(self, config: ModelConfig):
        self.config = config
        # Legacy: some adapters still use self.session; keep for compatibility.
        self.session = None
        # Shared HTTP client for new/updated adapters.
        self.http = AsyncHTTPClient.shared()

    @abstractmethod
    async def chat_completion(self, messages: list[dict[str, Any]], **params) -> dict[str, Any]:
        """Execute non-streaming chat completion request."""
        pass

    @abstractmethod
    async def stream_chat_completion(
        self, messages: list[dict[str, Any]], **params
    ) -> AsyncGenerator[str, None]:
        """Execute streaming chat completion request."""
        pass

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate and clamp request parameters to provider limits."""
        validated = {}

        if "max_tokens" in params:
            validated["max_tokens"] = min(params["max_tokens"], self.config.max_output_length)

        if "temperature" in params:
            validated["temperature"] = max(0.0, min(2.0, params["temperature"]))

        if "top_p" in params:
            validated["top_p"] = max(0.0, min(1.0, params["top_p"]))

        if "stop" in params:
            validated["stop"] = params["stop"]

        if "seed" in params and "seed" in self.config.supported_params:
            validated["seed"] = params["seed"]

        return validated

    def format_response(
        self,
        content: str,
        model: str,
        usage: UsageInfo | None = None,
        tool_calls: list[dict] | None = None,
        finish_reason: str = "stop",
    ) -> dict[str, Any]:
        """Format provider response into OpenAI-compatible schema."""
        response = {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
        }

        if tool_calls:
            response["choices"][0]["message"]["tool_calls"] = tool_calls

        if usage:
            response["usage"] = usage.to_dict()

        return response

    def format_stream_chunk(
        self, content: str, model: str, finish_reason: str | None = None, role: str | None = None
    ) -> str:
        """Format streaming chunk into SSE format."""
        return make_stream_chunk(
            model=model, content=content, finish_reason=finish_reason, role=role
        )

    def format_tool_chunk(self, tool_calls: list[dict[str, Any]], model: str) -> str:
        """Format tool calls into OpenAI-compatible streaming chunk.

        Args:
            tool_calls: List of tool call deltas in OpenAI format
            model: Model identifier

        Returns:
            SSE-formatted chunk containing tool_calls in delta
        """
        import json

        chunk = {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": tool_calls},
                    "finish_reason": None,
                }
            ],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    async def cleanup(self):
        """Clean up adapter resources (override if needed)."""
        if self.session:
            await self.session.close()
