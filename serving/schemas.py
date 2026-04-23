"""Pydantic schemas for API requests and responses.

These types model the OpenAI-compatible chat completion APIs and the
models listing endpoint. We keep them permissive enough to interoperate
with upstream providers while validating required fields and ranges.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):  # type: ignore[no-any-unimported]
    """Single chat message with role and content.

    Supports tool role for tool execution results in function calling flows.
    """

    role: Literal["system", "user", "assistant", "tool"]
    # Allow either plain string (OpenAI style) or structured blocks (provider-specific)
    content: Any | None = None
    # For role="tool" messages, associates result back to a prior tool call
    tool_call_id: str | None = None
    # Optional tool name for role="tool" messages
    name: str | None = None
    # For assistant messages carrying tool calls in OpenAI format
    tool_calls: list[dict[str, Any]] | None = None

    # Ignore unknown extra fields to be permissive with client payloads
    model_config = ConfigDict(extra="ignore")


class ResponseFormat(BaseModel):  # type: ignore[no-any-unimported]
    """Optional structured output hints for providers."""

    type: str | None = None
    # Some providers carry a JSON schema for guided decoding
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")


class ChatCompletionRequest(BaseModel):  # type: ignore[no-any-unimported]
    """OpenAI-compatible chat completions request payload."""

    model: str
    messages: list[ChatMessage]
    stream: bool | None = False

    # Sampling / decoding params
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)

    # Limits and stopping
    max_tokens: int | None = Field(default=None, ge=1)
    stop: str | list[str] | None = None
    seed: int | None = None

    # Penalties
    frequency_penalty: float | None = None
    presence_penalty: float | None = None

    # Tools / structured output
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: ResponseFormat | None = None

    # Pydantic v2 configuration: ignore extra fields in requests
    model_config = ConfigDict(extra="ignore")


# Response models


class ChoiceMessage(BaseModel):  # type: ignore[no-any-unimported]
    """Assistant message in a completion choice."""

    role: Literal["assistant"] = "assistant"
    content: str | None = ""
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionChoice(BaseModel):  # type: ignore[no-any-unimported]
    """One choice in the chat completion result set."""

    index: int
    message: ChoiceMessage
    finish_reason: str | None = None


class Usage(BaseModel):  # type: ignore[no-any-unimported]
    """Token usage accounting for the request/response.

    Extended to include cache and reasoning tokens for accurate cost tracking.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    # Optional fields for advanced billing
    reasoning_tokens: int | None = None  # For models like DeepSeek-R1
    cache_read_tokens: int | None = None  # Tokens read from cache (cheaper)
    cache_write_tokens: int | None = None  # Tokens written to cache


class ChatCompletionResponse(BaseModel):  # type: ignore[no-any-unimported]
    """OpenAI-compatible chat completions response payload."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage | None = None


class ModelItem(BaseModel):  # type: ignore[no-any-unimported]
    """Model metadata for listing endpoints."""

    id: str
    name: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str
    input_modalities: list[str]
    output_modalities: list[str]
    quantization: str
    context_length: int
    max_output_length: int
    pricing: dict[str, str]
    supported_sampling_parameters: list[str] = []
    supported_features: list[str] = []
    openrouter: dict[str, Any] | None = None


class ModelList(BaseModel):  # type: ignore[no-any-unimported]
    """List of models supported by the server."""

    object: Literal["list"] = "list"
    data: list[ModelItem]


# Error schemas for documenting non-2xx responses
class ErrorDetail(BaseModel):  # type: ignore[no-any-unimported]
    """Error detail payload aligned with OpenAI error shape."""

    type: str | None = None
    message: str
    code: int | None = None
    # Optional rate limit / routing metadata
    model: str | None = None
    retry_after: int | None = None
    tokens_requested: int | None = None
    queue_size: int | None = None


class ErrorResponse(BaseModel):  # type: ignore[no-any-unimported]
    """Top-level error wrapper."""

    error: ErrorDetail


# Backward-compatibility aliases for older tests referring to Choice
# New code should import ChatCompletionChoice explicitly.
Choice = ChatCompletionChoice
