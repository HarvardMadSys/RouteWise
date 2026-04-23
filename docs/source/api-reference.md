# API Reference

Complete reference for the HybridInference API.

## Base URL

```
https://freeinference.org/v1
```

## Authentication

All API requests require authentication using an API key in the `Authorization` header:

```
Authorization: Bearer YOUR_API_KEY
```

## Endpoints

### Chat Completions

Create a chat completion using a specified model.

**Endpoint:** `POST /v1/chat/completions`

**Headers:**
```
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY
```

**Request Body:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model ID to use (e.g., `llama-3.3-70b-instruct`) |
| `messages` | array | Yes | Array of message objects |
| `temperature` | number | No | Sampling temperature (0-2). Default: 1 |
| `max_tokens` | integer | No | Maximum tokens to generate |
| `top_p` | number | No | Nucleus sampling parameter (0-1) |
| `stream` | boolean | No | Whether to stream responses. Default: false |
| `stop` | string or array | No | Stop sequences |
| `response_format` | object | No | Format of response (e.g., `{"type": "json_object"}`) |
| `tools` | array | No | Function calling tools |
| `tool_choice` | string or object | No | Tool choice strategy |

**Message Object:**

| Field | Type | Description |
|-------|------|-------------|
| `role` | string | Role: `system`, `user`, or `assistant` |
| `content` | string or array | Message content (text or multimodal) |

**Example Request:**

```json
{
  "model": "llama-3.3-70b-instruct",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "temperature": 0.7,
  "max_tokens": 1000
}
```

**Response:**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "llama-3.3-70b-instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 8,
    "total_tokens": 28
  }
}
```

**Streaming Response:**

When `stream: true`, responses are sent as Server-Sent Events (SSE). With curl, use `-N` (no-buffer) to see tokens as they arrive:

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"llama-3.3-70b-instruct","choices":[{"index":0,"delta":{"role":"assistant","content":"The"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"llama-3.3-70b-instruct","choices":[{"index":0,"delta":{"content":" capital"},"finish_reason":null}]}

data: [DONE]
```

---

### List Models

Get a list of available models.

**Endpoint:** `GET /v1/models`

**Headers:**
```
Authorization: Bearer YOUR_API_KEY
```

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "llama-3.3-70b-instruct",
      "object": "model",
      "created": 1234567890,
      "owned_by": "system",
      "context_length": 131072,
      "architecture": {
        "modality": "text",
        "tokenizer": "llama3",
        "instruct_type": "llama3"
      },
      "pricing": {
        "prompt": "0",
        "completion": "0",
        "request": "0"
      }
    }
  ]
}
```

---

### Health Check

Check API health status.

**Endpoint:** `GET /health`

**Response:**

```json
{
  "status": "ok",
  "timestamp": "2025-10-26T08:00:00Z"
}
```

---

## Parameters Reference

### Temperature

Controls randomness in responses.

- **Range:** 0.0 - 2.0
- **Default:** 1.0
- **Lower values:** More focused and deterministic
- **Higher values:** More creative and diverse

**Examples:**
- `0.0` - Deterministic (good for factual tasks)
- `0.7` - Balanced (general use)
- `1.5` - Creative (storytelling, brainstorming)

### Max Tokens

Maximum number of tokens to generate.

- **Default:** Model-specific
- **Note:** Input + output tokens cannot exceed model's context length

### Top P (Nucleus Sampling)

Alternative to temperature for controlling diversity.

- **Range:** 0.0 - 1.0
- **Default:** 1.0
- **Lower values:** More focused
- **Higher values:** More diverse

**Note:** It's recommended to use either `temperature` or `top_p`, not both.

### Stop Sequences

Sequences where the model will stop generating.

**Examples:**
```json
{
  "stop": "\n"  // Single stop sequence
}
```

```json
{
  "stop": ["\n", "###", "END"]  // Multiple stop sequences
}
```

---

## Response Formats

### Standard Text Response

Default response format.

```json
{
  "response_format": {"type": "text"}
}
```

### JSON Mode

Forces the model to respond with valid JSON.

```json
{
  "response_format": {"type": "json_object"}
}
```

**Example:**
```json
{
  "model": "llama-3.3-70b-instruct",
  "messages": [
    {
      "role": "user",
      "content": "Extract person info: John is a 30-year-old engineer. Return as JSON."
    }
  ],
  "response_format": {"type": "json_object"}
}
```

**Response:**
```json
{
  "name": "John",
  "age": 30,
  "occupation": "engineer"
}
```

---

## Function Calling

Enable the model to call functions you define.

### Tool Definition

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "City name"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"]
            }
          },
          "required": ["location"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

### Tool Choice Options

- `"auto"` - Model decides whether to call a function
- `"none"` - Model will not call any function
- `{"type": "function", "function": {"name": "function_name"}}` - Force specific function

### Response with Function Call

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"location\": \"Paris\", \"unit\": \"celsius\"}"
            }
          }
        ]
      }
    }
  ]
}
```

---

## Vision (Multimodal)

Send images along with text (requires vision-capable models like `llama-4-maverick`).

### Image URL

```json
{
  "model": "llama-4-maverick",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What's in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "https://example.com/image.jpg"
          }
        }
      ]
    }
  ]
}
```

### Base64 Image

```json
{
  "type": "image_url",
  "image_url": {
    "url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
  }
}
```

---

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Invalid or missing API key |
| 404 | Not Found - Model or endpoint not found |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error - Server error |
| 503 | Service Unavailable - Server overloaded |

### Error Response Format

```json
{
  "error": {
    "message": "Invalid API key provided",
    "type": "invalid_request_error",
    "code": "invalid_api_key"
  }
}
```

---

## Rate Limits

Current rate limits (subject to change):

- **Requests per minute:** Based on your API key tier
- **Tokens per minute:** Based on your API key tier

Rate limit headers are included in responses:

```
X-RateLimit-Limit-Requests: 100
X-RateLimit-Remaining-Requests: 99
X-RateLimit-Reset-Requests: 2025-10-26T08:01:00Z
```

---

## OpenRouter Compatibility

This API is fully compatible with [OpenRouter](https://openrouter.ai/) clients and libraries. Simply change the base URL to:

```python
client = openai.OpenAI(
    base_url="https://freeinference.org/v1",
    api_key="your-api-key-here"
)
```

All OpenRouter-compatible parameters and features are supported.
