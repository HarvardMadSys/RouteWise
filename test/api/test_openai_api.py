"""OpenAI API integration test for XHS internal runway API.

This test verifies that the OpenAI adapter correctly integrates with the
internal XHS OpenAI-compatible API which has specific requirements:
- Uses 'api-key' header instead of 'Authorization: Bearer'
- Requires 'api-version' query parameter
- Does not require 'model' parameter in request body
"""

import json
import os

import httpx
import pytest

pytestmark = pytest.mark.external

# Test configuration
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_VERSION = "2024-12-01-preview"

if not OPENAI_API_KEY:
    pytest.skip("OPENAI_API_KEY not configured", allow_module_level=True)


def test_non_streaming_completion():
    """Test basic non-streaming chat completion."""
    print("\n=== Testing Non-Streaming Completion ===")

    url = f"{OPENAI_BASE_URL}/chat/completions?api-version={API_VERSION}"
    headers = {
        "api-key": OPENAI_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello, World!' and nothing else."},
        ],
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    content = data["choices"][0]["message"]["content"]
    print(f"Response: {content}")
    print(f"Usage: {data.get('usage', 'N/A')}")
    print(f"Finish reason: {data['choices'][0]['finish_reason']}")

    assert data["choices"][0]["message"]["role"] == "assistant"
    assert content is not None
    assert len(content) > 0
    print("Non-streaming test passed!\n")


def test_streaming_completion():
    """Test streaming chat completion."""
    print("\n=== Testing Streaming Completion ===")

    url = f"{OPENAI_BASE_URL}/chat/completions?api-version={API_VERSION}"
    headers = {
        "api-key": OPENAI_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Count from 1 to 5."},
        ],
        "stream": True,
        "stream_options": {"include_usage": True},  # Required for accurate billing
    }

    chunks = []
    usage_data = None

    with (
        httpx.Client(timeout=30.0) as client,
        client.stream("POST", url, headers=headers, json=payload) as response,
    ):
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.strip():
                continue
            if line.startswith("data: "):
                data_str = line[6:]  # Remove "data: " prefix
                if data_str == "[DONE]":
                    break
                try:
                    chunk_data = json.loads(data_str)
                    # Capture usage from final chunk
                    if "usage" in chunk_data:
                        usage_data = chunk_data["usage"]
                    # Extract content delta
                    if chunk_data.get("choices"):
                        delta = chunk_data["choices"][0].get("delta", {})
                        if "content" in delta:
                            content = delta["content"]
                            chunks.append(content)
                            print(content, end="", flush=True)
                except json.JSONDecodeError:
                    continue

    full_response = "".join(chunks)
    print(f"\n\n Received {len(chunks)} chunks")
    print(f"Full response: {full_response}")
    if usage_data:
        print(f"Usage: {usage_data}")

    assert len(chunks) > 0
    assert len(full_response) > 0
    print("Streaming test passed!\n")


def test_with_vision():
    """Test vision capability with image URL."""
    print("\n=== Testing Vision Capability ===")

    url = f"{OPENAI_BASE_URL}/chat/completions?api-version={API_VERSION}"
    headers = {
        "api-key": OPENAI_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"
                        },
                    },
                ],
            }
        ],
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    content = data["choices"][0]["message"]["content"]
    print(f"Response: {content}")
    print(f"Usage: {data.get('usage', 'N/A')}")

    assert content is not None
    assert len(content) > 0
    print("Vision test passed!\n")


def test_high_detail_image():
    """Test high-detail image processing."""
    print("\n=== Testing High-Detail Image Processing ===")

    url = f"{OPENAI_BASE_URL}/chat/completions?api-version={API_VERSION}"
    headers = {
        "api-key": OPENAI_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    content = data["choices"][0]["message"]["content"]
    print(f"Response: {content[:200]}...")  # Print first 200 chars
    print(f"Usage: {data.get('usage', 'N/A')}")

    assert content is not None
    assert len(content) > 0
    print("High-detail image test passed!\n")


if __name__ == "__main__":
    print("Starting OpenAI API Integration Tests (XHS Runway)")
    print(f"Base URL: {OPENAI_BASE_URL}")
    print(f"API Key: {'*' * 8}{OPENAI_API_KEY[-4:] if OPENAI_API_KEY else 'NOT SET'}")
    print(f"API Version: {API_VERSION}")

    try:
        test_non_streaming_completion()
        test_streaming_completion()
        test_with_vision()
        test_high_detail_image()

        print("\n" + "=" * 50)
        print("All tests passed successfully!")
        print("=" * 50)
    except Exception as e:
        print(f"\n Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
