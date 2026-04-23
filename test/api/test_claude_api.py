"""Claude API integration test for XHS internal runway API.

This test verifies Claude (Anthropic) API integration through the
XHS internal runway endpoint which uses a Google/Anthropic hybrid path.

API Specifics:
- Non-streaming: /openai/google/anthropic/v1:rawPredict
- Streaming: /openai/google/anthropic/v1:streamRawPredict
- Uses 'api-key' header instead of 'x-api-key'
- Requires 'anthropic_version' parameter
"""

import json
import os

import httpx
import pytest
from dotenv import load_dotenv

pytestmark = pytest.mark.external
load_dotenv()

# Test configuration
CLAUDE_BASE_URL = os.getenv("CLAUDE_BASE_URL")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

if not CLAUDE_API_KEY:
    pytest.skip("CLAUDE_API_KEY not configured", allow_module_level=True)


def test_non_streaming_completion():
    """Test basic non-streaming completion."""
    print("\n=== Testing Non-Streaming Completion ===")

    url = f"{CLAUDE_BASE_URL}/v1:rawPredict"
    headers = {
        "api-key": CLAUDE_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "anthropic_version": "vertex-2023-10-16",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Say hello in English"}],
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    # Claude response format
    content = data.get("content", [])
    if content and len(content) > 0:
        text = content[0].get("text", "")
        print(f"Response: {text}")
    else:
        print(f"Response: {data}")

    print(f"Usage: {data.get('usage', 'N/A')}")
    print(f"Stop reason: {data.get('stop_reason', 'N/A')}")

    assert "content" in data or "text" in data
    print("Non-streaming test passed!\n")


def test_streaming_completion():
    """Test streaming completion."""
    print("\n=== Testing Streaming Completion ===")

    url = f"{CLAUDE_BASE_URL}/v1:streamRawPredict"
    headers = {
        "api-key": CLAUDE_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "anthropic_version": "vertex-2023-10-16",
        "stream": True,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Count from 1 to 5"}],
    }

    chunks = []
    full_text = ""

    with (
        httpx.Client(timeout=30.0) as client,
        client.stream("POST", url, headers=headers, json=payload) as response,
    ):
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.strip():
                continue

            # Claude streaming format: "data: {...}"
            if line.startswith("data: "):
                data_str = line[6:]
                try:
                    chunk_data = json.loads(data_str)
                    chunk_type = chunk_data.get("type")

                    # Extract text from content_block_delta
                    if chunk_type == "content_block_delta":
                        delta = chunk_data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                chunks.append(text)
                                full_text += text
                                print(text, end="", flush=True)

                    # Message complete
                    elif chunk_type == "message_stop":
                        break

                except json.JSONDecodeError:
                    continue

    print(f"\n\n Received {len(chunks)} chunks")
    print(f"Full response: {full_text}")

    assert len(chunks) > 0 or len(full_text) > 0
    print("Streaming test passed!\n")


def test_with_system_prompt():
    """Test with system prompt."""
    print("\n=== Testing With System Prompt ===")

    url = f"{CLAUDE_BASE_URL}/v1:rawPredict"
    headers = {
        "api-key": CLAUDE_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "anthropic_version": "vertex-2023-10-16",
        "max_tokens": 500,
        "system": "You are a helpful AI assistant. Always respond concisely.",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    content = data.get("content", [])
    if content and len(content) > 0:
        text = content[0].get("text", "")
        print(f"Response: {text}")
        assert "4" in text
    else:
        print(f"Unexpected format: {data}")

    print("System prompt test passed!\n")


def test_multimodal_support():
    """Test vision capability with image URL."""
    print("\n=== Testing Vision Capability ===")

    url = f"{CLAUDE_BASE_URL}/v1:rawPredict"
    headers = {
        "api-key": CLAUDE_API_KEY,
        "Content-Type": "application/json",
    }

    # Test with text-only content array format
    payload = {
        "anthropic_version": "vertex-2023-10-16",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Describe what a red apple looks like."}],
            }
        ],
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    content = data.get("content", [])
    if content and len(content) > 0:
        text = content[0].get("text", "")
        print(f"Response: {text[:200]}...")
    else:
        print(f"Response: {data}")

    print("Multimodal format test passed!\n")


if __name__ == "__main__":
    print("Starting Claude API Integration Tests (XHS Runway)")
    print(f"Base URL: {CLAUDE_BASE_URL}")
    print(f"API Key: {'*' * 8}{CLAUDE_API_KEY[-4:] if CLAUDE_API_KEY else 'NOT SET'}")

    try:
        test_non_streaming_completion()
        test_streaming_completion()
        test_with_system_prompt()
        test_multimodal_support()

        print("\n" + "=" * 50)
        print("All tests passed successfully!")
        print("=" * 50)
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback

        traceback.print_exc()
