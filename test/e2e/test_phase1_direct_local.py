#!/usr/bin/env python3
"""Direct SGLang/vLLM server smoke tests (Phase A).

This module tests local model servers directly without going through the
hybridInference gateway. It validates that the underlying model services
(SGLang, vLLM, etc.) are functioning correctly.

Test Coverage:
    - GET /v1/models: Model listing
    - POST /v1/chat/completions: Non-streaming inference
    - POST /v1/chat/completions (stream=true): Streaming inference

Configuration:
    1. Default: Uses test/fixtures/local_servers.yaml
    2. Custom config: Set SGLANG_TEST_CONFIG environment variable
    3. Quick test: Set SGLANG_TEST_URLS="url1,url2,..." environment variable

Example Usage:
    # Test single server
    SGLANG_TEST_URLS="http://localhost:8001" pytest test/external/test_local_sglang.py -v

    # Test all configured servers
    pytest test/external/test_local_sglang.py -v

    # See server health summary
    pytest test/external/test_local_sglang.py::test_all_servers_summary -v -s
"""

import json
import os
from pathlib import Path
from typing import Any

import pytest
import requests
import yaml

pytestmark = pytest.mark.external


# ============================================================================
# Configuration Loading
# ============================================================================


def load_server_configs() -> list[dict[str, Any]]:
    """Load server configurations from YAML or environment variable.

    Priority order:
        1. SGLANG_TEST_URLS environment variable (comma-separated URLs)
        2. SGLANG_TEST_CONFIG environment variable (path to YAML file)
        3. test/fixtures/local_servers.yaml (default location)
        4. Fallback to single server at http://localhost:8001

    Returns:
        List of server configuration dictionaries. Each dict contains:
            - name: Server identifier
            - url: Base URL (e.g., "http://localhost:8001")
            - enabled: Whether to test this server
            - timeout: Request timeout in seconds
            - expected_model_id: Optional model ID to verify (can be None)
            - test_prompts: Optional custom test prompts
            - max_tokens: Max tokens for test requests
            - temperature: Temperature for test requests
    """
    # Option 1: Direct URL list from environment (highest priority)
    env_urls = os.getenv("SGLANG_TEST_URLS")
    if env_urls:
        urls = [url.strip() for url in env_urls.split(",") if url.strip()]
        return [
            {
                "name": f"server-{i}",
                "url": url,
                "timeout": 30,
                "enabled": True,
                "expected_model_id": None,
                "max_tokens": 64,
                "temperature": 0.1,
            }
            for i, url in enumerate(urls)
        ]

    # Option 2: Config file path from environment or default location
    config_path_str = os.getenv("SGLANG_TEST_CONFIG", "test/fixtures/local_servers.yaml")
    config_path = Path(config_path_str)

    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        servers = config.get("servers", [])
        defaults = config.get("defaults", {})

        # Only return enabled servers, apply defaults
        enabled_servers = []
        for server in servers:
            if not server.get("enabled", True):
                continue

            # Apply defaults for missing fields
            server.setdefault("timeout", defaults.get("timeout", 30))
            server.setdefault("max_tokens", defaults.get("max_tokens", 64))
            server.setdefault("temperature", defaults.get("temperature", 0.1))
            server.setdefault("expected_model_id", None)

            # Merge test_prompts with defaults
            if "test_prompts" not in server and "test_prompts" in defaults:
                server["test_prompts"] = defaults["test_prompts"]

            enabled_servers.append(server)

        if enabled_servers:
            return enabled_servers

    # Option 3: Fallback to single default server
    return [
        {
            "name": "default-sglang",
            "url": "http://localhost:8001",
            "timeout": 30,
            "enabled": True,
            "expected_model_id": None,
            "max_tokens": 64,
            "temperature": 0.1,
        }
    ]


# ============================================================================
# Pytest Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def server_configs() -> list[dict[str, Any]]:
    """All server configurations for this test session.

    Returns:
        List of all configured servers (enabled and disabled).
    """
    return load_server_configs()


@pytest.fixture(scope="module", params=load_server_configs(), ids=lambda s: s["name"])
def server(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Parametrized fixture: creates one test instance per server.

    This fixture causes each test function to be executed once for each
    configured server, making it easy to identify which servers pass/fail.

    Args:
        request: Pytest request object

    Returns:
        Server configuration dictionary
    """
    return request.param


def ensure_server_up(server: dict[str, Any]) -> None:
    """Check if server is reachable, skip test if not available.

    This function performs a quick health check by calling GET /v1/models.
    If the server is not reachable, the test is skipped (not failed).

    Args:
        server: Server configuration dict with 'url' and 'name' fields

    Raises:
        pytest.skip: If server is not reachable within timeout
    """
    # Normalize base URL to avoid double slashes when joining paths.
    url = server["url"].rstrip("/")
    name = server["name"]

    try:
        response = requests.get(f"{url}/v1/models", timeout=2)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        pytest.skip(
            f"Server '{name}' at {url} not reachable: {e}\n"
            f"Hint: Start the server or set enabled=false in config."
        )


# ============================================================================
# Phase A: Core Smoke Tests
# ============================================================================


def test_models_endpoint(server):
    """Test GET /v1/models endpoint (OpenAI compatibility).

    Validates:
        - HTTP 200 status code
        - Response has {"object": "list", "data": [...]} structure
        - Data array is non-empty
        - Each model has an "id" field
        - Expected model ID is present (if configured)

    This is the most basic test - if this fails, the server is not
    running or not OpenAI-compatible.
    """
    ensure_server_up(server)

    url = server["url"]
    name = server["name"]

    # Make request
    response = requests.get(f"{url}/v1/models", timeout=10)

    # Validate HTTP status
    assert response.status_code == 200, (
        f"[{name}] Models endpoint returned {response.status_code}. Response: {response.text[:200]}"
    )

    # Parse JSON response
    data = response.json()

    # Validate OpenAI-compatible structure
    assert data.get("object") == "list", (
        f"[{name}] Expected object='list', got '{data.get('object')}'. "
        f"This may indicate the server is not OpenAI-compatible."
    )

    models = data.get("data", [])
    assert len(models) > 0, f"[{name}] No models returned by /v1/models"

    # Validate each model has required fields
    for i, model in enumerate(models):
        assert "id" in model, f"[{name}] Model #{i} missing 'id' field: {model}"

    # Optional: Verify expected model ID (if configured)
    expected_id = server.get("expected_model_id")
    if expected_id:
        model_ids = [m["id"] for m in models]
        assert expected_id in model_ids, (
            f"[{name}] Expected model '{expected_id}' not found.\nAvailable models: {model_ids}"
        )

    # Success message
    model_ids_str = ", ".join([m["id"] for m in models[:3]])
    if len(models) > 3:
        model_ids_str += f", ... ({len(models)} total)"
    print(f"OK [{name}] Found models: {model_ids_str}")


def test_non_streaming_completion(server):
    """Test non-streaming chat completion (POST /v1/chat/completions).

    Validates:
        - HTTP 200 status code
        - Response has "choices" array
        - First choice has message with role="assistant"
        - Message content is non-empty
        - Response follows OpenAI schema

    This verifies that the server can perform basic inference.
    """
    ensure_server_up(server)

    url = server["url"].rstrip("/")
    name = server["name"]

    # Step 1: Get model ID from /v1/models
    models_resp = requests.get(f"{url}/v1/models", timeout=5)
    models_resp.raise_for_status()
    model_id = models_resp.json()["data"][0]["id"]

    # Step 2: Prepare chat completion request
    prompt = server.get("test_prompts", {}).get("simple", "Say 'Hello, World!' and nothing else.")

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": server.get("max_tokens", 64),
        "temperature": server.get("temperature", 0.1),
    }

    # Step 3: Make request
    response = requests.post(
        f"{url}/v1/chat/completions", json=payload, timeout=server.get("timeout", 30)
    )

    # Validate HTTP status
    assert response.status_code == 200, (
        f"[{name}] Completion failed with status {response.status_code}.\n"
        f"Response: {response.text[:500]}"
    )

    # Parse response
    data = response.json()

    # Validate response structure
    assert "choices" in data, f"[{name}] Response missing 'choices' field"
    assert len(data["choices"]) > 0, f"[{name}] Empty choices array"

    choice = data["choices"][0]
    assert "message" in choice, f"[{name}] Choice missing 'message' field"

    message = choice["message"]
    assert message.get("role") == "assistant", (
        f"[{name}] Expected role='assistant', got '{message.get('role')}'"
    )

    content = message.get("content", "")
    assert len(content) > 0, f"[{name}] Empty response content"

    # Success message with truncated response
    content_preview = content[:100] + "..." if len(content) > 100 else content
    print(f"OK [{name}] Response: {content_preview}")


def test_streaming_completion(server):
    """Test streaming chat completion (POST /v1/chat/completions with stream=true).

    Validates:
        - HTTP 200 status code
        - Server sends Server-Sent Events (SSE) format
        - Receives chunks with "data: " prefix
        - At least one chunk contains content
        - Stream ends with "data: [DONE]" marker

    This verifies that the server supports streaming inference,
    which is critical for interactive applications.
    """
    ensure_server_up(server)

    url = server["url"].rstrip("/")
    name = server["name"]

    # Step 1: Get model ID
    models_resp = requests.get(f"{url}/v1/models", timeout=5)
    models_resp.raise_for_status()
    model_id = models_resp.json()["data"][0]["id"]

    # Step 2: Prepare streaming request
    prompt = server.get("test_prompts", {}).get("streaming", "Count from 1 to 3.")

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": server.get("max_tokens", 64),
        "temperature": server.get("temperature", 0.1),
        "stream": True,
    }

    # Step 3: Make streaming request
    # Use a (connect_timeout, read_timeout) tuple for streaming stability.
    # Avoid accessing response.text on streaming responses to prevent blocking.
    stream_timeout = server.get("timeout", 30)
    with requests.post(
        f"{url}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=(5, stream_timeout),
    ) as response:
        # Validate HTTP status
        assert response.status_code == 200, (
            f"[{name}] Streaming failed with status {response.status_code}.\n"
            f"Headers: {dict(response.headers)}"
        )

        # Step 4: Process SSE stream
        chunk_count = 0
        content_chunk_count = 0
        found_done = False
        collected_content = []

        for line in response.iter_lines():
            if not line:
                continue

            decoded = line.decode("utf-8")

            # Verify SSE format
            if not decoded.startswith("data: "):
                # Some servers may send comments or other SSE events
                continue

            body = decoded[6:].strip()

            # Check for stream termination
            if body == "[DONE]":
                found_done = True
                break

            # Parse JSON chunk
            try:
                chunk = json.loads(body)
                chunk_count += 1

                # Extract content from delta
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    if delta.get("content"):
                        content_chunk_count += 1
                        collected_content.append(delta["content"])

            except json.JSONDecodeError:
                # Ignore malformed chunks (shouldn't happen but be defensive)
                continue

        # Validate streaming behavior
        assert chunk_count > 0, (
            f"[{name}] No valid JSON chunks received. Server may not be streaming properly."
        )

        assert content_chunk_count > 0, (
            f"[{name}] No content chunks received. "
            f"Received {chunk_count} chunks but none contained content."
        )

        assert found_done, (
            f"[{name}] Stream did not send [DONE] marker. This may cause client hangs."
        )

        # Success message with response preview
        full_response = "".join(collected_content)
        response_preview = (
            full_response[:100] + "..." if len(full_response) > 100 else full_response
        )
        print(f"OK [{name}] Streamed {content_chunk_count} chunks: {response_preview}")


# ============================================================================
# Summary Test
# ============================================================================


def test_all_servers_summary(server_configs):
    """Print health summary of all configured servers.

    This test runs once (not parametrized) and provides a quick overview
    of which servers are reachable and how many models they expose.

    Useful for:
        - Quick health checks before running full test suite
        - Debugging configuration issues
        - CI/CD dashboards

    Example output:
        ======================================================================
        Local SGLang Servers Health Summary
        ======================================================================
        UP           llama-3.2-3b-local      http://localhost:8001      (1 models)
        DOWN         qwen-2.5-7b-local       http://localhost:8002
        ======================================================================
    """
    results = {}

    for config in server_configs:
        name = config["name"]
        url = config["url"]

        try:
            base = url.rstrip("/")
            response = requests.get(f"{base}/v1/models", timeout=2)

            if response.status_code == 200:
                models = response.json().get("data", [])
                results[name] = {
                    "status": "UP",
                    "url": url,
                    "models": len(models),
                }
            else:
                results[name] = {
                    "status": f"HTTP {response.status_code}",
                    "url": url,
                }

        except requests.exceptions.Timeout:
            results[name] = {
                "status": "TIMEOUT",
                "url": url,
            }
        except requests.exceptions.ConnectionError:
            results[name] = {
                "status": "DOWN",
                "url": url,
            }
        except Exception as e:
            results[name] = {
                "status": "ERROR",
                "url": url,
                "error": str(e)[:50],
            }

    # Print formatted summary table
    print("\n" + "=" * 80)
    print("Local SGLang Servers Health Summary")
    print("=" * 80)

    for name, info in results.items():
        status = info["status"]
        url = info["url"]
        extra = f"({info['models']} models)" if "models" in info else ""
        error = f"[{info['error']}]" if "error" in info else ""

        print(f"{status:15s} {name:30s} {url:25s} {extra} {error}")

    print("=" * 80)

    # Skip if no servers are reachable
    up_count = sum(1 for r in results.values() if "UP" in r["status"])

    if up_count == 0:
        pytest.skip(
            "No servers are available!\n"
            "Hint: Start at least one SGLang server or check your configuration.\n"
            f"Config file: {os.getenv('SGLANG_TEST_CONFIG', 'test/fixtures/local_servers.yaml')}"
        )

    print(f"\nOK {up_count}/{len(results)} servers are healthy\n")
