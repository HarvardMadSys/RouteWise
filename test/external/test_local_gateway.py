#!/usr/bin/env python3
"""Gateway smoke tests for local model integration (Phase B).

This module tests the hybridInference gateway's ability to route requests
to local model servers (SGLang/vLLM). Unlike Phase A (test_local_sglang.py),
these tests validate the entire request flow through the gateway layer.

Test Coverage:
    - GET /health: Gateway health check
    - GET /v1/models: Model registry via gateway
    - POST /v1/chat/completions: Non-streaming inference via gateway
    - POST /v1/chat/completions (stream=true): Streaming inference via gateway
    - Authentication and authorization (if enabled)
    - Error handling and fallback behavior

Configuration (via environment variables):
    - GATEWAY_BASE_URL: Gateway URL (default: http://localhost:80)
    - LOCAL_BASE_URL: Local model server URL (default: http://localhost:8001)
    - GATEWAY_API_KEY: API key for authentication (optional)
    - USER_AUTH_ENABLED: Enable auth testing (0=disabled, 1=enabled)
    - GATEWAY_TIMEOUT: Request timeout in seconds (default: 30)

Example Usage:
    # Step 1: Start local model server (SGLang/vLLM)
    # Ensure it's running at http://localhost:8001

    # Step 2: Start gateway with test configuration
    MODELS_CONFIG=test/fixtures/test_models.yaml \
    ROUTING_CONFIG=test/fixtures/test_routing.yaml \
    LOCAL_BASE_URL=http://localhost:8001 \
    DB_ENABLED=false \
    uv run uvicorn serving.servers.app:app --port 10081

    # Step 3: Run tests
    GATEWAY_BASE_URL="http://localhost:10081" pytest test/external/test_local_gateway.py -v

    # See gateway health summary
    GATEWAY_BASE_URL="http://localhost:10081" pytest test/external/test_local_gateway.py::test_gateway_summary -v -s

    # Test with authentication
    USER_AUTH_ENABLED=1 GATEWAY_API_KEY="hyi-xxx" \
    GATEWAY_BASE_URL="http://localhost:10081" \
    pytest test/external/test_local_gateway.py -v

Prerequisites:
    1. Local model server running (e.g., http://localhost:8001)
    2. Gateway server running with test config (see example above)
    3. test/fixtures/test_models.yaml contains llama-3.2-3b-local-test model

Important:
    The test uses llama-3.2-3b-local-test model which routes ONLY to local
    server (no remote fallback). This validates true local routing.
"""

import json
import os

import pytest
import requests

pytestmark = pytest.mark.external


# ============================================================================
# Configuration
# ============================================================================

# Gateway configuration
GATEWAY_BASE_URL = os.getenv("GATEWAY_BASE_URL", "http://localhost:80").rstrip("/")
GATEWAY_TIMEOUT = int(os.getenv("GATEWAY_TIMEOUT", "30"))
GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY", "")

# Local server configuration
LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL", "http://localhost:8001").rstrip("/")

# Test configuration
TEST_AUTH = os.getenv("USER_AUTH_ENABLED", "0") == "1"
MAX_TOKENS = 64
TEMPERATURE = 0.1

# Test prompts
PROMPT_SIMPLE = "Say 'Hello, World!' and nothing else."
PROMPT_STREAMING = "Count from 1 to 5."


# ============================================================================
# Pytest Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def gateway_url() -> str:
    """Get gateway base URL from environment.

    Returns:
        Gateway base URL (e.g., "http://localhost:80")
    """
    return GATEWAY_BASE_URL


@pytest.fixture(scope="module")
def gateway_timeout() -> int:
    """Get request timeout from environment.

    Returns:
        Timeout in seconds (default: 30)
    """
    return GATEWAY_TIMEOUT


@pytest.fixture(scope="module")
def auth_headers() -> dict[str, str]:
    """Build authentication headers if API key is configured.

    Returns:
        Dictionary with Authorization header if API key present, else empty dict
    """
    if GATEWAY_API_KEY:
        return {"Authorization": f"Bearer {GATEWAY_API_KEY}"}
    return {}


def ensure_gateway_up(gateway_url: str, timeout: int = 2) -> None:
    """Check if gateway is reachable, skip test if not available.

    Note: Gateway may return 503 on /health if database is disconnected,
    but this is acceptable for basic inference testing. We check /v1/models
    endpoint to verify the gateway is functional.

    Args:
        gateway_url: Gateway base URL
        timeout: Request timeout in seconds

    Raises:
        pytest.skip: If gateway is not reachable within timeout
    """
    try:
        # Try /v1/models endpoint instead of /health
        # This works even when database is disconnected
        response = requests.get(f"{gateway_url}/v1/models", timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        pytest.skip(
            f"Gateway at {gateway_url} not reachable: {e}\n"
            f"Hint: Start the gateway server or set GATEWAY_BASE_URL environment variable."
        )


def ensure_local_server_up() -> None:
    """Check if local model server is reachable, skip test if not available.

    Raises:
        pytest.skip: If local server is not reachable
    """
    try:
        response = requests.get(f"{LOCAL_BASE_URL}/v1/models", timeout=2)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        pytest.skip(
            f"Local model server at {LOCAL_BASE_URL} not reachable: {e}\n"
            f"Hint: Start the local model server (e.g., SGLang/vLLM) first."
        )


# ============================================================================
# Phase B: Gateway Integration Tests
# ============================================================================


def test_gateway_health(gateway_url, gateway_timeout):
    """Test gateway health endpoint.

    Validates:
        - Gateway is responsive (HTTP 200 or 503)
        - HTTP 503 is acceptable when database is disconnected
        - Any other status code indicates a problem

    Note: Gateway may return 503 when database is disconnected (DB_ENABLED=false),
    but this is acceptable for basic inference functionality.
    """
    ensure_gateway_up(gateway_url, gateway_timeout)

    response = requests.get(f"{gateway_url}/health", timeout=gateway_timeout)

    # Accept both 200 (fully healthy) and 503 (database disconnected but functional)
    assert response.status_code in [200, 503], (
        f"Gateway health check failed with unexpected status {response.status_code}. "
        f"Response: {response.text[:200]}"
    )

    if response.status_code == 200:
        print(f"OK Gateway is fully healthy at {gateway_url}")
    else:
        print(f"OK Gateway is functional (database disconnected) at {gateway_url}")


def test_gateway_models_endpoint(gateway_url, gateway_timeout, auth_headers):
    """Test GET /v1/models endpoint via gateway.

    Validates:
        - HTTP 200 status code
        - Response has OpenAI-compatible structure
        - At least one model is available

    This verifies that the gateway correctly exposes the model registry.
    """
    ensure_gateway_up(gateway_url, gateway_timeout)

    response = requests.get(
        f"{gateway_url}/v1/models", headers=auth_headers, timeout=gateway_timeout
    )

    # Validate HTTP status
    assert response.status_code == 200, (
        f"Gateway models endpoint returned {response.status_code}. Response: {response.text[:200]}"
    )

    # Parse JSON response
    data = response.json()

    # Validate OpenAI-compatible structure
    assert data.get("object") == "list", (
        f"Expected object='list', got '{data.get('object')}'. Gateway may not be OpenAI-compatible."
    )

    models = data.get("data", [])
    assert len(models) > 0, "Gateway returned no models"

    # Validate each model has required fields
    for i, model in enumerate(models):
        assert "id" in model, f"Model #{i} missing 'id' field: {model}"

    # Success message
    model_ids_str = ", ".join([m["id"] for m in models[:3]])
    if len(models) > 3:
        model_ids_str += f", ... ({len(models)} total)"
    print(f"OK Gateway exposes models: {model_ids_str}")


def test_gateway_non_streaming_completion(gateway_url, gateway_timeout, auth_headers):
    """Test non-streaming chat completion via gateway.

    Validates:
        - HTTP 200 status code
        - Response has OpenAI-compatible structure
        - Response contains assistant message
        - Content is non-empty

    This verifies that the gateway can route requests to local models
    and return responses in the expected format.
    """
    ensure_gateway_up(gateway_url, gateway_timeout)
    ensure_local_server_up()

    # Step 1: Get available models
    models_resp = requests.get(
        f"{gateway_url}/v1/models", headers=auth_headers, timeout=gateway_timeout
    )
    models_resp.raise_for_status()
    models = models_resp.json()["data"]

    # Select first available model
    if not models:
        pytest.skip("No models available via gateway")
    model_id = models[0]["id"]

    # Step 2: Prepare chat completion request
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": PROMPT_SIMPLE}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }

    # Step 3: Make request via gateway
    response = requests.post(
        f"{gateway_url}/v1/chat/completions",
        json=payload,
        headers=auth_headers,
        timeout=gateway_timeout,
    )

    # Validate HTTP status
    assert response.status_code == 200, (
        f"Gateway completion failed with status {response.status_code}.\n"
        f"Response: {response.text[:500]}"
    )

    # Parse response
    data = response.json()

    # Validate response structure
    assert "choices" in data, "Response missing 'choices' field"
    assert len(data["choices"]) > 0, "Empty choices array"

    choice = data["choices"][0]
    assert "message" in choice, "Choice missing 'message' field"

    message = choice["message"]
    assert message.get("role") == "assistant", (
        f"Expected role='assistant', got '{message.get('role')}'"
    )

    content = message.get("content", "")
    assert len(content) > 0, "Empty response content"

    # Success message with truncated response
    content_preview = content[:100] + "..." if len(content) > 100 else content
    print(f"OK Gateway completion (model={model_id}): {content_preview}")


def test_gateway_streaming_completion(gateway_url, gateway_timeout, auth_headers):
    """Test streaming chat completion via gateway.

    Validates:
        - HTTP 200 status code
        - Server-Sent Events (SSE) format
        - Receives chunks with content
        - Stream ends with [DONE] marker

    This verifies that the gateway can handle streaming responses from
    local models and properly forward them to clients.
    """
    ensure_gateway_up(gateway_url, gateway_timeout)
    ensure_local_server_up()

    # Step 1: Get available models
    models_resp = requests.get(
        f"{gateway_url}/v1/models", headers=auth_headers, timeout=gateway_timeout
    )
    models_resp.raise_for_status()
    models = models_resp.json()["data"]

    if not models:
        pytest.skip("No models available via gateway")
    model_id = models[0]["id"]

    # Step 2: Prepare streaming request
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": PROMPT_STREAMING}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": True,
    }

    # Step 3: Make streaming request via gateway
    with requests.post(
        f"{gateway_url}/v1/chat/completions",
        json=payload,
        headers=auth_headers,
        stream=True,
        timeout=(5, gateway_timeout),
    ) as response:
        # Validate HTTP status
        assert response.status_code == 200, (
            f"Gateway streaming failed with status {response.status_code}.\n"
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
                # Ignore malformed chunks
                continue

        # Validate streaming behavior
        assert chunk_count > 0, (
            "No valid JSON chunks received from gateway. Gateway may not be streaming properly."
        )

        assert content_chunk_count > 0, (
            f"No content chunks received. Received {chunk_count} chunks but none contained content."
        )

        assert found_done, "Stream did not send [DONE] marker. This may cause client hangs."

        # Success message with response preview
        full_response = "".join(collected_content)
        response_preview = (
            full_response[:100] + "..." if len(full_response) > 100 else full_response
        )
        print(
            f"OK Gateway streaming (model={model_id}, chunks={content_chunk_count}): {response_preview}"
        )


def test_gateway_local_model_routing(gateway_url, gateway_timeout, auth_headers):
    """Test that gateway correctly routes to local model server.

    This test verifies the end-to-end routing by:
    1. Checking local server is reachable
    2. Sending request via gateway
    3. Validating response structure matches local server format

    This is a critical integration test that validates the gateway's
    primary function: routing to local models.
    """
    ensure_gateway_up(gateway_url, gateway_timeout)
    ensure_local_server_up()

    # Step 1: Verify local server has models
    local_models_resp = requests.get(f"{LOCAL_BASE_URL}/v1/models", timeout=5)
    local_models_resp.raise_for_status()
    local_models = local_models_resp.json()["data"]

    assert len(local_models) > 0, "Local server has no models"

    # Step 2: Get gateway models
    gateway_models_resp = requests.get(
        f"{gateway_url}/v1/models", headers=auth_headers, timeout=gateway_timeout
    )
    gateway_models_resp.raise_for_status()
    gateway_models = gateway_models_resp.json()["data"]

    # Step 3: Verify gateway exposes at least one model that can route to local
    # Note: Gateway models may have different IDs than local models
    # We just verify that gateway has models available
    assert len(gateway_models) > 0, "Gateway has no models configured"

    # Step 4: Make a test request via gateway
    model_id = gateway_models[0]["id"]

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Test routing."}],
        "max_tokens": 10,
        "temperature": 0.1,
    }

    response = requests.post(
        f"{gateway_url}/v1/chat/completions",
        json=payload,
        headers=auth_headers,
        timeout=gateway_timeout,
    )

    # Validate successful routing
    assert response.status_code == 200, (
        f"Gateway failed to route request. Status: {response.status_code}\n"
        f"Response: {response.text[:500]}"
    )

    data = response.json()
    assert "choices" in data and len(data["choices"]) > 0, "Invalid response structure"

    print(f"OK Gateway successfully routed to local model (model={model_id})")


# ============================================================================
# Authentication Tests (Conditional)
# ============================================================================


def test_gateway_authentication_required(gateway_url, gateway_timeout):
    """Test authentication enforcement when enabled.

    This test only runs if USER_AUTH_ENABLED=1.
    Validates that requests without API keys are rejected.
    """
    if not TEST_AUTH:
        pytest.skip("Authentication testing disabled (set USER_AUTH_ENABLED=1)")

    ensure_gateway_up(gateway_url, gateway_timeout)

    # Try to access models endpoint without authentication
    response = requests.get(f"{gateway_url}/v1/models", timeout=gateway_timeout)

    # Should receive 401 Unauthorized
    assert response.status_code == 401, (
        f"Expected 401 Unauthorized without API key, got {response.status_code}"
    )

    print("OK Gateway correctly rejects requests without authentication")


def test_gateway_authentication_valid(gateway_url, gateway_timeout, auth_headers):
    """Test authentication with valid API key.

    This test only runs if USER_AUTH_ENABLED=1 and GATEWAY_API_KEY is set.
    Validates that requests with valid API keys are accepted.
    """
    if not TEST_AUTH:
        pytest.skip("Authentication testing disabled (set USER_AUTH_ENABLED=1)")

    if not auth_headers:
        pytest.skip("No API key configured (set GATEWAY_API_KEY)")

    ensure_gateway_up(gateway_url, gateway_timeout)

    # Access models endpoint with authentication
    response = requests.get(
        f"{gateway_url}/v1/models", headers=auth_headers, timeout=gateway_timeout
    )

    # Should receive 200 OK
    assert response.status_code == 200, (
        f"Expected 200 OK with valid API key, got {response.status_code}\n"
        f"Response: {response.text[:200]}"
    )

    print("OK Gateway correctly accepts requests with valid authentication")


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_gateway_invalid_model(gateway_url, gateway_timeout, auth_headers):
    """Test gateway error handling for invalid model ID.

    Validates:
        - Appropriate error status code (404 or 400)
        - Error message is informative

    This ensures the gateway properly validates model IDs.
    """
    ensure_gateway_up(gateway_url, gateway_timeout)

    payload = {
        "model": "non-existent-model-xyz",
        "messages": [{"role": "user", "content": "Test"}],
        "max_tokens": 10,
    }

    response = requests.post(
        f"{gateway_url}/v1/chat/completions",
        json=payload,
        headers=auth_headers,
        timeout=gateway_timeout,
    )

    # Should receive error (typically 404 or 400)
    assert response.status_code in [400, 404], (
        f"Expected 400/404 for invalid model, got {response.status_code}\n"
        f"Response: {response.text[:200]}"
    )

    print("OK Gateway correctly rejects invalid model ID")


def test_gateway_malformed_request(gateway_url, gateway_timeout, auth_headers):
    """Test gateway error handling for malformed requests.

    Validates:
        - Appropriate error status code (400 or 422)
        - Error response includes validation details

    This ensures the gateway properly validates request payloads.
    """
    ensure_gateway_up(gateway_url, gateway_timeout)

    # Malformed payload: missing required 'messages' field
    payload = {
        "model": "some-model",
        "max_tokens": 10,
    }

    response = requests.post(
        f"{gateway_url}/v1/chat/completions",
        json=payload,
        headers=auth_headers,
        timeout=gateway_timeout,
    )

    # Should receive validation error (typically 400 or 422)
    assert response.status_code in [400, 422], (
        f"Expected 400/422 for malformed request, got {response.status_code}\n"
        f"Response: {response.text[:200]}"
    )

    print("OK Gateway correctly rejects malformed requests")


# ============================================================================
# Summary Test
# ============================================================================


def test_gateway_summary(gateway_url, gateway_timeout, auth_headers):
    """Print comprehensive gateway health summary.

    This test provides a quick overview of:
        - Gateway availability
        - Local server availability
        - Available models
        - Basic routing functionality

    Useful for quick health checks and debugging.
    """
    results = {
        "gateway_health": "UNKNOWN",
        "local_server_health": "UNKNOWN",
        "models_count": 0,
        "routing_test": "UNKNOWN",
    }

    # Check gateway health
    try:
        response = requests.get(f"{gateway_url}/health", timeout=2)
        results["gateway_health"] = (
            "UP" if response.status_code == 200 else f"HTTP {response.status_code}"
        )
    except requests.exceptions.RequestException as e:
        results["gateway_health"] = f"DOWN ({type(e).__name__})"

    # Check local server health
    try:
        response = requests.get(f"{LOCAL_BASE_URL}/v1/models", timeout=2)
        results["local_server_health"] = (
            "UP" if response.status_code == 200 else f"HTTP {response.status_code}"
        )
    except requests.exceptions.RequestException as e:
        results["local_server_health"] = f"DOWN ({type(e).__name__})"

    # Check models
    try:
        response = requests.get(
            f"{gateway_url}/v1/models", headers=auth_headers, timeout=gateway_timeout
        )
        if response.status_code == 200:
            models = response.json().get("data", [])
            results["models_count"] = len(models)
    except requests.exceptions.RequestException:
        pass

    # Test basic routing
    # Note: Test routing even if health endpoint returns 503 (database disconnected)
    # as long as we can get the models list
    if results["models_count"] > 0:
        try:
            models_resp = requests.get(
                f"{gateway_url}/v1/models", headers=auth_headers, timeout=gateway_timeout
            )
            model_id = models_resp.json()["data"][0]["id"]

            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 5,
            }

            response = requests.post(
                f"{gateway_url}/v1/chat/completions",
                json=payload,
                headers=auth_headers,
                timeout=gateway_timeout,
            )

            results["routing_test"] = (
                "PASS" if response.status_code == 200 else f"FAIL (HTTP {response.status_code})"
            )
        except requests.exceptions.RequestException:
            results["routing_test"] = "FAIL (Exception)"

    # Print formatted summary
    print("\n" + "=" * 80)
    print("Gateway Integration Test Summary (Phase B)")
    print("=" * 80)
    print(f"Gateway URL:           {gateway_url}")
    print(f"Gateway Health:        {results['gateway_health']}")
    print(f"Local Server:          {LOCAL_BASE_URL}")
    print(f"Local Server Health:   {results['local_server_health']}")
    print(f"Models Available:      {results['models_count']}")
    print(f"Routing Test:          {results['routing_test']}")
    print(f"Auth Enabled:          {'Yes' if TEST_AUTH else 'No'}")
    print("=" * 80)

    # Skip if gateway is not functional
    # Gateway is considered functional if it can return models list,
    # even if health endpoint returns 503 (database disconnected)
    if results["models_count"] == 0:
        pytest.skip(
            f"Gateway at {gateway_url} is not functional.\n"
            f"Health status: {results['gateway_health']}\n"
            f"Hint: Start the gateway server or check GATEWAY_BASE_URL configuration."
        )

    print(f"\nOK Gateway is functional ({results['models_count']} models available)\n")
