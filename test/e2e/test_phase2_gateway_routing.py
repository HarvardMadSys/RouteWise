#!/usr/bin/env python3
"""Phase 2: Gateway routing tests for ALL local models.

This module extends test_local_gateway.py to test EACH local model individually,
ensuring both Llama 3.2 3B (port 8001) and Qwen3 Coder 30B (port 8003) are
properly routed through the gateway.

Test Coverage:
    - Non-streaming completion for each local model
    - Streaming completion for each local model
    - Model-specific validation (e.g., code generation for Qwen3)

Configuration (via environment variables):
    - GATEWAY_BASE_URL: Gateway URL (default: http://localhost:80)
    - GATEWAY_TIMEOUT: Request timeout in seconds (default: 30)

Example Usage:
    # Start gateway first
    MODELS_CONFIG=test/fixtures/test_models.yaml \
    DB_ENABLED=false \
    uv run uvicorn serving.servers.app:app --port 10081

    # Run tests
    GATEWAY_BASE_URL="http://localhost:10081" \
    pytest test/e2e/test_phase2_gateway_all_models.py -v -s
"""

import json
import os

import pytest
import requests

pytestmark = pytest.mark.external


# ============================================================================
# Configuration
# ============================================================================

GATEWAY_BASE_URL = os.getenv("GATEWAY_BASE_URL", "http://localhost:80").rstrip("/")
GATEWAY_TIMEOUT = int(os.getenv("GATEWAY_TIMEOUT", "60"))

# Local model IDs to test (from test/fixtures/test_models.yaml)
LOCAL_MODEL_IDS = [
    "llama-3.2-3b-local-test",
    "qwen3-coder-30b",
]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def gateway_url() -> str:
    """Get gateway base URL from environment."""
    return GATEWAY_BASE_URL


@pytest.fixture(scope="module")
def gateway_timeout() -> int:
    """Get request timeout from environment."""
    return GATEWAY_TIMEOUT


def ensure_gateway_up(gateway_url: str, timeout: int = 2) -> None:
    """Check if gateway is reachable, skip test if not available."""
    try:
        response = requests.get(f"{gateway_url}/v1/models", timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        pytest.skip(f"Gateway at {gateway_url} not reachable: {e}")


@pytest.fixture(scope="module")
def available_models(gateway_url, gateway_timeout):
    """Get list of available models from gateway."""
    ensure_gateway_up(gateway_url, gateway_timeout)

    response = requests.get(f"{gateway_url}/v1/models", timeout=gateway_timeout)
    response.raise_for_status()

    models = response.json()["data"]
    return {m["id"]: m for m in models}


# ============================================================================
# Phase 2: Per-Model Tests
# ============================================================================


@pytest.mark.parametrize("model_id", LOCAL_MODEL_IDS)
def test_local_model_non_streaming(gateway_url, gateway_timeout, available_models, model_id):
    """Test non-streaming completion for each local model.

    This ensures both Llama (8001) and Qwen3 (8003) are properly routed.
    """
    if model_id not in available_models:
        pytest.skip(f"Model {model_id} not available in gateway")

    # Prepare request with model-specific prompt
    if "qwen" in model_id.lower():
        prompt = "Write a Python function to add two numbers."
        max_tokens = 200
    else:
        prompt = "Explain what is a binary tree in one sentence."
        max_tokens = 100

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }

    response = requests.post(
        f"{gateway_url}/v1/chat/completions",
        json=payload,
        timeout=gateway_timeout,
    )

    assert response.status_code == 200, (
        f"Model {model_id} completion failed: {response.status_code}\n"
        f"Response: {response.text[:500]}"
    )

    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0

    content = data["choices"][0]["message"]["content"]
    assert len(content) > 10, f"Model {model_id} returned too short response"

    # Model-specific validation
    if "qwen" in model_id.lower():
        # Qwen3 should generate code
        assert "def" in content or "function" in content.lower(), (
            f"Qwen3 should generate code, got: {content[:100]}"
        )

    print(f"✓ [{model_id}] Non-streaming: {content[:80]}...")


@pytest.mark.parametrize("model_id", LOCAL_MODEL_IDS)
def test_local_model_streaming(gateway_url, gateway_timeout, available_models, model_id):
    """Test streaming completion for each local model.

    This ensures both Llama (8001) and Qwen3 (8003) support streaming.
    """
    if model_id not in available_models:
        pytest.skip(f"Model {model_id} not available in gateway")

    # Prepare streaming request
    if "qwen" in model_id.lower():
        prompt = "Write a hello world function."
        max_tokens = 150
    else:
        prompt = "Count from 1 to 3."
        max_tokens = 50

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "stream": True,
    }

    chunk_count = 0
    content_chunks = []
    found_done = False

    with requests.post(
        f"{gateway_url}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=(5, gateway_timeout),
    ) as response:
        assert response.status_code == 200, (
            f"Model {model_id} streaming failed: {response.status_code}"
        )

        for line in response.iter_lines():
            if not line:
                continue

            decoded = line.decode("utf-8")
            if not decoded.startswith("data: "):
                continue

            body = decoded[6:].strip()
            if body == "[DONE]":
                found_done = True
                break

            try:
                chunk = json.loads(body)
                chunk_count += 1

                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    if delta.get("content"):
                        content_chunks.append(delta["content"])
            except json.JSONDecodeError:
                continue

    assert chunk_count > 0, f"Model {model_id} sent no chunks"
    assert len(content_chunks) > 0, f"Model {model_id} sent no content"
    assert found_done, f"Model {model_id} did not send [DONE]"

    full_response = "".join(content_chunks)
    print(f"✓ [{model_id}] Streaming ({len(content_chunks)} chunks): {full_response[:80]}...")


# ============================================================================
# Summary Test
# ============================================================================


def test_phase2_all_models_summary(gateway_url, gateway_timeout, available_models):
    """Print summary of all local models tested through gateway."""
    print("\n" + "=" * 80)
    print("Phase 2: Gateway Routing - All Local Models Summary")
    print("=" * 80)
    print(f"Gateway URL: {gateway_url}")
    print(f"Total models available: {len(available_models)}")
    print()

    for model_id in LOCAL_MODEL_IDS:
        if model_id in available_models:
            model = available_models[model_id]
            print(f"✓ {model_id:30s} - {model.get('name', 'N/A')}")

            # Quick test
            try:
                payload = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 10,
                }
                response = requests.post(
                    f"{gateway_url}/v1/chat/completions",
                    json=payload,
                    timeout=gateway_timeout,
                )
                if response.status_code == 200:
                    print("  → Routing: PASS")
                else:
                    print(f"  → Routing: FAIL (HTTP {response.status_code})")
            except Exception as e:
                print(f"  → Routing: ERROR ({type(e).__name__})")
        else:
            print(f"✗ {model_id:30s} - NOT AVAILABLE")

    print("=" * 80)

    # Verify at least one local model is available
    available_local = [m for m in LOCAL_MODEL_IDS if m in available_models]
    assert len(available_local) > 0, "No local models available through gateway"

    print(
        f"\n✓ Phase 2 Complete: {len(available_local)}/{len(LOCAL_MODEL_IDS)} local models available\n"
    )
