#!/usr/bin/env python3
"""External server tests (health, models, completions) against an OpenRouter gateway."""

import json
import os

import pytest
import requests

from serving.schemas import ChatCompletionResponse, ModelList

BASE_URL = os.getenv("EXTERNAL_BASE_URL", "http://localhost").rstrip("/")

pytestmark = pytest.mark.external


def _ensure_up(base_url: str) -> None:
    try:
        requests.get(f"{base_url}/health", timeout=0.5)
    except Exception:
        pytest.skip(f"Server not reachable at {base_url}")


@pytest.fixture(scope="module")
def base_url() -> str:
    return BASE_URL


def test_health_check(base_url: str) -> None:
    _ensure_up(base_url)
    response = requests.get(f"{base_url}/health", timeout=2)
    assert response.status_code == 200


def test_models_endpoint(base_url: str) -> None:
    _ensure_up(base_url)
    response = requests.get(f"{base_url}/v1/models", timeout=10)
    assert response.status_code == 200
    parsed = ModelList.model_validate(response.json())
    assert parsed.data, "models list should not be empty"


def test_non_streaming_completion(base_url: str) -> None:
    _ensure_up(base_url)
    models_payload = requests.get(f"{base_url}/v1/models", timeout=5).json().get("data", [])
    model_id = models_payload[0]["id"] if models_payload else "llama-4-scout"
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! Please respond with 'Hello, World!'"},
        ],
        "max_tokens": 32,
        "temperature": 0.1,
    }
    response = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=15)
    assert response.status_code == 200
    parsed = ChatCompletionResponse.model_validate(response.json())
    assert parsed.choices, "chat completion must return at least one choice"
    assert parsed.choices[0].message.role == "assistant"


def test_streaming_completion(base_url: str) -> None:
    _ensure_up(base_url)
    models_payload = requests.get(f"{base_url}/v1/models", timeout=5).json().get("data", [])
    model_id = models_payload[0]["id"] if models_payload else "llama-4-scout"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Stream a short reply."}],
        "max_tokens": 64,
        "temperature": 0.1,
        "stream": True,
    }
    response = requests.post(
        f"{base_url}/v1/chat/completions", json=payload, stream=True, timeout=30
    )
    assert response.status_code == 200
    chunk_count = 0
    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if not decoded.startswith("data: "):
            continue
        body = decoded[6:].strip()
        if body == "[DONE]":
            break
        try:
            json.loads(body)
        except json.JSONDecodeError:
            continue
        chunk_count += 1

    assert chunk_count > 0
