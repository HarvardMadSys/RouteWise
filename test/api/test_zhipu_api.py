"""Test Zhipu AI GLM models API directly (external)."""

import os

import pytest
from dotenv import load_dotenv
from openai import OpenAI

pytestmark = pytest.mark.external


def test_zhipu_glm45_basic():
    """Basic GLM-4.5 call; requires network and ZAI_API_KEY."""
    load_dotenv()
    api_key = os.environ.get("ZAI_API_KEY")
    if not api_key:
        pytest.skip("ZAI_API_KEY not configured", allow_module_level=False)

    client = OpenAI(api_key=api_key, base_url="https://api.z.ai/api/coding/paas/v4/")
    completion = client.chat.completions.create(
        model="glm-4.5",
        messages=[
            {
                "role": "user",
                "content": "Hello! Please respond with 'Hi there' to confirm you're working.",
            },
        ],
        temperature=0.7,
        max_tokens=20,
    )
    assert completion is not None


def test_zhipu_glm45_air_basic():
    """Basic GLM-4.5-Air call; requires network and ZAI_API_KEY."""
    load_dotenv()
    api_key = os.environ.get("ZAI_API_KEY")
    if not api_key:
        pytest.skip("ZAI_API_KEY not configured", allow_module_level=False)

    client = OpenAI(api_key=api_key, base_url="https://api.z.ai/api/coding/paas/v4/")
    completion = client.chat.completions.create(
        model="glm-4.5-air",
        messages=[
            {
                "role": "user",
                "content": "Hello! Please respond with 'Hi there' to confirm you're working.",
            },
        ],
        temperature=0.7,
        max_tokens=20,
    )
    assert completion is not None
    assert completion.choices[0].message.content is not None


def test_zhipu_glm46_basic():
    """Basic GLM-4.6 call; requires network and ZAI_API_KEY."""
    load_dotenv()
    api_key = os.environ.get("ZAI_API_KEY")
    if not api_key:
        pytest.skip("ZAI_API_KEY not configured", allow_module_level=False)

    client = OpenAI(api_key=api_key, base_url="https://api.z.ai/api/coding/paas/v4/")
    completion = client.chat.completions.create(
        model="glm-4.6",
        messages=[
            {
                "role": "user",
                "content": "Hello! Please respond with 'Hi there' to confirm you're working.",
            },
        ],
        temperature=0.7,
        max_tokens=20,
    )
    assert completion is not None
    assert completion.choices[0].message.content is not None
