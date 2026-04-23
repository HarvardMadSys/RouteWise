import os

import pytest
from dotenv import load_dotenv
from openai import OpenAI

pytestmark = pytest.mark.external


def test_llama_basic_models():
    """Call Llama models via OpenAI compatibility API; requires network."""
    load_dotenv()
    api_key = os.environ.get("LLAMA_API_KEY")
    if not api_key:
        pytest.skip("LLAMA_API_KEY not configured", allow_module_level=False)

    client = OpenAI(api_key=api_key, base_url="https://api.llama.com/compat/v1/")

    completion8 = client.chat.completions.create(
        model="Llama-3.3-8B-Instruct",
        messages=[
            {"role": "developer", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
    )
    assert completion8 is not None

    completion70 = client.chat.completions.create(
        model="Llama-3.3-70B-Instruct",
        messages=[
            {"role": "developer", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
    )
    assert completion70 is not None


def test_llama_scout_models():
    """Try Llama Scout variants; may fail depending on availability."""
    load_dotenv()
    api_key = os.environ.get("LLAMA_API_KEY")
    if not api_key:
        pytest.skip("LLAMA_API_KEY not configured", allow_module_level=False)

    client = OpenAI(api_key=api_key, base_url="https://api.llama.com/compat/v1/")

    try:
        completion_scout = client.chat.completions.create(
            model="Llama-4-Scout-17B-16E-Instruct-FP8",
            messages=[
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello Scout!"},
            ],
        )
        assert completion_scout is not None
    except Exception:
        pytest.xfail("Llama-4-Scout-17B-16E-Instruct-FP8 may not be available")

    try:
        completion_scout2 = client.chat.completions.create(
            model="Llama-4-Scout-17B-Instruct",
            messages=[
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello Scout!"},
            ],
        )
        assert completion_scout2 is not None
    except Exception:
        pytest.xfail("Llama-4-Scout-17B-Instruct may not be available")
