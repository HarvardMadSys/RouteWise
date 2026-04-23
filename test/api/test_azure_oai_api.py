import os

import dotenv
import pytest
from openai import AzureOpenAI

pytestmark = pytest.mark.external

dotenv.load_dotenv()

# api_key = os.getenv("OAI_KEY")
# api_endpoint = os.getenv("OAI_ENDPOINT")
# client = AzureOpenAI(
#   azure_endpoint = api_endpoint,
#   api_key=api_key,
#   api_version="2024-12-01-preview"
# )


# response = client.chat.completions.create(
#     model="o4-mini", # replace with the model deployment name of your o1-preview, or o1-mini model
#     messages=[
#         {"role": "user", "content": "What steps should I think about when writing my first Python API?"},
#     ],
#     max_completion_tokens = 5000
# )
def test_azure_chat_completion():
    """Call Azure OpenAI chat completions; requires valid env and network."""
    if not os.getenv("OAI_KEY") or not os.getenv("OAI_ENDPOINT"):
        pytest.skip("Azure OpenAI credentials not configured", allow_module_level=False)

    api_key = os.getenv("OAI_KEY")
    api_endpoint = os.getenv("OAI_ENDPOINT")
    client = AzureOpenAI(azure_endpoint=api_endpoint, api_key=api_key, api_version="2025-04-14")

    response = client.chat.completions.create(
        model="o4-mini",
        messages=[
            {
                "role": "user",
                "content": "What steps should I think about when writing my first Python API?",
            },
        ],
        max_completion_tokens=5000,
    )

    assert response is not None
