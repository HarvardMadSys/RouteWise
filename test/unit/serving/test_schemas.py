from __future__ import annotations

import pytest
from pydantic import ValidationError

from serving.schemas import ChatCompletionRequest, ModelList


@pytest.mark.unit
def test_chat_completion_request_basic_and_extra_ignored():
    req = ChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
            ],
            "temperature": 0.7,
            "extra_field": "ignored",
        }
    )
    assert req.model == "m"
    assert req.temperature == 0.7
    # Ensure extra not present on instance
    assert not hasattr(req, "extra_field")


@pytest.mark.unit
def test_chat_completion_request_invalid_ranges():
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            {"model": "m", "messages": [{"role": "user", "content": "u"}], "temperature": 5}
        )
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            {"model": "m", "messages": [{"role": "user", "content": "u"}], "top_p": 2}
        )


@pytest.mark.unit
def test_model_list_schema_roundtrip():
    data = {
        "object": "list",
        "data": [
            {
                "id": "m",
                "name": "M",
                "object": "model",
                "created": 1,
                "owned_by": "p",
                "input_modalities": ["text"],
                "output_modalities": ["text"],
                "quantization": "bf16",
                "context_length": 1,
                "max_output_length": 1,
                "pricing": {"prompt": "0", "completion": "0"},
                "supported_sampling_parameters": ["temperature"],
                "supported_features": ["tools"],
            }
        ],
    }
    obj = ModelList.model_validate(data)
    assert obj.object == "list"
    assert obj.data[0].id == "m"
