from __future__ import annotations

import pytest

from experiments.offline_stage.value_estimators import BucketMeanOutputPredictor
from experiments.simulation.common import (
    SingleModelOutputPredictor,
    _build_workload_predictor,
    _normalize_predictor_arg,
)
from rwsim.schemas import Request


def _request(
    request_id: int,
    *,
    request_tokens: int,
    response_tokens: int | None = None,
    model: str | None = None,
) -> Request:
    return Request(
        id=request_id,
        timestamp=request_id,
        request_tokens=request_tokens,
        response_tokens=response_tokens,
        model=model,
    )


def test_bucket_mean_uses_matching_input_length_bucket() -> None:
    predictor = BucketMeanOutputPredictor(
        min_samples_bucket=2,
        min_samples_global=4,
        default_output_tokens=999.0,
    )
    target = _request(10, request_tokens=128, model="different-label")

    assert predictor.predict(target).tokens == pytest.approx(999.0)
    assert not predictor.predict(target).is_warmed_up

    predictor.update(_request(1, request_tokens=128, response_tokens=100, model="a"))
    assert predictor.predict(target).tokens == pytest.approx(999.0)

    predictor.update(_request(2, request_tokens=128, response_tokens=140, model="b"))
    prediction = predictor.predict(target)

    assert prediction.tokens == pytest.approx(120.0)
    assert not hasattr(prediction, "q10")
    assert not hasattr(prediction, "q50")
    assert not hasattr(prediction, "q90")
    assert prediction.is_warmed_up


def test_bucket_mean_falls_back_to_global_mean_for_sparse_bucket() -> None:
    predictor = BucketMeanOutputPredictor(
        min_samples_bucket=3,
        min_samples_global=3,
        default_output_tokens=500.0,
    )
    predictor.update(_request(1, request_tokens=64, response_tokens=30))
    predictor.update(_request(2, request_tokens=128, response_tokens=60))
    predictor.update(_request(3, request_tokens=256, response_tokens=90))

    sparse_bucket_request = _request(4, request_tokens=4096)

    assert predictor.predict(sparse_bucket_request).tokens == pytest.approx(60.0)
    assert predictor.predict(sparse_bucket_request).is_warmed_up


def test_simulation_predictor_factory_supports_bucket_mean() -> None:
    spec = _normalize_predictor_arg("bucket_mean")

    assert spec == {"kind": "bucket_mean"}
    predictor = _build_workload_predictor(spec, requests=[])

    assert isinstance(predictor, BucketMeanOutputPredictor)
    assert not isinstance(predictor, SingleModelOutputPredictor)
