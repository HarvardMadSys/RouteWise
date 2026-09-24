"""Tests for the facade's dependency-free output-length estimator."""

from __future__ import annotations

import pytest

from llm_routewise._output_length import _OutputLengthEstimator


def test_default_until_a_fallback_threshold_is_reached() -> None:
    estimator = _OutputLengthEstimator()

    assert estimator.predict(128) == 500.0
    for index in range(4):
        assert estimator.update(128, 100 + index)

    assert estimator.predict(128) == 500.0
    assert estimator.global_count == 4


def test_bucket_mean_wins_at_five_matching_samples() -> None:
    estimator = _OutputLengthEstimator()
    outputs = [100, 200, 300, 400, 500]

    for output in outputs:
        estimator.update(8, output)

    assert estimator.predict(8) == pytest.approx(300.0)
    assert estimator.predict(16) == 500.0


def test_global_mean_is_used_at_twenty_samples() -> None:
    estimator = _OutputLengthEstimator()

    # Four observations in each distinct log2 bucket: no bucket is warm.
    for exponent in range(5):
        for _ in range(4):
            estimator.update(1 << exponent, 250)

    assert estimator.global_count == 20
    assert estimator.is_warmed_up
    assert estimator.predict(1 << 10) == 250.0


def test_bucket_mean_takes_precedence_over_warm_global_mean() -> None:
    estimator = _OutputLengthEstimator()
    for exponent in range(5):
        for _ in range(4):
            estimator.update(1 << exponent, 100)
    for _ in range(5):
        estimator.update(1024, 900)

    assert estimator.predict(1024) == 900.0
    assert estimator.predict(2048) == pytest.approx((20 * 100 + 5 * 900) / 25)


def test_nonpositive_outputs_do_not_train() -> None:
    estimator = _OutputLengthEstimator()

    assert estimator.update(100, 0) is False
    assert estimator.update(100, -10) is False
    assert estimator.global_count == 0
    assert estimator.predict(100) == 500.0


def test_reset_forgets_every_observation() -> None:
    estimator = _OutputLengthEstimator()
    for _ in range(5):
        estimator.update(8, 100)

    estimator.reset()

    assert estimator.global_count == 0
    assert not estimator.is_warmed_up
    assert estimator.predict(8) == 500.0


@pytest.mark.parametrize("bad_input", [-1, True, 1.5])
def test_input_token_validation(bad_input: object) -> None:
    estimator = _OutputLengthEstimator()

    with pytest.raises((TypeError, ValueError)):
        estimator.predict(bad_input)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_output", [True, float("inf"), float("nan"), "12"])
def test_output_token_validation(bad_output: object) -> None:
    estimator = _OutputLengthEstimator()

    with pytest.raises((TypeError, ValueError)):
        estimator.update(100, bad_output)  # type: ignore[arg-type]
