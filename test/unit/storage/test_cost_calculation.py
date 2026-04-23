"""Unit tests for cost calculation helpers."""

from __future__ import annotations

import pytest

from serving.storage.database import calculate_cost


def test_cost_calculation_basic() -> None:
    usage = {"prompt_tokens": 1000, "completion_tokens": 500}
    pricing = {"prompt": "0.15", "completion": "1.25"}

    cost = calculate_cost(usage, pricing)

    assert cost == pytest.approx(0.000775)


def test_cost_calculation_with_cache_and_reasoning() -> None:
    usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "reasoning_tokens": 200,
        "cache_read_tokens": 5000,
    }
    pricing = {
        "prompt": "0.28",
        "completion": "0.42",
        "input_cache_reads": "0.028",
    }

    cost = calculate_cost(usage, pricing)

    expected = (
        (1000 * 0.28 / 1_000_000)
        + (500 * 0.42 / 1_000_000)
        + (200 * 0.42 / 1_000_000)
        + (5000 * 0.028 / 1_000_000)
    )
    assert cost == pytest.approx(expected)


def test_cost_calculation_missing_pricing_returns_none() -> None:
    usage = {"prompt_tokens": 1000}

    assert calculate_cost(usage, None) is None


def test_cost_calculation_invalid_types_returns_none() -> None:
    usage = {"prompt_tokens": "invalid"}
    pricing = {"prompt": "0.15"}

    assert calculate_cost(usage, pricing) is None


def test_cost_calculation_zero_pricing() -> None:
    usage = {"prompt_tokens": 1000, "completion_tokens": 500}
    pricing = {"prompt": "0", "completion": "0"}

    assert calculate_cost(usage, pricing) == 0.0
