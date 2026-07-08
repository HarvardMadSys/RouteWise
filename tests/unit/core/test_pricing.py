"""Tests for the shared cache-aware token-pricing arithmetic."""

from __future__ import annotations

import pytest

from rwsim.core.pricing import (
    cache_discounted_cost_per_million_usd,
    cache_discounted_token_cost_usd,
    capped_cached_input_tokens,
)


def test_capped_cached_input_tokens() -> None:
    assert capped_cached_input_tokens(1000, None) == 0
    assert capped_cached_input_tokens(1000, 400) == 400
    assert capped_cached_input_tokens(1000, 5000) == 1000
    assert capped_cached_input_tokens(1000, -3) == 0
    assert capped_cached_input_tokens(0, 400) == 0


def test_per_token_cost_matches_manual_formula() -> None:
    cost = cache_discounted_token_cost_usd(
        prompt_tokens=1000,
        completion_tokens=200,
        cached_input_tokens=400,
        input_price_per_token=3e-6,
        cached_input_price_per_token=3e-7,
        output_price_per_token=15e-6,
    )
    expected = 3e-6 * 600 + 3e-7 * 400 + 15e-6 * 200
    assert cost == expected


def test_per_token_cost_caps_cached_at_prompt() -> None:
    cost = cache_discounted_token_cost_usd(
        prompt_tokens=100,
        completion_tokens=0,
        cached_input_tokens=500,
        input_price_per_token=1e-6,
        cached_input_price_per_token=1e-7,
        output_price_per_token=2e-6,
    )
    assert cost == pytest.approx(1e-7 * 100)


def test_per_million_cost_matches_real_eval_formula() -> None:
    cost = cache_discounted_cost_per_million_usd(
        prompt_tokens=1000,
        completion_tokens=200,
        cached_input_tokens=400,
        input_price_per_m=3.0,
        cached_input_price_per_m=0.3,
        output_price_per_m=15.0,
    )
    expected = (3.0 * 600 + 0.3 * 400 + 15.0 * 200) / 1_000_000.0
    assert cost == expected


def test_per_million_cost_zero_cache_is_linear_pricing() -> None:
    cost = cache_discounted_cost_per_million_usd(
        prompt_tokens=1000,
        completion_tokens=200,
        cached_input_tokens=0,
        input_price_per_m=3.0,
        cached_input_price_per_m=0.3,
        output_price_per_m=15.0,
    )
    assert cost == pytest.approx((3.0 * 1000 + 15.0 * 200) / 1_000_000.0)
