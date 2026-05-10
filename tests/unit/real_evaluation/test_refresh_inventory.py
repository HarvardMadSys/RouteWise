"""Tests for real-eval inventory refresh helpers."""

from __future__ import annotations

from experiments.real_evaluation.refresh_inventory import (
    dedup_by_cheapest,
    to_provider_entries,
)


def test_refresh_inventory_writes_cached_input_price_when_available() -> None:
    endpoints = [
        {
            "provider_name": "Cached Provider",
            "pricing": {
                "prompt": "0.0000001",
                "input_cache_read": "0.00000002",
                "completion": "0.000001",
            },
            "uptime_last_1d": 99.0,
            "tag": "test",
        }
    ]

    entries = to_provider_entries(dedup_by_cheapest(endpoints), "test/model")

    assert entries[0]["cached_input_price_per_m"] == 0.02


def test_refresh_inventory_omits_zero_cached_input_price() -> None:
    endpoints = [
        {
            "provider_name": "No Cache Provider",
            "pricing": {
                "prompt": "0.0000001",
                "input_cache_read": "0",
                "completion": "0.000001",
            },
            "uptime_last_1d": 99.0,
            "tag": "test",
        }
    ]

    entries = to_provider_entries(dedup_by_cheapest(endpoints), "test/model")

    assert "cached_input_price_per_m" not in entries[0]
