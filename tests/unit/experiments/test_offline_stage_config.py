"""Offline-stage config should derive shared plan facts from canonical YAML."""

from __future__ import annotations

from experiments.offline_stage.config import load_default_config


def test_offline_stage_chutes_facts_come_from_subscription_plans_yaml():
    config = load_default_config()

    provider = config.get_provider("chutes-subscription")
    assert provider.monthly_fee == 20.0
    assert provider.daily_quota == 5000

    payload = config.to_dict()
    chutes = payload["subscriptions"]["chutes"]
    assert chutes["subscription_plan"] == "chutes"
    assert chutes["monthly_fee"] == 20.0
    assert chutes["daily_quota"] == 5000
    assert chutes["quota_window_sec"] == 86400.0
