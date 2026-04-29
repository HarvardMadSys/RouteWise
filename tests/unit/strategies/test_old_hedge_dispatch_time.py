"""Regression tests for legacy oldhedge dispatch-time accounting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import experiments.tiered_capacity.lp_budget_eval as lp_eval
from rwsim.schemas import Request
from rwsim.world.capacity import ConcurrencyState, ProviderTier
from rwsim.world.distributions import LogNormal
from rwsim.world.providers import TieredProvider
from rwsim.world.scenarios import ScenarioConfig

if TYPE_CHECKING:
    import pytest


NOW = 10.0
TTFT = LogNormal(0.0, 0.1)
TPS = LogNormal(4.0, 0.1)


def _api_provider(name: str) -> TieredProvider:
    return TieredProvider(
        name=name,
        cost_per_token=0.001,
        ttft_dist=TTFT,
        tps_dist=TPS,
        tier=ProviderTier.S_A,
    )


def _concurrency_provider(name: str, concurrency: ConcurrencyState) -> TieredProvider:
    return TieredProvider(
        name=name,
        cost_per_token=0.0,
        ttft_dist=TTFT,
        tps_dist=TPS,
        tier=ProviderTier.S_C,
        concurrency=concurrency,
    )


def _scenario(providers: list[TieredProvider]) -> ScenarioConfig:
    return ScenarioConfig(
        name="oldhedge-dispatch-time",
        description="legacy hedge dispatch-time test",
        providers=providers,
        primary_slo_ms=1000.0,
    )


def _request() -> Request:
    return Request(
        id=1,
        timestamp=NOW,
        request_tokens=10,
        response_tokens=10,
        total_tokens=20,
    )


def test_oldhedge_accounts_backup_at_wait_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    primary = _api_provider("api-primary")
    backup_concurrency = ConcurrencyState(limit=1)
    backup = _concurrency_provider("backup-sc", backup_concurrency)
    scenario = _scenario([primary, backup])

    ttft_samples = iter([1200.0, 50.0])
    service_samples = iter([0.2, 0.3])
    monkeypatch.setattr(
        lp_eval,
        "_sample_ttft",
        lambda provider, rng, now: next(ttft_samples),
    )
    monkeypatch.setattr(
        lp_eval,
        "_sample_service_time",
        lambda provider, rng, now, response_tokens, ttft_ms=None: next(service_samples),
    )

    final_ttft_ms, _, hedged, observed_samples, _ = lp_eval._apply_existing_hedge(
        scenario,
        scenario.providers,
        primary,
        _request(),
        NOW,
        np.random.default_rng(0),
        {},
        U=1.0,
        L=1e-3,
    )

    dispatch_time = NOW + scenario.primary_slo_ms * 0.5 / 1000.0
    assert hedged
    assert final_ttft_ms == 550.0
    assert backup_concurrency.active == [(dispatch_time, dispatch_time + 0.3, 10_000_001)]
    assert observed_samples == [
        ("api-primary", NOW + 1.2, 1200.0),
        ("backup-sc", dispatch_time + 0.05, 50.0),
    ]


def test_oldhedge_does_not_use_backup_unavailable_at_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = _api_provider("api-primary")
    backup_concurrency = ConcurrencyState(limit=1)
    dispatch_time = NOW + 0.5
    backup_concurrency.active.append((dispatch_time - 0.1, dispatch_time + 1.0, 99))
    backup = _concurrency_provider("backup-sc", backup_concurrency)
    scenario = _scenario([primary, backup])

    monkeypatch.setattr(lp_eval, "_sample_ttft", lambda provider, rng, now: 1200.0)
    monkeypatch.setattr(
        lp_eval,
        "_sample_service_time",
        lambda provider, rng, now, response_tokens, ttft_ms=None: 0.2,
    )

    final_ttft_ms, _, hedged, observed_samples, _ = lp_eval._apply_existing_hedge(
        scenario,
        scenario.providers,
        primary,
        _request(),
        NOW,
        np.random.default_rng(0),
        {},
        U=1.0,
        L=1e-3,
    )

    assert not hedged
    assert final_ttft_ms == 1200.0
    assert observed_samples == [("api-primary", NOW + 1.2, 1200.0)]
    assert backup_concurrency.active == [(dispatch_time - 0.1, dispatch_time + 1.0, 99)]
