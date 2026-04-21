"""Pareto frontier sweep: parameterized V2+hedge and LP+hedge strategies.

Produces (cost, P99) points across parameter sweeps to visualize the
cost-latency trade-off space that RouteWise can span. Paper framing
(per Juncheng, 2026-04-17): front-page figure is a Pareto plot where
RouteWise instantiations dominate fixed baselines on at least one
dimension while matching them on the other.

Parameters swept:
  - cost_ratio:  hedge aggressiveness (lower -> more hedging, lower P99,
                 higher cost).
  - p50_band:    V2 near-best-P50 band width (wider -> cheaper provider
                 in band, potentially higher P99).

All baselines (cheapest_fixed, fastest_fixed, round_robin, oracle,
openrouter-equivalent) are single points on the same plot.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Reuse the existing synthetic infrastructure.
_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiment.data.schema import Request

from .providers import ShiftingProvider, SyntheticProvider
from .scenarios import ScenarioConfig
from .runner import (
    V2Router,
    OnlineLatencyRouter,
    SmartHedger,
    HedgingParams,
    HedgingStrategy,
    BackupSelectionMethod,
    select_backup,
    StrategyRun,
    _sample,
    _costs_dict,
    _warm_up_router,
    _cheapest_provider_name,
    _probe_providers,
    _run_cheapest_fixed,
    _run_fastest_fixed,
    _run_round_robin,
    _run_oracle_per_window,
)


# ---------------------------------------------------------------------------
# Parameterized hedge runners
# ---------------------------------------------------------------------------


def run_v2_hedge_param(
    scenario: ScenarioConfig,
    requests: list[Request],
    rng: np.random.Generator,
    slo_sec: float,
    cost_ratio: float,
    p50_band: float,
    dispatch_overhead_sec: float = 0.05,
    hedge_as_probe: bool = False,
    label: str | None = None,
) -> StrategyRun:
    """V2Router + SmartHedger with custom cost_ratio and p50_band.

    ``hedge_as_probe=True`` enables Explorer: when a hedge fires, the backup
    latency sample is fed back into the router profile. This is the
    mechanism Juncheng asked us to add.
    """
    costs = _costs_dict(scenario)
    pdict = {p.name: p for p in scenario.providers}

    router = V2Router(
        costs=costs,
        slo_sec=slo_sec,
        lp_update_interval=60.0,
        p50_band=p50_band,
    )
    hedger = SmartHedger(
        HedgingParams(
            strategy=HedgingStrategy.SMART_ECONOMIC,
            slo_sec=slo_sec,
            cost_ratio=cost_ratio,
            dispatch_overhead_sec=dispatch_overhead_sec,
            backup_method=BackupSelectionMethod.FASTEST,
        ),
        costs=costs,
    )

    t0 = float(requests[0].timestamp) if requests else 0.0
    _warm_up_router(router, scenario, t0, rng)
    fallback_name = _cheapest_provider_name(scenario)

    ttft_ms, cost_usd, provider_sel, timestamps = [], [], [], []
    hedged_flags: list[bool] = []

    for req in requests:
        t = float(req.timestamp)
        primary_name = router.route(t)
        if primary_name is None:
            primary_name = fallback_name

        backup_name = select_backup(
            BackupSelectionMethod.FASTEST,
            router.profiles,
            costs,
            primary_name,
            slo_sec,
            t,
        )
        if backup_name is None:
            backup_name = primary_name

        p_primary = pdict[primary_name]
        T_primary_ms, _ = _sample(p_primary, req.response_tokens, rng, t)

        p_backup = pdict[backup_name]
        T_backup_ms, _ = _sample(p_backup, req.response_tokens, rng, t)

        result = hedger.simulate_request(
            primary=primary_name,
            profiles=router.profiles,
            now=t,
            T_primary_sec=T_primary_ms / 1000.0,
            err_primary=None,
            T_backup_sec=T_backup_ms / 1000.0,
            err_backup=None,
            backup=backup_name,
        )

        final_ttft_ms = result.final_ttft_sec * 1000.0
        router.add_sample(primary_name, t, T_primary_ms)
        # Explorer: feed backup sample back to profile when hedge fires.
        if hedge_as_probe and result.hedged and backup_name != primary_name:
            router.add_sample(backup_name, t, T_backup_ms)
        _probe_providers(router, scenario, primary_name, t, rng)

        c = p_primary.cost_per_token * req.total_tokens
        if result.hedged:
            c += p_backup.cost_per_token * req.total_tokens

        ttft_ms.append(final_ttft_ms)
        cost_usd.append(c)
        provider_sel.append(primary_name)
        timestamps.append(t)
        hedged_flags.append(result.hedged)

    default_family = "v2_explorer" if hedge_as_probe else "v2_hedge"
    strat_label = label or f"{default_family}(cr={cost_ratio:.2f},band={p50_band:.2f})"
    return StrategyRun(
        strategy=strat_label,
        ttft_ms=np.array(ttft_ms),
        cost_usd=np.array(cost_usd),
        provider=provider_sel,
        timestamp=np.array(timestamps),
        hedge_triggered=np.array(hedged_flags, dtype=bool),
    )


def run_lp_hedge_param(
    scenario: ScenarioConfig,
    requests: list[Request],
    rng: np.random.Generator,
    slo_sec: float,
    cost_ratio: float,
    dispatch_overhead_sec: float = 0.05,
    hedge_as_probe: bool = False,
    label: str | None = None,
) -> StrategyRun:
    """OnlineLatencyRouter + SmartHedger with custom cost_ratio.

    ``hedge_as_probe=True`` enables Explorer: backup latency samples from
    triggered hedges are fed back into the router profile, improving
    diversification quality especially in heavy-tail regimes.
    """
    costs = _costs_dict(scenario)
    pdict = {p.name: p for p in scenario.providers}

    router = OnlineLatencyRouter(
        costs=costs,
        slo_sec=slo_sec,
        lp_update_interval=60.0,
    )
    hedger = SmartHedger(
        HedgingParams(
            strategy=HedgingStrategy.SMART_ECONOMIC,
            slo_sec=slo_sec,
            cost_ratio=cost_ratio,
            dispatch_overhead_sec=dispatch_overhead_sec,
            backup_method=BackupSelectionMethod.FASTEST,
        ),
        costs=costs,
    )

    t0 = float(requests[0].timestamp) if requests else 0.0
    _warm_up_router(router, scenario, t0, rng)
    fallback_name = _cheapest_provider_name(scenario)

    ttft_ms, cost_usd, provider_sel, timestamps = [], [], [], []
    hedged_flags: list[bool] = []

    for req in requests:
        t = float(req.timestamp)
        primary_name = router.route(t)
        if primary_name is None:
            primary_name = fallback_name

        backup_name = select_backup(
            BackupSelectionMethod.FASTEST,
            router.profiles,
            costs,
            primary_name,
            slo_sec,
            t,
        )
        if backup_name is None:
            backup_name = primary_name

        p_primary = pdict[primary_name]
        T_primary_ms, _ = _sample(p_primary, req.response_tokens, rng, t)

        p_backup = pdict[backup_name]
        T_backup_ms, _ = _sample(p_backup, req.response_tokens, rng, t)

        result = hedger.simulate_request(
            primary=primary_name,
            profiles=router.profiles,
            now=t,
            T_primary_sec=T_primary_ms / 1000.0,
            err_primary=None,
            T_backup_sec=T_backup_ms / 1000.0,
            err_backup=None,
            backup=backup_name,
        )

        final_ttft_ms = result.final_ttft_sec * 1000.0
        router.add_sample(primary_name, t, T_primary_ms)
        # Explorer: feed backup sample back to profile when hedge fires.
        if hedge_as_probe and result.hedged and backup_name != primary_name:
            router.add_sample(backup_name, t, T_backup_ms)
        _probe_providers(router, scenario, primary_name, t, rng)

        c = p_primary.cost_per_token * req.total_tokens
        if result.hedged:
            c += p_backup.cost_per_token * req.total_tokens

        ttft_ms.append(final_ttft_ms)
        cost_usd.append(c)
        provider_sel.append(primary_name)
        timestamps.append(t)
        hedged_flags.append(result.hedged)

    default_family = "lp_explorer" if hedge_as_probe else "lp_hedge"
    strat_label = label or f"{default_family}(cr={cost_ratio:.2f})"
    return StrategyRun(
        strategy=strat_label,
        ttft_ms=np.array(ttft_ms),
        cost_usd=np.array(cost_usd),
        provider=provider_sel,
        timestamp=np.array(timestamps),
        hedge_triggered=np.array(hedged_flags, dtype=bool),
    )


# ---------------------------------------------------------------------------
# Parameter sweeps
# ---------------------------------------------------------------------------


@dataclass
class ParetoPoint:
    """One point on the Pareto plot."""

    family: str       # "v2_hedge", "lp_hedge", "baseline"
    label: str        # Human readable label
    mean_cost: float
    p50_ms: float
    p99_ms: float
    slo_violation_rate: float  # at 2s SLO
    hedge_rate: float
    provider_fractions: dict


def _metrics(run: StrategyRun, slo_ms: float = 2000.0) -> ParetoPoint:
    """Compute standard metrics from a StrategyRun."""
    return ParetoPoint(
        family="",
        label=run.strategy,
        mean_cost=float(np.mean(run.cost_usd)),
        p50_ms=float(np.percentile(run.ttft_ms, 50)),
        p99_ms=float(np.percentile(run.ttft_ms, 99)),
        slo_violation_rate=float(np.mean(run.ttft_ms > slo_ms)),
        hedge_rate=float(np.mean(run.hedge_triggered)),
        provider_fractions={
            p: run.provider.count(p) / max(len(run.provider), 1)
            for p in sorted(set(run.provider))
        },
    )


def sweep_scenario(
    scenario: ScenarioConfig,
    requests: list[Request],
    seed: int = 42,
    cost_ratios: list[float] | None = None,
    p50_bands: list[float] | None = None,
) -> list[ParetoPoint]:
    """Run all baselines + V2-hedge sweep + LP-hedge sweep on one scenario."""
    if cost_ratios is None:
        cost_ratios = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    if p50_bands is None:
        p50_bands = [0.10, 0.30]  # one tight, one wide

    slo_sec = scenario.primary_slo_ms / 1000.0

    points: list[ParetoPoint] = []

    # Baselines
    for strategy_name, fn in [
        ("cheapest_fixed", _run_cheapest_fixed),
        ("fastest_fixed", _run_fastest_fixed),
        ("round_robin", _run_round_robin),
        ("oracle_per_window", _run_oracle_per_window),
    ]:
        rng = np.random.default_rng(seed)
        run = fn(scenario, requests, rng)
        p = _metrics(run)
        p.family = "baseline"
        p.label = strategy_name
        points.append(p)

    # V2 + hedge parameter sweep (no explorer)
    for cr in cost_ratios:
        for band in p50_bands:
            rng = np.random.default_rng(seed)
            run = run_v2_hedge_param(
                scenario, requests, rng, slo_sec,
                cost_ratio=cr, p50_band=band,
                hedge_as_probe=False,
            )
            p = _metrics(run)
            p.family = "v2_hedge"
            p.label = f"v2_hedge(cr={cr:.2f},band={band:.2f})"
            points.append(p)

    # V2 + Explorer (hedge-as-probe) sweep — Juncheng's optimization.
    for cr in cost_ratios:
        for band in p50_bands:
            rng = np.random.default_rng(seed)
            run = run_v2_hedge_param(
                scenario, requests, rng, slo_sec,
                cost_ratio=cr, p50_band=band,
                hedge_as_probe=True,
            )
            p = _metrics(run)
            p.family = "v2_explorer"
            p.label = f"v2_explorer(cr={cr:.2f},band={band:.2f})"
            points.append(p)

    # LP + hedge parameter sweep (no explorer)
    for cr in cost_ratios:
        rng = np.random.default_rng(seed)
        run = run_lp_hedge_param(
            scenario, requests, rng, slo_sec, cost_ratio=cr,
            hedge_as_probe=False,
        )
        p = _metrics(run)
        p.family = "lp_hedge"
        p.label = f"lp_hedge(cr={cr:.2f})"
        points.append(p)

    # LP + Explorer sweep.
    for cr in cost_ratios:
        rng = np.random.default_rng(seed)
        run = run_lp_hedge_param(
            scenario, requests, rng, slo_sec, cost_ratio=cr,
            hedge_as_probe=True,
        )
        p = _metrics(run)
        p.family = "lp_explorer"
        p.label = f"lp_explorer(cr={cr:.2f})"
        points.append(p)

    return points


def pareto_front(points: list[ParetoPoint], x_key: str = "mean_cost", y_key: str = "p99_ms") -> list[ParetoPoint]:
    """Return the Pareto-optimal subset (lower x AND lower y)."""
    def _get(p, k):
        return getattr(p, k)

    non_dominated: list[ParetoPoint] = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            if _get(q, x_key) <= _get(p, x_key) and _get(q, y_key) <= _get(p, y_key):
                if _get(q, x_key) < _get(p, x_key) or _get(q, y_key) < _get(p, y_key):
                    dominated = True
                    break
        if not dominated:
            non_dominated.append(p)
    return sorted(non_dominated, key=lambda p: _get(p, x_key))
