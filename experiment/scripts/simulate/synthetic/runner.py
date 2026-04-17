"""Simulation runner: executes each routing strategy against a synthetic scenario.

Strategies
----------
cheapest_fixed     Always route to the provider with the lowest cost/token.
fastest_fixed      Always route to the provider with the lowest true P50 (hindsight).
round_robin        Rotate uniformly across providers.
lp_mix             OnlineLatencyRouter: LP-based mixing, no hedging.
lp_hedge           OnlineLatencyRouter routing + SmartHedger (SMART_ECONOMIC).
v2_only            V2Router (P50 rank + near-best band), no hedging.
v2_p50_hedge       V2Router + SmartHedger (SMART_ECONOMIC).
oracle_per_window  Hindsight best-P50 provider per 15-min window.

Separation rationale
--------------------
lp_mix vs lp_hedge   : does hedging help on top of LP routing?
v2_only vs v2_p50_hedge : does hedging help on top of V2 routing?
lp_hedge vs v2_p50_hedge : does the routing algorithm matter when both hedge?

All strategies share the same workload (list[Request]) and random seed, so
results are directly comparable. Samples are fed back into adaptive routers so
they accumulate realistic latency profiles over simulated time.
"""

from __future__ import annotations

import importlib.util as _ilu
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project path setup — load strategy modules without triggering __init__.py
# ---------------------------------------------------------------------------

_project_root = str(Path(__file__).resolve().parents[4])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_strategies_dir = os.path.join(_project_root, "experiment", "strategies")


def _load_mod(name: str, filename: str):
    path = os.path.join(_strategies_dir, filename)
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_olr_mod = _load_mod("_synth_olr", "online_latency_router.py")
_v2_mod = _load_mod("_synth_v2", "v2_router.py")
_hedge_mod = _load_mod("_synth_hedge", "smart_hedging.py")

OnlineLatencyRouter = _olr_mod.OnlineLatencyRouter
V2Router = _v2_mod.V2Router
SmartHedger = _hedge_mod.SmartHedger
HedgingParams = _hedge_mod.HedgingParams
HedgingStrategy = _hedge_mod.HedgingStrategy
BackupSelectionMethod = _hedge_mod.BackupSelectionMethod
select_backup = _hedge_mod.select_backup

from experiment.data.schema import Request  # noqa: E402

from .providers import ShiftingProvider, SyntheticProvider  # noqa: E402
from .scenarios import ScenarioConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

STRATEGIES = [
    "cheapest_fixed",
    "fastest_fixed",
    "round_robin",
    "lp_mix",
    "lp_hedge",
    "v2_only",
    "v2_p50_hedge",
    "oracle_per_window",
]

# Tokens assumed when computing the router's fixed cost-per-request value.
_TYPICAL_TOKENS = 200

# In production, the system sends periodic lightweight probe requests to every
# provider so profiles stay fresh. We simulate this at 5% of the request rate
# per non-primary provider (one probe per 20 requests on average).
_PROBE_RATE = 0.05


@dataclass
class StrategyRun:
    """Per-request results for one strategy on one scenario."""

    strategy: str
    ttft_ms: np.ndarray        # shape (n_requests,)
    cost_usd: np.ndarray       # shape (n_requests,)
    provider: list[str]        # shape (n_requests,)
    timestamp: np.ndarray      # shape (n_requests,) — simulated seconds
    hedge_triggered: np.ndarray  # shape (n_requests,) bool

    # ------------------------------------------------------------------ #
    # Derived statistics                                                   #
    # ------------------------------------------------------------------ #

    def slo_violation_rate(self, slo_ms: float) -> float:
        return float(np.mean(self.ttft_ms > slo_ms))

    def mean_cost_usd(self) -> float:
        return float(np.mean(self.cost_usd))

    def p50_ms(self) -> float:
        return float(np.percentile(self.ttft_ms, 50))

    def p99_ms(self) -> float:
        return float(np.percentile(self.ttft_ms, 99))

    def hedge_rate(self) -> float:
        return float(np.mean(self.hedge_triggered))

    def provider_fractions(self) -> dict[str, float]:
        total = len(self.provider)
        if total == 0:
            return {}
        names = set(self.provider)
        return {n: self.provider.count(n) / total for n in sorted(names)}

    def provider_fractions_over_time(
        self, window_sec: float = 300.0
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Compute rolling provider selection fractions in time windows.

        Returns (window_midpoints, {provider: fraction_array}).
        """
        if len(self.timestamp) == 0:
            return np.array([]), {}

        t_min = float(self.timestamp[0])
        t_max = float(self.timestamp[-1])
        edges = np.arange(t_min, t_max + window_sec, window_sec)
        mids = (edges[:-1] + edges[1:]) / 2.0

        provider_names = sorted(set(self.provider))
        fracs: dict[str, list[float]] = {n: [] for n in provider_names}

        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (self.timestamp >= lo) & (self.timestamp < hi)
            window_providers = [self.provider[i] for i in range(len(self.provider)) if mask[i]]
            n_window = len(window_providers)
            for pname in provider_names:
                if n_window == 0:
                    fracs[pname].append(0.0)
                else:
                    fracs[pname].append(window_providers.count(pname) / n_window)

        return mids, {n: np.array(fracs[n]) for n in provider_names}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _costs_dict(scenario: ScenarioConfig) -> dict[str, float]:
    """Fixed cost-per-request dict for router initialisation."""
    return {p.name: p.cost_per_token * _TYPICAL_TOKENS for p in scenario.providers}


def _sample(
    provider: SyntheticProvider,
    output_tokens: int,
    rng: np.random.Generator,
    current_time: float,
) -> tuple[float, float]:
    """Sample (ttft_ms, e2e_ms) from provider at current_time."""
    return provider.sample_request(output_tokens, rng, current_time)


def _true_p50(provider: SyntheticProvider, t: float) -> float:
    """Analytical P50 TTFT in ms at simulated time t."""
    return provider.true_p50_ms(t)


def _oracle_best_at(providers: list[SyntheticProvider], t: float) -> str:
    """Provider with the lowest TRUE P50 at simulated time t."""
    return min(providers, key=lambda p: _true_p50(p, t)).name


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


def _run_cheapest_fixed(
    scenario: ScenarioConfig,
    requests: list[Request],
    rng: np.random.Generator,
) -> StrategyRun:
    cheapest = min(scenario.providers, key=lambda p: p.cost_per_token)
    ttft_ms, cost_usd, provider, timestamps = [], [], [], []

    for req in requests:
        t = float(req.timestamp)
        tm, _ = _sample(cheapest, req.response_tokens, rng, t)
        ttft_ms.append(tm)
        cost_usd.append(cheapest.cost_per_token * req.total_tokens)
        provider.append(cheapest.name)
        timestamps.append(t)

    return StrategyRun(
        strategy="cheapest_fixed",
        ttft_ms=np.array(ttft_ms),
        cost_usd=np.array(cost_usd),
        provider=provider,
        timestamp=np.array(timestamps),
        hedge_triggered=np.zeros(len(ttft_ms), dtype=bool),
    )


def _run_fastest_fixed(
    scenario: ScenarioConfig,
    requests: list[Request],
    rng: np.random.Generator,
) -> StrategyRun:
    """Route all requests to the provider with the lowest hindsight P50.

    For ShiftingProviders, we take the time-averaged P50 across the full
    simulation window (sampling half the requests from each phase).
    """
    seed_rng = np.random.default_rng(0)
    half = 5000

    best_p50: dict[str, float] = {}
    for p in scenario.providers:
        if isinstance(p, ShiftingProvider):
            s_before = p.ttft_dist.sample(seed_rng, half)
            s_after = p.ttft_dist_after.sample(seed_rng, half)
            best_p50[p.name] = float(np.median(np.concatenate([s_before, s_after])))
        else:
            best_p50[p.name] = float(np.median(p.ttft_dist.sample(seed_rng, half * 2)))

    fastest_name = min(best_p50, key=lambda n: best_p50[n])
    fastest = next(p for p in scenario.providers if p.name == fastest_name)

    ttft_ms, cost_usd, provider, timestamps = [], [], [], []
    for req in requests:
        t = float(req.timestamp)
        tm, _ = _sample(fastest, req.response_tokens, rng, t)
        ttft_ms.append(tm)
        cost_usd.append(fastest.cost_per_token * req.total_tokens)
        provider.append(fastest.name)
        timestamps.append(t)

    return StrategyRun(
        strategy="fastest_fixed",
        ttft_ms=np.array(ttft_ms),
        cost_usd=np.array(cost_usd),
        provider=provider,
        timestamp=np.array(timestamps),
        hedge_triggered=np.zeros(len(ttft_ms), dtype=bool),
    )


def _run_round_robin(
    scenario: ScenarioConfig,
    requests: list[Request],
    rng: np.random.Generator,
) -> StrategyRun:
    names = [p.name for p in scenario.providers]
    pdict = {p.name: p for p in scenario.providers}
    n = len(names)

    ttft_ms, cost_usd, provider, timestamps = [], [], [], []
    for idx, req in enumerate(requests):
        t = float(req.timestamp)
        pname = names[idx % n]
        p = pdict[pname]
        tm, _ = _sample(p, req.response_tokens, rng, t)
        ttft_ms.append(tm)
        cost_usd.append(p.cost_per_token * req.total_tokens)
        provider.append(pname)
        timestamps.append(t)

    return StrategyRun(
        strategy="round_robin",
        ttft_ms=np.array(ttft_ms),
        cost_usd=np.array(cost_usd),
        provider=provider,
        timestamp=np.array(timestamps),
        hedge_triggered=np.zeros(len(ttft_ms), dtype=bool),
    )


def _warm_up_router(router, scenario: ScenarioConfig, t0: float, rng: np.random.Generator) -> None:
    """Seed router profiles with samples spread over the 14-minute window before t0.

    This simulates the baseline probing data available before the simulation
    window opens. Samples are distributed evenly so they don't all expire
    simultaneously, and use the provider's true P50 (± noise) as latency.
    """
    N_PER_PROVIDER = 50
    window = 14 * 60  # 14 minutes of history (stays within the 15-min rolling window)

    for p in scenario.providers:
        for i in range(N_PER_PROVIDER):
            # Spread timestamps uniformly over the 14-min warm-up window.
            t_sample = t0 - window + (i / N_PER_PROVIDER) * window
            # Sample from the true distribution at that time.
            ttft_ms = p.sample_ttft(rng, t_sample)
            router.add_sample(p.name, t_sample, ttft_ms)


def _probe_providers(
    router,
    scenario: ScenarioConfig,
    primary_name: str,
    t: float,
    rng: np.random.Generator,
) -> None:
    """Send lightweight probe samples to non-primary providers.

    Simulates production-level background probing that keeps latency profiles
    fresh for providers not currently receiving production traffic.
    """
    for p in scenario.providers:
        if p.name != primary_name and rng.random() < _PROBE_RATE:
            probe_ttft = p.sample_ttft(rng, t)
            router.add_sample(p.name, t, probe_ttft)


def _run_lp_mix(
    scenario: ScenarioConfig,
    requests: list[Request],
    rng: np.random.Generator,
    slo_sec: float,
) -> StrategyRun:
    costs = _costs_dict(scenario)
    pdict = {p.name: p for p in scenario.providers}

    router = OnlineLatencyRouter(
        costs=costs,
        slo_sec=slo_sec,
        lp_update_interval=60.0,
    )

    t0 = float(requests[0].timestamp) if requests else 0.0
    _warm_up_router(router, scenario, t0, rng)

    ttft_ms, cost_usd, provider_sel, timestamps = [], [], [], []

    for req in requests:
        t = float(req.timestamp)
        pname = router.route(t)
        if pname is None:
            pname = scenario.providers[0].name

        p = pdict[pname]
        tm, _ = _sample(p, req.response_tokens, rng, t)

        router.add_sample(pname, t, tm)
        _probe_providers(router, scenario, pname, t, rng)

        ttft_ms.append(tm)
        cost_usd.append(p.cost_per_token * req.total_tokens)
        provider_sel.append(pname)
        timestamps.append(t)

    return StrategyRun(
        strategy="lp_mix",
        ttft_ms=np.array(ttft_ms),
        cost_usd=np.array(cost_usd),
        provider=provider_sel,
        timestamp=np.array(timestamps),
        hedge_triggered=np.zeros(len(ttft_ms), dtype=bool),
    )


def _run_v2_p50_hedge(
    scenario: ScenarioConfig,
    requests: list[Request],
    rng: np.random.Generator,
    slo_sec: float,
) -> StrategyRun:
    costs = _costs_dict(scenario)
    pdict = {p.name: p for p in scenario.providers}

    router = V2Router(
        costs=costs,
        slo_sec=slo_sec,
        lp_update_interval=60.0,
    )
    hedger = SmartHedger(
        HedgingParams(
            strategy=HedgingStrategy.SMART_ECONOMIC,
            slo_sec=slo_sec,
            cost_ratio=0.1,
            dispatch_overhead_sec=0.05,
            backup_method=BackupSelectionMethod.FASTEST,
        ),
        costs=costs,
    )

    t0 = float(requests[0].timestamp) if requests else 0.0
    _warm_up_router(router, scenario, t0, rng)

    ttft_ms, cost_usd, provider_sel, timestamps = [], [], [], []
    hedged_flags: list[bool] = []

    for req in requests:
        t = float(req.timestamp)

        primary_name = router.route(t)
        if primary_name is None:
            primary_name = scenario.providers[0].name

        backup_name = select_backup(
            BackupSelectionMethod.FASTEST,
            router.profiles,
            costs,
            primary_name,
            slo_sec,
            t,
        )
        # Ensure backup is never None — hedger always needs a valid name.
        if backup_name is None:
            backup_name = primary_name

        # Sample primary and pre-sample backup (only charged if hedge triggers).
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

        # Feed primary sample back to router and probe non-primaries.
        router.add_sample(primary_name, t, T_primary_ms)
        _probe_providers(router, scenario, primary_name, t, rng)

        # Compute cost: both are charged if hedge triggered.
        c = p_primary.cost_per_token * req.total_tokens
        if result.hedged:
            c += p_backup.cost_per_token * req.total_tokens

        ttft_ms.append(final_ttft_ms)
        cost_usd.append(c)
        provider_sel.append(primary_name)
        timestamps.append(t)
        hedged_flags.append(result.hedged)

    return StrategyRun(
        strategy="v2_p50_hedge",
        ttft_ms=np.array(ttft_ms),
        cost_usd=np.array(cost_usd),
        provider=provider_sel,
        timestamp=np.array(timestamps),
        hedge_triggered=np.array(hedged_flags, dtype=bool),
    )


def _make_hedger(costs: dict[str, float], slo_sec: float) -> SmartHedger:
    return SmartHedger(
        HedgingParams(
            strategy=HedgingStrategy.SMART_ECONOMIC,
            slo_sec=slo_sec,
            cost_ratio=0.1,
            dispatch_overhead_sec=0.05,
            backup_method=BackupSelectionMethod.FASTEST,
        ),
        costs=costs,
    )


def _run_v2_only(
    scenario: ScenarioConfig,
    requests: list[Request],
    rng: np.random.Generator,
    slo_sec: float,
) -> StrategyRun:
    """V2Router routing decisions only — no hedging."""
    costs = _costs_dict(scenario)
    pdict = {p.name: p for p in scenario.providers}

    router = V2Router(costs=costs, slo_sec=slo_sec, lp_update_interval=60.0)

    t0 = float(requests[0].timestamp) if requests else 0.0
    _warm_up_router(router, scenario, t0, rng)

    ttft_ms, cost_usd, provider_sel, timestamps = [], [], [], []

    for req in requests:
        t = float(req.timestamp)
        pname = router.route(t)
        if pname is None:
            pname = scenario.providers[0].name

        p = pdict[pname]
        tm, _ = _sample(p, req.response_tokens, rng, t)

        router.add_sample(pname, t, tm)
        _probe_providers(router, scenario, pname, t, rng)

        ttft_ms.append(tm)
        cost_usd.append(p.cost_per_token * req.total_tokens)
        provider_sel.append(pname)
        timestamps.append(t)

    return StrategyRun(
        strategy="v2_only",
        ttft_ms=np.array(ttft_ms),
        cost_usd=np.array(cost_usd),
        provider=provider_sel,
        timestamp=np.array(timestamps),
        hedge_triggered=np.zeros(len(ttft_ms), dtype=bool),
    )


def _run_lp_hedge(
    scenario: ScenarioConfig,
    requests: list[Request],
    rng: np.random.Generator,
    slo_sec: float,
) -> StrategyRun:
    """LP routing (OnlineLatencyRouter) + SmartHedger (SMART_ECONOMIC)."""
    costs = _costs_dict(scenario)
    pdict = {p.name: p for p in scenario.providers}

    router = OnlineLatencyRouter(costs=costs, slo_sec=slo_sec, lp_update_interval=60.0)
    hedger = _make_hedger(costs, slo_sec)

    t0 = float(requests[0].timestamp) if requests else 0.0
    _warm_up_router(router, scenario, t0, rng)

    ttft_ms, cost_usd, provider_sel, timestamps = [], [], [], []
    hedged_flags: list[bool] = []

    for req in requests:
        t = float(req.timestamp)
        primary_name = router.route(t)
        if primary_name is None:
            primary_name = scenario.providers[0].name

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
        _probe_providers(router, scenario, primary_name, t, rng)

        c = p_primary.cost_per_token * req.total_tokens
        if result.hedged:
            c += p_backup.cost_per_token * req.total_tokens

        ttft_ms.append(final_ttft_ms)
        cost_usd.append(c)
        provider_sel.append(primary_name)
        timestamps.append(t)
        hedged_flags.append(result.hedged)

    return StrategyRun(
        strategy="lp_hedge",
        ttft_ms=np.array(ttft_ms),
        cost_usd=np.array(cost_usd),
        provider=provider_sel,
        timestamp=np.array(timestamps),
        hedge_triggered=np.array(hedged_flags, dtype=bool),
    )


def _run_oracle_per_window(
    scenario: ScenarioConfig,
    requests: list[Request],
    rng: np.random.Generator,
) -> StrategyRun:
    """Hindsight oracle: picks the TRUE best provider per 15-min window."""
    pdict = {p.name: p for p in scenario.providers}
    WINDOW = 15 * 60  # seconds

    ttft_ms, cost_usd, provider_sel, timestamps = [], [], [], []

    for req in requests:
        t = float(req.timestamp)
        # Window midpoint for evaluating "true" best provider.
        window_start = (t // WINDOW) * WINDOW
        window_mid = window_start + WINDOW / 2.0

        best_name = _oracle_best_at(scenario.providers, window_mid)
        p = pdict[best_name]
        tm, _ = _sample(p, req.response_tokens, rng, t)

        ttft_ms.append(tm)
        cost_usd.append(p.cost_per_token * req.total_tokens)
        provider_sel.append(best_name)
        timestamps.append(t)

    return StrategyRun(
        strategy="oracle_per_window",
        ttft_ms=np.array(ttft_ms),
        cost_usd=np.array(cost_usd),
        provider=provider_sel,
        timestamp=np.array(timestamps),
        hedge_triggered=np.zeros(len(ttft_ms), dtype=bool),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_strategy(
    scenario: ScenarioConfig,
    requests: list[Request],
    strategy: str,
    seed: int = 42,
) -> StrategyRun:
    """Simulate one routing strategy on a pre-generated workload.

    Args:
        scenario: Scenario configuration (providers + parameters).
        requests: Pre-generated list of Request objects.
        strategy: One of the STRATEGIES list.
        seed: RNG seed for latency sampling.

    Returns:
        StrategyRun with per-request results.
    """
    rng = np.random.default_rng(seed)
    slo_sec = scenario.primary_slo_ms / 1000.0

    if strategy == "cheapest_fixed":
        return _run_cheapest_fixed(scenario, requests, rng)
    elif strategy == "fastest_fixed":
        return _run_fastest_fixed(scenario, requests, rng)
    elif strategy == "round_robin":
        return _run_round_robin(scenario, requests, rng)
    elif strategy == "lp_mix":
        return _run_lp_mix(scenario, requests, rng, slo_sec)
    elif strategy == "lp_hedge":
        return _run_lp_hedge(scenario, requests, rng, slo_sec)
    elif strategy == "v2_only":
        return _run_v2_only(scenario, requests, rng, slo_sec)
    elif strategy == "v2_p50_hedge":
        return _run_v2_p50_hedge(scenario, requests, rng, slo_sec)
    elif strategy == "oracle_per_window":
        return _run_oracle_per_window(scenario, requests, rng)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Choose from {STRATEGIES}")


def run_scenario(
    scenario: ScenarioConfig,
    requests: list[Request],
    seeds: list[int] | None = None,
    strategies: list[str] | None = None,
) -> dict[str, list[StrategyRun]]:
    """Run all strategies on a scenario across multiple seeds.

    Args:
        scenario: Scenario configuration.
        requests: Workload (same for all strategies / seeds).
        seeds: List of RNG seeds. Defaults to [42, 43, 44].
        strategies: Subset of STRATEGIES to run. Defaults to all.

    Returns:
        Dict mapping strategy name to list of StrategyRun (one per seed).
    """
    if seeds is None:
        seeds = [42, 43, 44]
    if strategies is None:
        strategies = STRATEGIES

    results: dict[str, list[StrategyRun]] = {s: [] for s in strategies}
    for strategy in strategies:
        for seed in seeds:
            run = run_strategy(scenario, requests, strategy, seed=seed)
            results[strategy].append(run)
    return results
