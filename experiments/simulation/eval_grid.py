"""Eval grid factory for the simulator paper experiments.

The simulator evaluation is generated from provider setup, latency family, and
workload axes. This module builds the ``stage × latency_family`` cross product
(12 scenarios; 4 stages × 3 distributions), exposes the six paper-name policy
presets, and declares the workload list. See ``docs/DESIGN_PRINCIPLES.md`` for
the full design rationale.

This module is intentionally self-contained: it does **not** load YAML
configs or call into the legacy hand-authored scenario files under
``configs/``. New experimental questions should be expressed as cells in
this grid, not as new YAML files.

Public surface:

- :class:`LatencyFamily` — enum of distribution families
- :class:`ProviderSetup` — enum of provider-setup stages
- :data:`PAPER_GRID_VARIANTS` — the six runnable paper-name policy presets
- :data:`PAPER_WORKLOADS` — paper-facing workload identifiers
- :data:`WORKLOAD_DATASET_IDS` — paper id → runner dataset id
- :func:`make_eval_grid_scenarios` — builds the 12 scenarios (4 stages × 3 distributions)
- :func:`assert_grid_invariants` — structural (config) invariants
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum

from rwsim.world import (
    ConcurrencyState,
    ProviderTier,
    QuotaState,
    ScenarioConfig,
    TieredProvider,
)
from rwsim.world.distributions import LogNormal, Normal, Uniform


# ---------------------------------------------------------------------------
# Axis 1: provider setup (stage)
# ---------------------------------------------------------------------------


class ProviderSetup(str, Enum):
    """Provider setup for each grid stage.

    Stage progression follows Juncheng's directive (2026-04-28 meeting):
    "Build complexity gradually — one variable per experiment." Each stage
    adds exactly one new dimension over the previous one.
    """

    SAME_LATENCY = "same_latency"  # Stage 0: 3 S_A, same latency, different cost
    SAME_COST = "same_cost"  # Stage 1: 3 S_A, same cost, different latency
    COST_LATENCY_TRADEOFF = "cost_latency_tradeoff"  # Stage 2: cost-latency tradeoff
    JOINT_PROVIDER = "joint_provider"  # Stage 3: S_A + S_Q + S_C


# ---------------------------------------------------------------------------
# Axis 2: latency distribution family
# ---------------------------------------------------------------------------


class LatencyFamily(str, Enum):
    """Latency distribution family used to fill provider TTFT/TPS slots."""

    UNIFORM = "uniform"
    NORMAL = "normal"
    HEAVY_TAIL = "heavy_tail"


# Per-family default shape parameter. Each family interprets the same
# numerical value differently; the defaults below are picked so that:
#
#   1. Heavy-tail visibly dominates Normal which dominates Uniform on the
#      tail (P99/P50 ratio strictly increases across the families).
#   2. The Normal floor-clipping kicks in only on a negligible mass of
#      samples for any P50 used by the grid (smallest P50 is 100 ms; with
#      ``shape=0.3`` the 3-sigma lower bound stays at 10 ms, well above
#      ``MIN_LATENCY_MS=1``).
#
# These defaults are documented in ``docs/DESIGN_PRINCIPLES.md`` and should
# only be changed there.
_DEFAULT_SHAPE_BY_FAMILY: dict[LatencyFamily, float] = {
    LatencyFamily.UNIFORM: 0.5,  # half-width as fraction of P50
    LatencyFamily.NORMAL: 0.3,  # sigma / P50 ratio (CV)
    LatencyFamily.HEAVY_TAIL: 0.5,  # LogNormal sigma directly
}


def make_ttft_distribution(
    family: LatencyFamily,
    p50_ms: float,
    *,
    shape: float | None = None,
):
    """Build a distribution targeting the given P50.

    ``shape`` controls the relative spread; the meaning is family-specific
    (see :data:`_DEFAULT_SHAPE_BY_FAMILY` for documentation). When
    ``shape`` is ``None``, the per-family default is used.
    """
    if p50_ms <= 0.0:
        raise ValueError(f"p50_ms must be positive, got {p50_ms}")
    if shape is None:
        shape = _DEFAULT_SHAPE_BY_FAMILY[family]

    if family is LatencyFamily.UNIFORM:
        half_width = p50_ms * shape
        low = max(p50_ms - half_width, 1.0)
        high = p50_ms + half_width
        return Uniform(low=low, high=high)

    if family is LatencyFamily.NORMAL:
        return Normal(mean_ms=p50_ms, sigma=p50_ms * shape)

    if family is LatencyFamily.HEAVY_TAIL:
        return LogNormal(mu=math.log(p50_ms), sigma=shape)

    raise ValueError(f"Unknown latency family: {family!r}")


# ---------------------------------------------------------------------------
# Provider builders (stage-aware)
# ---------------------------------------------------------------------------


# Same TPS distribution across all providers in the grid: TPS is not the
# variable under study, so keep it constant. Mu=5.5 corresponds to
# P50 = 245 tokens/sec, which sits in the realistic range for hosted
# inference of mid-sized models.
_DEFAULT_TPS = LogNormal(mu=5.5, sigma=0.3)


@dataclass
class _ProviderSpec:
    """Internal struct describing one provider in a grid scenario."""

    name: str
    tier: ProviderTier
    cost_per_token: float
    p50_ms: float
    quota_size: int | None = None
    concurrency_limit: int | None = None


def _build_provider(spec: _ProviderSpec, family: LatencyFamily) -> TieredProvider:
    """Materialize a provider spec under the given latency family."""
    ttft_dist = make_ttft_distribution(family, spec.p50_ms)

    quota = None
    if spec.tier is ProviderTier.S_Q:
        if spec.quota_size is None:
            raise ValueError(f"S_Q provider {spec.name!r} requires quota_size")
        quota = QuotaState(size=spec.quota_size, window_sec=86400.0)

    concurrency = None
    if spec.tier is ProviderTier.S_C:
        if spec.concurrency_limit is None:
            raise ValueError(f"S_C provider {spec.name!r} requires concurrency_limit")
        concurrency = ConcurrencyState(limit=spec.concurrency_limit)

    return TieredProvider(
        name=spec.name,
        cost_per_token=spec.cost_per_token,
        ttft_dist=ttft_dist,
        tps_dist=_DEFAULT_TPS,
        tier=spec.tier,
        quota=quota,
        concurrency=concurrency,
    )


# ---------------------------------------------------------------------------
# Stage scenario factories
# ---------------------------------------------------------------------------

# A canonical "fast / medium / slow" P50 ladder, reused across stages so
# that the only thing varying between stages is the cost and capacity
# configuration, not the latency landscape itself.
_P50_FAST_MS = 100.0
_P50_MEDIUM_MS = 300.0
_P50_SLOW_MS = 1000.0

# Cost ladder for S_A providers, in $/token. Three points spaced 1:2:4 so
# adjacent steps are clean doublings ($1, $2, $4 per million tokens). The
# absolute values are synthetic; what matters for the LP is the ratio
# c_max / c_min = 4, which gives the cost budget B_p a meaningful sweep
# range (see DESIGN_PRINCIPLES.md §2.1 Stage 2 rationale).
_COST_CHEAP = 1e-6  # $1 / M
_COST_MEDIUM = 2e-6  # $2 / M
_COST_EXPENSIVE = 4e-6  # $4 / M


def _stage0_specs() -> list[_ProviderSpec]:
    """Stage 0: 3 S_A providers, **same latency, different cost**.

    Per Juncheng's 2026-04-28 directive — a "fantasy test" for the cost
    router in isolation. Latency is identical across all three providers,
    so the only meaningful decision the LP can make is cost. The
    algorithm should always pick the cheapest provider; if it does not,
    something is wrong with the cost-routing path before we move on to
    cost-latency tradeoffs in Stage 2.
    """
    return [
        _ProviderSpec("S0_cheap", ProviderTier.S_A, _COST_CHEAP, _P50_MEDIUM_MS),
        _ProviderSpec("S0_medium", ProviderTier.S_A, _COST_MEDIUM, _P50_MEDIUM_MS),
        _ProviderSpec("S0_expensive", ProviderTier.S_A, _COST_EXPENSIVE, _P50_MEDIUM_MS),
    ]


def _stage1_specs() -> list[_ProviderSpec]:
    """Stage 1: 3 S_A providers, same cost, different latency."""
    return [
        _ProviderSpec("S1_fast", ProviderTier.S_A, _COST_MEDIUM, _P50_FAST_MS),
        _ProviderSpec("S1_medium", ProviderTier.S_A, _COST_MEDIUM, _P50_MEDIUM_MS),
        _ProviderSpec("S1_slow", ProviderTier.S_A, _COST_MEDIUM, _P50_SLOW_MS),
    ]


def _stage2_specs() -> list[_ProviderSpec]:
    """Stage 2: 3 S_A providers, cost-latency tradeoff (inverse cost vs latency)."""
    return [
        _ProviderSpec("S2_cheap_slow", ProviderTier.S_A, _COST_CHEAP, _P50_SLOW_MS),
        _ProviderSpec("S2_medium", ProviderTier.S_A, _COST_MEDIUM, _P50_MEDIUM_MS),
        _ProviderSpec("S2_expensive_fast", ProviderTier.S_A, _COST_EXPENSIVE, _P50_FAST_MS),
    ]


def _stage3_specs(
    *,
    quota_size: int = 1000,
    concurrency_limit: int = 4,
) -> list[_ProviderSpec]:
    """Stage 3: full joint — S_Q + S_C + 2 S_A.

    Built on top of Stage 2: the two S_A providers preserve the cost-latency
    tradeoff so we can still observe Pareto behaviour in the API tier,
    while S_Q and S_C add the capacity / subscription dimensions.
    """
    if quota_size <= 0:
        raise ValueError(f"quota_size must be positive, got {quota_size}")
    if concurrency_limit <= 0:
        raise ValueError(
            f"concurrency_limit must be positive, got {concurrency_limit}"
        )
    return [
        # S_Q: free at the margin but quota-limited and moderately fast.
        _ProviderSpec(
            "S3_quota_chutes",
            ProviderTier.S_Q,
            cost_per_token=0.0,
            p50_ms=_P50_MEDIUM_MS,
            quota_size=quota_size,
        ),
        # S_C: free at the margin but concurrency-limited and fast.
        _ProviderSpec(
            "S3_conc_featherless",
            ProviderTier.S_C,
            cost_per_token=0.0,
            p50_ms=_P50_FAST_MS,
            concurrency_limit=concurrency_limit,
        ),
        # S_A overflow: cheap-slow.
        _ProviderSpec("S3_api_cheap_slow", ProviderTier.S_A, _COST_CHEAP, _P50_SLOW_MS),
        # S_A overflow: expensive-fast.
        _ProviderSpec("S3_api_expensive_fast", ProviderTier.S_A, _COST_EXPENSIVE, _P50_FAST_MS),
    ]


_STAGE_BUILDERS: dict[ProviderSetup, Callable[[], list[_ProviderSpec]]] = {
    ProviderSetup.SAME_LATENCY: _stage0_specs,
    ProviderSetup.SAME_COST: _stage1_specs,
    ProviderSetup.COST_LATENCY_TRADEOFF: _stage2_specs,
    ProviderSetup.JOINT_PROVIDER: _stage3_specs,
}


def make_scenario(
    stage: ProviderSetup,
    family: LatencyFamily,
    *,
    stage3_quota_size: int | None = None,
    stage3_concurrency_limit: int | None = None,
) -> ScenarioConfig:
    """Build one scenario for a given (stage, latency family) cell."""
    has_stage3_override = (
        stage3_quota_size is not None or stage3_concurrency_limit is not None
    )
    if has_stage3_override and stage is not ProviderSetup.JOINT_PROVIDER:
        raise ValueError(
            "stage3_quota_size / stage3_concurrency_limit only apply to "
            f"ProviderSetup.JOINT_PROVIDER, got {stage!r}"
        )

    if stage is ProviderSetup.JOINT_PROVIDER and has_stage3_override:
        quota_size = 1000 if stage3_quota_size is None else stage3_quota_size
        concurrency_limit = (
            4 if stage3_concurrency_limit is None else stage3_concurrency_limit
        )
        specs = _stage3_specs(
            quota_size=quota_size,
            concurrency_limit=concurrency_limit,
        )
        name = stage3_capacity_scenario_name(
            family,
            quota_size=quota_size,
            concurrency_limit=concurrency_limit,
        )
    else:
        specs = _STAGE_BUILDERS[stage]()
        name = f"grid_{stage.value}_{family.value}"

    providers = [_build_provider(spec, family) for spec in specs]
    return ScenarioConfig(
        name=name,
        description=(
            f"Eval grid cell: stage={stage.value}, latency_family={family.value}. "
            f"Providers: {', '.join(p.name for p in providers)}."
        ),
        providers=providers,
        n_requests=2000,
        duration_seconds=3600.0,
        arrival_process="poisson",
        primary_slo_ms=2000.0,
    )


def make_eval_grid_scenarios() -> dict[str, ScenarioConfig]:
    """Build the full 4-stage × 3-family grid (12 scenarios)."""
    grid: dict[str, ScenarioConfig] = {}
    for stage in ProviderSetup:
        for family in LatencyFamily:
            scenario = make_scenario(stage, family)
            grid[scenario.name] = scenario
    return grid


def stage3_capacity_scenario_name(
    family: LatencyFamily,
    *,
    quota_size: int,
    concurrency_limit: int,
) -> str:
    """Stable scenario id for Stage 3 capacity-sensitivity cells."""
    return (
        f"grid_{ProviderSetup.JOINT_PROVIDER.value}_{family.value}"
        f"_q{quota_size}_c{concurrency_limit}"
    )


def make_stage3_capacity_scenarios(
    *,
    quota_sizes: Iterable[int],
    concurrency_limits: Iterable[int],
    families: Iterable[LatencyFamily] = (LatencyFamily.HEAVY_TAIL,),
) -> dict[str, ScenarioConfig]:
    """Build Stage 3 sensitivity scenarios over quota/concurrency settings.

    The default family is ``heavy_tail`` because Stage 3 is the paper's
    representative joint-provider case; uniform/normal can be added by
    passing ``families`` explicitly for appendix sweeps.
    """
    grid: dict[str, ScenarioConfig] = {}
    for family in families:
        for quota_size in quota_sizes:
            for concurrency_limit in concurrency_limits:
                scenario = make_scenario(
                    ProviderSetup.JOINT_PROVIDER,
                    family,
                    stage3_quota_size=quota_size,
                    stage3_concurrency_limit=concurrency_limit,
                )
                grid[scenario.name] = scenario
    return grid


# ---------------------------------------------------------------------------
# Axis 3: policy presets
# ---------------------------------------------------------------------------

PAPER_GRID_VARIANTS: tuple[str, ...] = (
    "greedy_cost",
    "greedy_latency",
    "random",
    "ablation_lp_only",
    "ablation_lp_hedging",
    "routewise",
)
"""Canonical six-policy simulator surface.

The first three are baselines. The final three are RouteWise ablations:
LP only, LP plus hedging, and full RouteWise with explorer enabled."""

LP_COMPARISON_VARIANTS: tuple[str, ...] = PAPER_GRID_VARIANTS
"""Compatibility export for older suite code; no historical LP names remain."""

SHADOW_PRICE_ABLATION_VARIANTS: tuple[str, ...] = ()
"""Shadow pricing is internal RouteWise logic, not a top-level policy preset."""


# ---------------------------------------------------------------------------
# Axis 4: workloads
# ---------------------------------------------------------------------------


PAPER_WORKLOADS: tuple[str, ...] = ("sharegpt_burstgpt", "freeinference", "enterprise")
"""Paper-facing workload identifiers. These are the names used in the
paper text and figure captions."""


WORKLOAD_DATASET_IDS: dict[str, str] = {
    # Paper display name             → runner dataset id (must match
    # ``lp_budget_eval.TRACE_WORKLOAD_DATASETS``).
    "sharegpt_burstgpt": "burstgpt",  # BurstGPT 30d arrivals + reused ShareGPT text
    "freeinference": "freeinference",
    "enterprise": "rednote",  # internal trace name
}
"""Paper-facing workload id → runner dataset id used by
``experiments.simulation.lp_budget_eval``."""


def runner_dataset_id(paper_workload: str) -> str:
    """Translate a paper workload name to the runner dataset id."""
    try:
        return WORKLOAD_DATASET_IDS[paper_workload]
    except KeyError as exc:
        raise ValueError(
            f"Unknown paper workload {paper_workload!r}; "
            f"expected one of {sorted(WORKLOAD_DATASET_IDS)}"
        ) from exc


# ---------------------------------------------------------------------------
# Grid invariants (structural property checks)
# ---------------------------------------------------------------------------


def _summarize_invariants(scenarios: dict[str, ScenarioConfig]) -> list[str]:
    """Return a list of structural invariant violations.

    These are *config-level* checks: counts, costs, tier presence. They do
    not run the router. Behavioural claims about how the algorithm should
    distribute traffic on Stage 3 are deliberately **not** asserted here
    because the actual distribution depends on shadow-price interaction
    between ``ψ(z)`` and ``λ(u)``, capacity caps, and arrival pattern, and
    can be validated only by execution.
    """
    problems: list[str] = []

    expected_count = len(ProviderSetup) * len(LatencyFamily)
    if len(scenarios) != expected_count:
        problems.append(
            f"Expected {expected_count} grid scenarios, got {len(scenarios)}: "
            f"{sorted(scenarios)}"
        )

    for name, scenario in scenarios.items():
        # Stage 0: every provider should have the same latency (one P50)
        # but distinct costs (3 different cost_per_token values).
        if name.startswith("grid_same_latency_"):
            p50s = {p.ttft_dist.p50() for p in scenario.providers}
            if len(p50s) != 1:
                problems.append(
                    f"{name}: same_latency stage must have identical P50 across "
                    f"providers, got {p50s}"
                )
            costs = {p.cost_per_token for p in scenario.providers}
            if len(costs) < 2:
                problems.append(
                    f"{name}: same_latency stage must have at least 2 distinct "
                    f"cost_per_token values, got {costs}"
                )

        # Stage 1: every provider should have the same cost.
        if name.startswith("grid_same_cost_"):
            costs = {p.cost_per_token for p in scenario.providers}
            if len(costs) != 1:
                problems.append(
                    f"{name}: same_cost stage must have identical cost_per_token, "
                    f"got {costs}"
                )

        # Stage 2: cost and latency should both vary (3 distinct costs).
        if name.startswith("grid_cost_latency_tradeoff_"):
            costs = {p.cost_per_token for p in scenario.providers}
            if len(costs) < 3:
                problems.append(
                    f"{name}: cost_latency_tradeoff must have at least 3 distinct costs, "
                    f"got {costs}"
                )

        # Stage 3: must include all three tiers and a non-zero quota /
        # concurrency configuration so capacity-aware routing has
        # something to do.
        if name.startswith("grid_joint_provider_"):
            tiers = {p.tier for p in scenario.providers}
            missing = {ProviderTier.S_A, ProviderTier.S_Q, ProviderTier.S_C} - tiers
            if missing:
                problems.append(
                    f"{name}: joint_provider stage missing tiers {missing}"
                )
            for prov in scenario.providers:
                if prov.tier is ProviderTier.S_Q and (prov.quota is None or prov.quota.size <= 0):
                    problems.append(
                        f"{name}: S_Q provider {prov.name!r} must have positive quota"
                    )
                if prov.tier is ProviderTier.S_C and (
                    prov.concurrency is None or prov.concurrency.limit <= 0
                ):
                    problems.append(
                        f"{name}: S_C provider {prov.name!r} must have positive concurrency limit"
                    )

    return problems


def assert_grid_invariants(scenarios: dict[str, ScenarioConfig] | None = None) -> None:
    """Raise ``AssertionError`` if the grid violates structural invariants.

    See :func:`_summarize_invariants` for what is and is **not** checked.
    """
    if scenarios is None:
        scenarios = make_eval_grid_scenarios()
    problems = _summarize_invariants(scenarios)
    if problems:
        raise AssertionError("Grid invariant violations:\n  - " + "\n  - ".join(problems))


__all__ = [
    "LP_COMPARISON_VARIANTS",
    "SHADOW_PRICE_ABLATION_VARIANTS",
    "LatencyFamily",
    "PAPER_GRID_VARIANTS",
    "PAPER_WORKLOADS",
    "ProviderSetup",
    "WORKLOAD_DATASET_IDS",
    "assert_grid_invariants",
    "make_eval_grid_scenarios",
    "make_scenario",
    "make_stage3_capacity_scenarios",
    "make_ttft_distribution",
    "runner_dataset_id",
    "stage3_capacity_scenario_name",
]
