"""Eval grid factory for the simulator paper experiments.

The simulator evaluation is generated from three orthogonal axes plus a
workload axis. This module builds the ``stage × latency_family`` cross
product (9 scenarios), exposes the canonical 10-variant policy list, and
declares the workload list. See ``docs/DESIGN_PRINCIPLES.md`` for the full
design rationale.

This module is intentionally self-contained: it does **not** load YAML
configs or call into the legacy hand-authored scenario files under
``configs/``. New experimental questions should be expressed as cells in
this grid, not as new YAML files.

The paper-grid variant names produced here match the runnable variants
understood by ``experiments.tiered_capacity.lp_budget_eval.run_variant``
(``budget_range_p{0,25,50,75,100}`` and the ``_hedge`` suffix), so the grid
plugs directly into the existing runner without an intermediate translation
layer.

Public surface:

- :class:`LatencyFamily` — enum of distribution families
- :class:`ProviderSetup` — enum of provider-setup stages
- :data:`PAPER_P_VALUES` — the 5 cost-budget percentiles
- :data:`PAPER_HEDGE_MODES` — no_hedge / hedge (explorer is excluded by design;
  see PAPER_HEDGE_MODES docstring for the rationale)
- :data:`PAPER_GRID_VARIANTS` — the 10 runnable variant names
- :data:`PAPER_WORKLOADS` — paper-facing workload identifiers
- :data:`WORKLOAD_DATASET_IDS` — paper id → runner dataset id
- :func:`make_eval_grid_scenarios` — builds the 9 scenarios
- :func:`assert_grid_invariants` — structural (config) invariants
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable

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
    """Provider setup for each grid stage."""

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


def _stage3_specs() -> list[_ProviderSpec]:
    """Stage 3: full joint — S_Q + S_C + 2 S_A.

    Built on top of Stage 2: the two S_A providers preserve the cost-latency
    tradeoff so we can still observe Pareto behaviour in the API tier,
    while S_Q and S_C add the capacity / subscription dimensions.
    """
    return [
        # S_Q: free at the margin but quota-limited and moderately fast.
        _ProviderSpec(
            "S3_quota_chutes",
            ProviderTier.S_Q,
            cost_per_token=0.0,
            p50_ms=_P50_MEDIUM_MS,
            quota_size=1000,
        ),
        # S_C: free at the margin but concurrency-limited and fast.
        _ProviderSpec(
            "S3_conc_featherless",
            ProviderTier.S_C,
            cost_per_token=0.0,
            p50_ms=_P50_FAST_MS,
            concurrency_limit=4,
        ),
        # S_A overflow: cheap-slow.
        _ProviderSpec("S3_api_cheap_slow", ProviderTier.S_A, _COST_CHEAP, _P50_SLOW_MS),
        # S_A overflow: expensive-fast.
        _ProviderSpec("S3_api_expensive_fast", ProviderTier.S_A, _COST_EXPENSIVE, _P50_FAST_MS),
    ]


_STAGE_BUILDERS: dict[ProviderSetup, Callable[[], list[_ProviderSpec]]] = {
    ProviderSetup.SAME_COST: _stage1_specs,
    ProviderSetup.COST_LATENCY_TRADEOFF: _stage2_specs,
    ProviderSetup.JOINT_PROVIDER: _stage3_specs,
}


def make_scenario(stage: ProviderSetup, family: LatencyFamily) -> ScenarioConfig:
    """Build one scenario for a given (stage, latency family) cell."""
    specs = _STAGE_BUILDERS[stage]()
    providers = [_build_provider(spec, family) for spec in specs]
    return ScenarioConfig(
        name=f"grid_{stage.value}_{family.value}",
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
    """Build the full 3-stage × 3-family grid (9 scenarios)."""
    grid: dict[str, ScenarioConfig] = {}
    for stage in ProviderSetup:
        for family in LatencyFamily:
            scenario = make_scenario(stage, family)
            grid[scenario.name] = scenario
    return grid


# ---------------------------------------------------------------------------
# Axis 3: policy variants
# ---------------------------------------------------------------------------

# The cost-budget percentiles ``p`` used in the LP latency router.
# Following the paper convention ``B_p = c_min + p * (c_max - c_min)``,
# ``p=0`` minimises cost (always cheapest) and ``p=1`` ignores cost
# (minimise latency). Sweeping these values traces the Pareto frontier.
PAPER_P_VALUES: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

# Two hedge modes — explorer is intentionally excluded from the simulator
# grid. Per Juncheng's directive (2026-04-22 / 2026-04-28), the production
# design uses hedge-as-probe so probing is "free" in an active system; in
# a stationary simulator the dedicated-probing and hedge-as-probe paths
# converge to the same routing decisions. Showing both in figures wastes
# space without conveying signal. Explorer's value (operational
# simplicity + drift robustness) belongs to the real-experiment narrative,
# not the simulator grid.
PAPER_HEDGE_MODES: tuple[str, ...] = ("no_hedge", "hedge")


def _p_to_pct(p: float) -> int:
    """Convert a paper-facing p value (0.0–1.0) to the integer percentage
    used in the existing variant naming (``budget_range_p25`` etc.)."""
    pct_float = p * 100.0
    pct = int(round(pct_float))
    if not math.isclose(pct_float, pct, abs_tol=1e-9):
        raise ValueError(
            f"p={p!r} cannot be expressed as a whole-number percentage; "
            "extend PAPER_P_VALUES or _p_to_pct if you need finer p values."
        )
    return pct


_VALID_HEDGE_MODES: frozenset[str] = frozenset(("no_hedge", "hedge", "explorer"))


def variant_name(p: float, hedge_mode: str) -> str:
    """Canonical variant name: matches what ``lp_budget_eval.run_variant`` accepts.

    Mapping:

    - ``hedge_mode="no_hedge"`` → ``budget_range_p{pct}``
    - ``hedge_mode="hedge"``    → ``budget_range_p{pct}_hedge``
    - ``hedge_mode="explorer"`` → ``budget_range_p{pct}_explorer``

    Note: this constructor accepts ``"explorer"`` so callers can build
    runnable explorer ids on demand (e.g. for ad-hoc experiments). The
    paper grid (:data:`PAPER_HEDGE_MODES`) only contains
    ``("no_hedge", "hedge")`` — see that constant's docstring for why
    explorer is excluded from the paper-facing variant list.
    """
    if hedge_mode not in _VALID_HEDGE_MODES:
        raise ValueError(
            f"Unknown hedge mode: {hedge_mode!r}. "
            f"Expected one of {sorted(_VALID_HEDGE_MODES)}."
        )
    pct = _p_to_pct(p)
    base = f"budget_range_p{pct}"
    if hedge_mode == "no_hedge":
        return base
    return f"{base}_{hedge_mode}"


PAPER_GRID_VARIANTS: tuple[str, ...] = tuple(
    variant_name(p, mode) for p in PAPER_P_VALUES for mode in PAPER_HEDGE_MODES
)
"""Canonical 10-variant policy list = 5 p × 2 hedge modes (no_hedge, hedge).

Explorer (``hedge_as_probe``) is intentionally **not** included. In a
stationary simulator with the active-system probing assumption (probing
piggy-backs on hedging at no extra cost), explorer and hedge produce
identical routing — showing both wastes figure space. Explorer's value
is documented separately as a system-design contribution validated by
real-experiment data, not by simulator grid sweeps."""


# Variants used to compare the paper's new ``budget_range_p*`` LP family
# against the previous ``original_lp*`` SLO-anchored LP family.
#
# Old objective:  min Σ π_j × c_eff_j  s.t.  Σ π_j × F_j(SLO) ≥ ρ
#                 (parameterised by SLO satisfaction target ρ)
# New objective:  min Σ π_j × Tbar_j   s.t.  Σ π_j × c_eff_j ≤ B_p
#                 (parameterised by cost-budget percentile p)
#
# Both families produce points on the cost-latency Pareto plane. This
# constant exists so a single comparison run can plot all 12 points
# together and let reviewers see whether the new family dominates,
# matches, or diverges from the old one. See DESIGN_PRINCIPLES.md
# (forthcoming "lp comparison" finding) for the empirical result.
LP_COMPARISON_VARIANTS: tuple[str, ...] = (
    "original_lp",
    "original_lp_hedge",
    *PAPER_GRID_VARIANTS,
)
"""12-variant comparison list = 2 original_lp (no_hedge / hedge) +
10 budget_range_p*. Explorer is excluded for the same reason as
:data:`PAPER_GRID_VARIANTS` — see that docstring.

All entries are runnable ids accepted by ``lp_budget_eval.run_variant``."""


SHADOW_PRICE_ABLATION_VARIANTS: tuple[str, ...] = (
    "original_lp_rawcost",
    "original_lp_rawcost_hedge",
)
"""Shadow-price ablation set: ``original_lp`` with raw marginal cost
instead of ``effective_cost``. Pair with ``original_lp*`` to isolate the
contribution of ψ(z) / λ(u) shadow prices in Stage 3.

Explorer variant of rawcost is dropped for the same reason explorer is
out of :data:`PAPER_GRID_VARIANTS` (active-system / stationary simulator
makes hedge-as-probe redundant with dedicated probing). For Stage 2
(S_A only) the rawcost and effective-cost selectors should produce
identical routing because ``marginal_cost ≡ effective_cost`` for S_A.
The shadow-price benefit only shows up in Stage 3, where rawcost treats
S_Q and S_C as zero-cost (no quota / concurrency penalty)."""


# ---------------------------------------------------------------------------
# Axis 4: workloads
# ---------------------------------------------------------------------------


PAPER_WORKLOADS: tuple[str, ...] = ("sharegpt_burstgpt", "freeinference", "enterprise")
"""Paper-facing workload identifiers. These are the names used in the
paper text and figure captions."""


WORKLOAD_DATASET_IDS: dict[str, str] = {
    # Paper display name             → runner dataset id (must match
    # ``lp_budget_eval.TRACE_WORKLOAD_DATASETS``).
    "sharegpt_burstgpt": "sharegpt",  # ShareGPT prompts × BurstGPT arrivals
    "freeinference": "freeinference",
    "enterprise": "rednote",  # internal trace name
}
"""Paper-facing workload id → runner dataset id used by
``experiments.tiered_capacity.lp_budget_eval``."""


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
    "PAPER_HEDGE_MODES",
    "PAPER_P_VALUES",
    "PAPER_WORKLOADS",
    "ProviderSetup",
    "WORKLOAD_DATASET_IDS",
    "assert_grid_invariants",
    "make_eval_grid_scenarios",
    "make_scenario",
    "make_ttft_distribution",
    "runner_dataset_id",
    "variant_name",
]
