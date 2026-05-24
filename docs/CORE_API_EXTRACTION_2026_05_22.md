# RouteWise Public Core API Extraction

Date: 2026-05-22

## Goal

Extract the RouteWise math and decision primitives that are currently duplicated across SIM, REAL-EVAL, and hybridInference into a small public core API.

The core should be usable like:

```python
from routewise.core import solve_budget_lp, effective_cost, hedge_checkpoints_for_slo
```

Longer term, hybridInference should be able to depend on the package without pulling in the simulator harness, plotting stack, or experiment runners.

## Current Status

Three extraction targets have landed in this branch:

- `rwsim.core.lp.solve_budget_lp`
- `rwsim.core.lp.cost_tiebroken_objective`
- `rwsim.core.cost.effective_cost`
- `rwsim.core.cost.quota_effective_cost`
- `rwsim.core.cost.concurrency_effective_cost`
- `rwsim.core.cost.scarcity_price`
- `rwsim.core.hedging.hedge_checkpoints_for_slo`
- `rwsim.core.hedging.combined_success_probability`
- `rwsim.core.hedging.select_probability_backup`

SIM, REAL-EVAL, and ablation code now call these core APIs directly. The old
LP, hedging, and effective-cost helper wrappers were removed rather than kept
as compatibility aliases. REAL-EVAL no longer imports `scipy.optimize.linprog`
at policy runtime; scipy is used only as a test oracle/manual parity benchmark
for LP equivalence.

## What Belongs In Core

Core should contain pure, deterministic functions and shared data contracts:

- LP solver
- effective-cost calculation
- hedge checkpoint schedule
- combined success probability
- probability-target backup selection
- canonical routing/result schema types

Core should not contain provider transport, real API clients, CSV writers, PostgreSQL writers, simulator event queues, quota mutation, concurrency mutation, plotting, or benchmark orchestration.

## Design Principle

Core reads snapshots. Harnesses own mutable state.

| Concept | Core sees | Harness owns |
| --- | --- | --- |
| Quota | Read-only quota snapshot, fraction used, remaining budget | SIM/REAL elapsed-window state, PROD calendar/API reconciliation |
| Concurrency | Read-only availability/utilization snapshot | SIM virtual-time interval/heap, REAL/PROD counter/lock/release lifecycle |
| Latency profile | `cdf_at(ms)`, `mean_ms`, `success_rate`, quantiles | SIM empirical dist, REAL/PROD observed samples and error samples |
| Logging | Canonical record dataclass or normalized dict | SIM memory/CSV, REAL CSV, PROD PostgreSQL metadata |

This is the boundary that avoids the earlier H5 trap: we should not try to share a `try_acquire()` / `release()` implementation. Those are mutators, and they belong inside each harness. The shared core only needs the read-only facts required to decide.

## Current Module Layout

The core extraction currently lives inside the existing simulator package and is
re-exported through the public `routewise.core` alias:

```text
rwsim/core/
  __init__.py
  lp.py
  cost.py
  hedging.py

routewise/core/
  __init__.py
```

Still outside `rwsim.core`:

- Algorithm constants remain in `rwsim.const`; `rwsim.core.hedging` re-exports
  the hedging constants it needs.
- Simulator routing dataclasses remain in `rwsim.schemas`.
- The canonical request record remains in `rwsim.metrics.record`.

Later, if hybridInference needs an even smaller install surface, this can move or be published as a separate `routewise-core` package. For now, `rwsim.core` is the least disruptive path.

### Constants

Public source for algorithm constants is still `rwsim.const`:

- `HEDGE_SUCCESS_TARGET`
- `DISPATCH_OVERHEAD_MS`
- checkpoint schedule defaults

Do not add a second source of truth. If a future `rwsim.core.constants` module is
introduced, it should re-export from `rwsim.const` or replace it in one commit.

### Public Types

Provider-agnostic dataclasses should avoid simulator-specific classes such as
`Provider`, `TieredProvider`, or `ProviderState`.

Currently landed:

```python
@dataclass(frozen=True)
class BudgetLPCandidate:
    name: str
    objective: float
    effective_cost: float
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BudgetLPResult:
    feasible: bool
    weights: dict[str, float]
    budget: float
    objective: float | None = None
    expected_cost: float | None = None
    status: str = "infeasible"


@dataclass(frozen=True)
class BackupCandidate(Generic[ProviderT]):
    provider: ProviderT
    success_probability: float
    marginal_cost: float
    true_mean_ms: float
    success_target: float = HEDGE_SUCCESS_TARGET
```

The record schema has been aligned separately through `PerRequestRecord` tests,
but it has not moved to a `rwsim.core.records` module.

### `lp.py`

Public LP API:

```python
@dataclass(frozen=True)
class BudgetLPCandidate:
    name: str
    objective: float
    effective_cost: float
    metadata: Mapping[str, object] = field(default_factory=dict)


def solve_budget_lp(
    candidates: Sequence[BudgetLPCandidate],
    *,
    budget: float,
) -> BudgetLPResult:
    ...


```

`solve_budget_lp` is the public API. The low-level vector enumeration helper is
private to `rwsim.core.lp`; callers should pass named `BudgetLPCandidate`
snapshots instead of handling raw weight vectors.

Implementation uses exact enumeration, not `scipy.optimize.linprog`.

Reason: this LP has one budget constraint plus simplex constraints. The optimum is either a pure provider or a two-provider mix on the budget boundary. Enumeration is exact for this problem, fast, and production-friendly.

REAL-EVAL can keep a scipy equivalence test during migration, but runtime should not depend on scipy for the RouteWise router.

### `cost.py`

Public effective-cost API should operate on scalar snapshots:

```python
def effective_cost(
    tier: str,
    *,
    request_cost_usd: float = 0.0,
    quota_fraction_used: float | None = None,
    concurrency_utilization: float | None = None,
    L: float,
    U: float,
    quota_curve: ScarcityCurve = "exp_lu",
    concurrency_curve: ScarcityCurve | None = None,
) -> float:
    ...
```

SIM can compute `quota_fraction_used` from its virtual quota ledger. PROD can
compute it from calendar/API quota state. Core should not know how either state
is mutated. The main RouteWise concurrency path passes
`concurrency_curve=None`, which keeps reusable slots availability-only; ablation
paths can pass a curve such as `util_linear_u`.

### `hedging.py`

Public hedging API:

```python
def hedge_checkpoints_for_slo(slo_ms: float) -> tuple[float, ...]:
    ...


def combined_success_probability(
    primary_cdf_at_ms: Callable[[float], float],
    backup_cdf_at_ms: Callable[[float], float],
    *,
    elapsed_ms: float,
    slo_ms: float,
    dispatch_overhead_ms: float = 0.0,
) -> float:
    ...


def select_probability_backup(
    candidates: Sequence[BackupCandidate],
) -> BackupCandidate | None:
    ...
```

The `dispatch_overhead_ms` default should be explicit at call sites:

- SIM passes the modeled virtual dispatch overhead, currently 5 ms.
- REAL-EVAL passes 0 ms because wall-clock dispatch already elapsed.
- PROD should follow the same wall-clock rule unless it is making a pre-dispatch planning estimate.

Checkpoint scheduling now exposes the RouteWise paper behavior directly. The
old latest-safe helper was removed; REAL-EVAL waits by asking the same core
backup selector at future canonical checkpoints, while dispatch execution stays
inside the REAL-EVAL runner.

### `records.py`

Expose canonical schema types without forcing all harness writers to share one persistence implementation:

- `PerRequestRecord`
- status enums / literals
- normalization helpers for SIM, REAL-EVAL, and PROD records

The writer remains harness-specific:

- SIM writes in-memory summaries and simulator CSVs.
- REAL-EVAL writes CSV rows with extra transport/debug fields.
- PROD writes PostgreSQL rows plus `api_logs.metadata["routewise"]`.

## Migration Plan

### Phase 1: Add Core Without Behavior Change

- Create `rwsim.core` modules.
- Move or wrap existing pure helpers.
- Migrate call sites directly to core APIs; do not preserve duplicate helper wrappers.
- Add focused unit tests for the new public API.

Status: landed for LP, hedging, and effective cost.

Expected behavior change: none.

### Phase 2: Share LP Solver

- Move LP enumeration behind `rwsim.core.lp.solve_budget_lp`.
- Make SIM delegate to core.
- Make REAL-EVAL replace `scipy.optimize.linprog` runtime use with core enumeration.
- Add random small-case equivalence tests against scipy in test-only code while scipy remains in dev dependencies.

Status: landed for SIM, REAL-EVAL, and effective-cost ablation. The manual
parity/performance script now calls `rwsim.core.lp.solve_budget_lp`.

Expected behavior change: none, except tiny floating-point formatting differences.

### Phase 3: Share Hedging Math

- Move checkpoint schedule, combined success probability, and backup selection into `rwsim.core.hedging`.
- Make SIM and REAL-EVAL call the same functions with explicit dispatch-overhead semantics.
- Keep harness-specific dispatch execution separate.

Expected behavior change: none if call-site parameters match current intended semantics.

Status: landed for SIM and REAL-EVAL. The stale random explorer and latest-safe
helper paths were removed.

### Phase 4: Share Effective Cost

- Convert SIM effective-cost calculation to build scalar snapshots and call `rwsim.core.cost.effective_cost`.
- Convert REAL-EVAL adapters to the same scalar API.
- Keep quota/concurrency mutation and reconciliation in each harness.

Expected behavior change: none if snapshots match current state.

Status: landed for SIM, REAL-EVAL, and effective-cost ablations. PROD /
hybridInference integration remains a separate repo integration task.

### Phase 5: Stabilize Public Types

- Promote canonical routing/hedge/record dataclasses.
- Keep existing dataclasses as adapters where needed.
- Align canonical request schema tests with the public type definitions.

Expected behavior change: none.

### Phase 6: Packaging

Current `pyproject.toml` packages simulator, experiments, and CLI together. The
public core import path should not pull heavy dependencies such as scipy,
pandas, matplotlib, and pulp into a default install.

Status: option 1 has landed. The base distribution has no third-party runtime
dependencies and exposes only the lightweight `routewise.core` import path.
Heavier workflows are split into extras:

- `sim`: simulator dependencies, including the offline oracle dependency used
  by simulator cost-layer commands
- `real-eval`: live real-evaluation dependencies
- `offline`: offline optimizer dependencies
- `plots`: matplotlib/pandas plotting dependencies
- `scripts`: operational script dependencies

The import-boundary test imports `routewise.core` in a subprocess and asserts
that heavy modules such as numpy, scipy, pandas, and matplotlib were not loaded.
This locks the lazy-import behavior in `rwsim.__init__`, `rwsim.const`, and
`rwsim.core`.

## Non-Goals

- Do not share concurrency or quota mutator implementations.
- Do not force SIM, REAL-EVAL, and PROD to use the same persistence writer.
- Do not implement production hedging as part of this extraction.
- Coordinate environment reinstalls after the distribution rename because
  editable environments installed under the old `routewise-simulator` name will
  not update automatically.
- Do not expose simulator-specific provider objects in the public core API.

## Validation

Required checks before considering the extraction complete:

- Done: LP solver deterministic unit tests.
- Done: LP solver equivalence tests against scipy for random small cases.
- Done: LP parity/performance manual script using the core solver.
- Done: golden tests for hedge checkpoint schedule.
- Done: tests for combined success probability and backup selection.
- Done: cross-source canonical request schema parity tests.
- Done: REAL-EVAL policy tests proving scipy is no longer required at runtime.
- Remaining: simulator smoke comparison on the current minimax shared-profile run.
- Done: packaging check proving a lightweight core import path can avoid
  experiment/plot/runtime-heavy dependencies.

## Next Recommended Commit

The next semantic cleanup is the output predictor contract (H7):

1. Audit every route-time use of `PointPrediction` vs `QuantilePrediction`.
2. Keep `BucketMeanOutputPredictor` as a point predictor; do not let q10/q90
   paths accept it silently.
3. Prefer histogram/quantile predictors where the algorithm needs tail
   estimates.
4. Add focused tests for SIM and REAL-EVAL predictor selection behavior.

After that, the main remaining core-integration work is the hybridInference
adapter spike: import `routewise.core`, map production snapshots into the core
types, and keep production state mutation in the production harness.
