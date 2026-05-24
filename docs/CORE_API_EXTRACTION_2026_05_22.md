# RouteWise Public Core API Extraction

Date: 2026-05-22

## Goal

Extract the RouteWise math and decision primitives that are currently duplicated across SIM, REAL-EVAL, and hybridInference into a small public core API.

The core should be usable like:

```python
from rwsim.core import solve_budget_lp, effective_cost, hedge_checkpoints_for_slo
```

Longer term, hybridInference should be able to depend on the package without pulling in the simulator harness, plotting stack, or experiment runners.

## Current Status

Two extraction targets have landed in this branch:

- `rwsim.core.lp.solve_simplex_lp`
- `rwsim.core.lp.solve_budget_lp`
- `rwsim.core.lp.cost_tiebroken_objective`
- `rwsim.core.lp.normalize_weights`
- `rwsim.core.hedging.hedge_checkpoints_for_slo`
- `rwsim.core.hedging.combined_success_probability`
- `rwsim.core.hedging.select_probability_backup`

SIM, REAL-EVAL, and ablation code now call these core APIs directly. The old
LP and hedging helper wrappers were removed rather than kept as compatibility
aliases. REAL-EVAL no longer imports `scipy.optimize.linprog` at policy runtime;
scipy is used only as a test oracle for LP equivalence.

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

## Proposed Module Layout

Start inside the existing package to minimize packaging churn:

```text
rwsim/core/
  __init__.py
  constants.py
  types.py
  lp.py
  cost.py
  hedging.py
  records.py
```

Later, if hybridInference needs an even smaller install surface, this can move or be published as a separate `routewise-core` package. For now, `rwsim.core` is the least disruptive path.

### `constants.py`

Public source for algorithm constants:

- `HEDGE_SUCCESS_TARGET`
- `DISPATCH_OVERHEAD_MS`
- checkpoint schedule defaults

Existing imports from `rwsim.const` can be preserved as compatibility aliases.

### `types.py`

Provider-agnostic dataclasses. These should avoid simulator-specific classes such as `Provider`, `TieredProvider`, or `ProviderState`.

Sketch:

```python
@dataclass(frozen=True)
class CandidateSnapshot:
    provider: str
    tier: str
    marginal_cost_usd: float
    effective_cost_usd: float
    latency_objective_ms: float
    available: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LPResult:
    status: str
    weights: Mapping[str, float]
    budget_usd: float
    objective_ms: float | None
    expected_effective_cost_usd: float | None


@dataclass(frozen=True)
class BackupCandidate:
    provider: str
    success_probability: float
    marginal_cost_usd: float
    mean_ms: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

The exact field names can mirror the canonical request schema where possible.

### `lp.py`

Public LP API:

```python
def solve_budget_lp(
    candidates: Sequence[CandidateSnapshot],
    *,
    budget_usd: float,
) -> LPResult:
    ...
```

Implementation should use the enumeration solver currently in `rwsim.policies.routewise._solve_lp`, not `scipy.optimize.linprog`.

Reason: this LP has one budget constraint plus simplex constraints. The optimum is either a pure provider or a two-provider mix on the budget boundary. Enumeration is exact for this problem, fast, and production-friendly.

REAL-EVAL can keep a scipy equivalence test during migration, but runtime should not depend on scipy for the RouteWise router.

### `cost.py`

Public effective-cost API should operate on scalar snapshots:

```python
def effective_cost(
    *,
    tier: str,
    marginal_cost_usd: float,
    quota_fraction_used: float | None = None,
    quota_lower: float | None = None,
    quota_upper: float | None = None,
    concurrency_utilization: float | None = None,
    concurrency_alpha: float = 1.0,
) -> float:
    ...
```

SIM can compute `quota_fraction_used` from its virtual quota ledger. PROD can compute it from calendar/API quota state. Core should not know how either state is mutated.

### `hedging.py`

Public hedging API:

```python
def hedge_checkpoints_for_slo(
    slo_ms: float,
    *,
    first_fraction: float = 0.25,
    last_fraction: float = 0.90,
    step_fraction: float = 0.025,
) -> tuple[float, ...]:
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
    *,
    success_target: float = HEDGE_SUCCESS_TARGET,
) -> BackupCandidate | None:
    ...
```

The `dispatch_overhead_ms` default should be explicit at call sites:

- SIM passes the modeled virtual dispatch overhead, currently 5 ms.
- REAL-EVAL passes 0 ms because wall-clock dispatch already elapsed.
- PROD should follow the same wall-clock rule unless it is making a pre-dispatch planning estimate.

For checkpoint scheduling, the shared API should expose the RouteWise paper behavior. A later helper can support REAL-EVAL's latest-safe search:

```python
def latest_safe_checkpoint(
    checkpoints_sec: Sequence[float],
    *,
    primary_cdf_at_ms: Callable[[float], float],
    backup_profiles: Sequence[BackupProfileSnapshot],
    slo_sec: float,
    success_target: float = HEDGE_SUCCESS_TARGET,
    dispatch_overhead_ms: float = 0.0,
) -> float | None:
    ...
```

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

Expected behavior change: none.

### Phase 2: Share LP Solver

- Move `_solve_lp` enumeration to `rwsim.core.lp.solve_budget_lp`.
- Make SIM delegate to core.
- Make REAL-EVAL replace `scipy.optimize.linprog` runtime use with core enumeration.
- Add random small-case equivalence tests against scipy in test-only code while scipy remains in dev dependencies.

Expected behavior change: none, except tiny floating-point formatting differences.

### Phase 3: Share Hedging Math

- Move checkpoint schedule, combined success probability, and backup selection into `rwsim.core.hedging`.
- Make SIM and REAL-EVAL call the same functions with explicit dispatch-overhead semantics.
- Keep harness-specific dispatch execution separate.

Expected behavior change: none if call-site parameters match current intended semantics.

### Phase 4: Share Effective Cost

- Convert SIM effective-cost calculation to build scalar snapshots and call `rwsim.core.cost.effective_cost`.
- Convert REAL-EVAL and PROD adapters to the same scalar API.
- Keep quota/concurrency mutation and reconciliation in each harness.

Expected behavior change: none if snapshots match current state.

### Phase 5: Stabilize Public Types

- Promote canonical routing/hedge/record dataclasses.
- Keep existing dataclasses as adapters where needed.
- Align canonical request schema tests with the public type definitions.

Expected behavior change: none.

### Phase 6: Packaging

Current `pyproject.toml` packages simulator, experiments, and CLI together and pulls heavy dependencies such as scipy, pandas, matplotlib, and pulp into the default install. That is not ideal for production integration.

Options:

1. Keep one package but add extras:
   - base: core only
   - `sim`: simulator dependencies
   - `experiments`: real-eval/offline dependencies
   - `plots`: matplotlib/pandas

2. Split a separate `routewise-core` package later.

Recommendation: start with option 1. It is easier to land incrementally and still gives hybridInference a lightweight import path once dependency groups are cleaned up.

## Non-Goals

- Do not share concurrency or quota mutator implementations.
- Do not force SIM, REAL-EVAL, and PROD to use the same persistence writer.
- Do not implement production hedging as part of this extraction.
- Do not rename the package before the core API is stable.
- Do not expose simulator-specific provider objects in the public core API.

## Validation

Required checks before considering the extraction complete:

- LP solver deterministic unit tests.
- LP solver equivalence tests against scipy for random small cases.
- Golden tests for hedge checkpoint schedule.
- Golden tests for combined success probability under SIM and REAL-EVAL overhead semantics.
- Cross-source canonical request schema parity tests.
- Simulator smoke comparison on the current minimax shared-profile run.
- REAL-EVAL policy tests proving scipy is no longer required at runtime.

## Next Recommended Commit

Effective-cost extraction is the next best target:

1. Add `rwsim.core.cost`.
2. Move the scalar RouteWise effective-cost formula there.
3. Keep quota/concurrency/cache state in SIM and REAL-EVAL adapters.
4. Make SIM, REAL-EVAL, and the effective-cost ablation build scalar snapshots and call the same core cost helper.
