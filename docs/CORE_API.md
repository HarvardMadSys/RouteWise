# RouteWise Core API

This document describes the stable, lightweight RouteWise library surface:
`llm_routewise.core`.

If you are looking for the public API-provider interface, read
[API.md](API.md) or [API.zh-CN.md](API.zh-CN.md) first. This document covers
the lower-level mathematical primitives; the interface documents describe the
user-facing `0.1.0` facade.

`llm_routewise.core` contains pure routing math and small data contracts. It does
not import the simulator, experiment harnesses, plotting code, live-provider
transports, or production gateway integrations.

Use this API when you want to embed RouteWise's effective-cost, budgeted
provider-mixing, and hedging primitives inside your own router.

The core package is side-effect-free. Your application owns provider state,
network calls, request accounting, and latency profile updates.

## Install

The core API has no third-party runtime dependencies.

From a published release:

```bash
python -m pip install llm-routewise
```

For an editable checkout:

```bash
python -m pip install -e .
```

Do not install simulator or real-evaluation extras unless you need the paper
artifact workflows.

## Quickstart

This is the smallest complete routing step: price feasible providers, solve the
budgeted latency LP, and pick a primary provider from the returned weights.

```python
from llm_routewise.core import BudgetLPCandidate, effective_cost, solve_budget_lp

L = 0.0001
U = 0.0100
alpha = 0.25

candidates = [
    BudgetLPCandidate(
        name="api_fast",
        objective=180.0,
        effective_cost=effective_cost(
            "api",
            request_cost_usd=0.0015,
            L=L,
            U=U,
        ),
    ),
    BudgetLPCandidate(
        name="quota_pool",
        objective=420.0,
        effective_cost=effective_cost(
            "quota",
            quota_fraction_used=0.40,
            L=L,
            U=U,
        ),
    ),
]

c_min = min(candidate.effective_cost for candidate in candidates)
c_max = max(candidate.effective_cost for candidate in candidates)
budget = c_min + alpha * (c_max - c_min)

result = solve_budget_lp(candidates, budget=budget)
if not result.feasible:
    raise RuntimeError("no feasible provider under budget")

# Production routers usually sample from result.weights. This deterministic
# example picks the highest-probability provider.
primary_provider = max(result.weights, key=result.weights.get)
```

## Common Imports

Import from `llm_routewise.core` only:

```python
from llm_routewise.core import (
    BackupCandidate,
    BudgetLPCandidate,
    BudgetLPResult,
    CheckpointBackupDispatch,
    CheckpointBackupSelector,
    HedgeDispatch,
    RoutingDecision,
    combined_success_probability,
    cost_tiebroken_objective,
    effective_cost,
    has_feasible_backup,
    hedge_checkpoints_for_slo,
    select_probability_backup,
    solve_budget_lp,
)
```

The import surface above is the stable contract; the internal module layout
under `llm_routewise.core` may change between releases.

## Units

Use consistent units at your integration boundary:

| Value | Unit |
| --- | --- |
| Effective cost | USD per request |
| `L`, `U` envelope | USD per request |
| LP objective | Any lower-is-better scalar; usually milliseconds |
| LP budget | USD per request |
| Hedging CDF inputs | milliseconds |
| Hedge checkpoints | seconds elapsed after primary dispatch |
| Quota usage | fraction in `[0, 1]` |
| Budget knob `alpha` | fraction in `[0, 1]` |

## Effective Cost

RouteWise normalizes provider choices into one effective request cost.

```python
from llm_routewise.core import effective_cost

L = 0.0001   # P10 cheapest API request cost in the workload
U = 0.0100   # P90 cheapest API request cost in the workload

api_cost = effective_cost(
    "api",
    request_cost_usd=0.00042,
    L=L,
    U=U,
)

quota_cost = effective_cost(
    "quota",
    quota_fraction_used=0.70,
    L=L,
    U=U,
)

concurrency_cost = effective_cost(
    "concurrency",
    concurrency_utilization=0.25,
    L=L,
    U=U,
)
```

Expected semantics:

- `api`: returns the metered request cost you pass in.
- `quota`: returns the RouteWise exponential opportunity cost
  `L * (U / L) ** quota_fraction_used`.
- `concurrency`: returns `0.0` by default. RouteWise treats reusable concurrency
  slots as availability-gated capacity, not a depletable scarce resource.

Your integration owns provider state. Check quota/concurrency feasibility before
calling the LP, then pass only feasible providers onward.

## Cost Envelope

The quota shadow price requires a workload-level `(L, U)` envelope. The paper
uses the P10 and P90 of the cheapest API cost distribution for the workload.

```python
def workload_cost_envelope(request_costs: list[float]) -> tuple[float, float]:
    values = sorted(cost for cost in request_costs if cost > 0.0)
    if not values:
        return (1e-6, 1e-3)

    def percentile(p: float) -> float:
        index = round((len(values) - 1) * p)
        return float(values[index])

    L = percentile(0.10)
    U = percentile(0.90)
    if not (0.0 < L < U):
        L = max(U * 1e-3, 1e-9)
    return (L, U)
```

Do not calibrate `(L, U)` from a single request. That makes quota look almost
free for most of the quota window and changes the routing behavior.

## Budgeted Provider Mixer

`solve_budget_lp()` solves:

```text
minimize    sum_j weight_j * objective_j
subject to  sum_j weight_j * effective_cost_j <= budget
            sum_j weight_j = 1
            weight_j >= 0
```

The solver returns a sparse provider mixture. With one cost budget constraint,
an optimal solution has at most two non-zero weights.

```python
from llm_routewise.core import BudgetLPCandidate, solve_budget_lp

candidates = [
    BudgetLPCandidate(
        name="cheap_slow",
        objective=850.0,        # mean TTFT in ms
        effective_cost=0.0002,  # USD/request
    ),
    BudgetLPCandidate(
        name="fast_expensive",
        objective=180.0,
        effective_cost=0.0015,
    ),
]

alpha = 0.25
c_min = min(c.effective_cost for c in candidates)
c_max = max(c.effective_cost for c in candidates)
budget = c_min + alpha * (c_max - c_min)

result = solve_budget_lp(candidates, budget=budget)
if not result.feasible:
    raise RuntimeError("no provider fits the budget")

print(result.weights)
print(result.expected_cost)
print(result.objective)
```

The returned `weights` map provider name to routing probability. Your router
should sample one provider from this distribution, then commit the provider's
quota or concurrency state at dispatch time.

## Objective Tie-Breaking

If multiple providers have nearly identical latency objectives, use
`cost_tiebroken_objective()` before solving so lower effective cost wins stable
ties.

```python
from llm_routewise.core import BudgetLPCandidate, cost_tiebroken_objective, solve_budget_lp

latency_ms = [100.0, 100.0, 120.0]
costs = [0.003, 0.001, 0.0005]
objective = cost_tiebroken_objective(latency_ms, costs)

result = solve_budget_lp(
    [
        BudgetLPCandidate("a", objective[0], costs[0]),
        BudgetLPCandidate("b", objective[1], costs[1]),
        BudgetLPCandidate("c", objective[2], costs[2]),
    ],
    budget=0.002,
)
```

## Probability-Targeted Hedging

RouteWise hedging is checkpoint-based. The primary request is dispatched first.
At each checkpoint, your router computes whether dispatching a backup still
achieves the target SLO success probability. If a later checkpoint remains
feasible, wait. Dispatch at the latest feasible checkpoint.

`combined_success_probability()` returns the probability conditional on the
primary request not having completed by the checkpoint.

```python
from llm_routewise.core import (
    BackupCandidate,
    combined_success_probability,
    hedge_checkpoints_for_slo,
    select_probability_backup,
)

slo_ms = 3000.0
checkpoints_sec = hedge_checkpoints_for_slo(slo_ms)

def primary_cdf_ms(value_ms: float) -> float:
    # Return P(primary TTFT <= value_ms) from your rolling profile.
    ...

def backup_cdf_ms(value_ms: float) -> float:
    # Return P(backup TTFT <= value_ms) from your rolling profile.
    ...

elapsed_ms = 1500.0
success_probability = combined_success_probability(
    primary_cdf_ms,
    backup_cdf_ms,
    elapsed_ms=elapsed_ms,
    slo_ms=slo_ms,
)

candidate = BackupCandidate(
    provider="backup_provider",
    success_probability=success_probability,
    marginal_cost=0.0008,
    true_mean_ms=220.0,
)

selected = select_probability_backup([candidate])
if selected is not None:
    print(selected.provider)
```

`select_probability_backup()` filters out candidates below their
`success_target`, then chooses the cheapest feasible backup, breaking ties by
higher success probability and then lower mean latency.

## Decision Types

`RoutingDecision` and `HedgeDispatch` are lightweight data contracts. They are
useful if you want your router's policy layer to be side-effect-free while an
execution layer owns HTTP calls and provider state mutation.

```python
from llm_routewise.core import HedgeDispatch, RoutingDecision

decision = RoutingDecision(
    primary_provider="provider_a",
    hedge_checkpoints_sec=(0.75, 1.50, 2.25),
    metadata={"weights": {"provider_a": 1.0}},
)

dispatch = HedgeDispatch(
    backup_provider="provider_b",
    metadata={"combined_success": 0.992},
)
```

## API Reference

These are the stable core contracts. The functions are pure: they do not mutate
provider state, update counters, sample randomness, or perform network I/O.

### Data Contracts

- `BudgetLPCandidate`: one feasible provider snapshot for the budget LP.
  Fields: `name`, `objective`, `effective_cost`, `metadata`.
- `BudgetLPResult`: LP solver output with named routing probabilities.
  Fields: `feasible`, `weights`, `budget`, `objective`, `expected_cost`,
  `status`.
- `BackupCandidate`: one backup option scored at a checkpoint.
  Fields: `provider`, `success_probability`, `marginal_cost`, `true_mean_ms`,
  `success_target`.
- `CheckpointBackupDispatch`: optional advanced contract for in-flight
  checkpoint selectors. Fields: `backup`, `elapsed_sec`,
  `success_probability`, `release`, `metadata`.
- `RoutingDecision`: side-effect-free primary routing result.
  Fields: `primary_provider`, `hedge_checkpoints_sec`, `metadata`.
- `HedgeDispatch`: side-effect-free request to dispatch a backup.
  Fields: `backup_provider`, `metadata`.

`RoutingDecision` also accepts the legacy keyword `hedge_checkpoints` as an
alias for `hedge_checkpoints_sec`. Do not pass both names in the same call.

### Functions

- `effective_cost() -> float`: normalizes API, quota, or concurrency provider
  snapshots into USD/request.
- `quota_effective_cost() -> float`: lower-level quota opportunity-cost helper.
- `concurrency_effective_cost() -> float`: lower-level concurrency
  opportunity-cost helper.
- `scarcity_price() -> float`: advanced helper for explicit scarcity curves.
- `solve_budget_lp() -> BudgetLPResult`: exact solver for the one-budget
  RouteWise LP.
- `cost_tiebroken_objective() -> list[float]`: adds a tiny cost-based
  perturbation for stable latency ties.
- `hedge_checkpoints_for_slo() -> tuple[float, ...]`: converts an SLO in
  milliseconds into elapsed checkpoint seconds.
- `combined_success_probability() -> float`: conditional SLO success
  probability if hedging now.
- `select_probability_backup() -> BackupCandidate | None`: cheapest feasible
  backup under each candidate's `success_target`.
- `has_feasible_backup() -> bool`: convenience helper for backup feasibility
  checks.

`CheckpointBackupSelector` is a protocol for advanced integrations that want to
encapsulate checkpoint-specific backup admission and release logic.

## Failure Semantics

- `effective_cost()` raises `ValueError` for unknown tiers. Missing quota or
  concurrency snapshots return `0.0`.
- `scarcity_price()` raises `ValueError` for non-finite inputs or invalid
  envelopes such as `L <= 0` or `L >= U` for L/U curves.
- `solve_budget_lp()` reports invalid or infeasible inputs via
  `BudgetLPResult(feasible=False, weights={})`; it does not raise for empty
  candidates, duplicate names, non-finite values, or budgets below all provider
  costs.
- `cost_tiebroken_objective()` raises `ValueError` when latency and cost arrays
  have different lengths.
- `select_probability_backup()` returns `None` when no candidate reaches its
  `success_target`.
- `hedge_checkpoints_for_slo()` returns `()` for non-positive SLOs.

CDF callables passed to `combined_success_probability()` should return
probabilities in `[0, 1]`. RouteWise clamps the combined result, but your
integration is responsible for using calibrated latency profiles.

## Integration Pattern

A typical production integration has three layers:

1. State layer: provider metadata, quota counters, concurrency counters,
   rolling latency profiles, and request output prediction.
2. Policy layer: pure RouteWise math using `llm_routewise.core`.
3. Execution layer: sends requests, commits/refunds capacity where appropriate,
   races hedged requests, records outcomes, and updates profiles.

Pseudo-code:

```python
def route_request(request, state):
    feasible = [p for p in state.providers if p.can_admit()]
    L, U = state.cost_envelope

    candidates = []
    for p in feasible:
        cost = effective_cost(
            p.tier,
            request_cost_usd=p.estimated_request_cost(request),
            quota_fraction_used=p.quota_fraction_used(),
            concurrency_utilization=p.concurrency_utilization(),
            L=L,
            U=U,
        )
        candidates.append(
            BudgetLPCandidate(
                name=p.name,
                objective=p.rolling_mean_ttft_ms(),
                effective_cost=cost,
            )
        )

    c_min = min(c.effective_cost for c in candidates)
    c_max = max(c.effective_cost for c in candidates)
    budget = c_min + request.alpha * (c_max - c_min)
    result = solve_budget_lp(candidates, budget=budget)
    return sample_provider(result.weights)
```

## What Is Not Core

These modules are useful but not part of the stable core library contract:

- `llm_routewise.sim`: trace-driven simulator and research policies.
- `experiments`: paper section runners, ablation harnesses, and live replay.
- `plots`: figure generation scripts.

Use them as references or artifacts, not as stable dependencies for a library
consumer.

## Compatibility Policy

For a library release, treat these as stable:

- `llm_routewise.core` import path
- dataclass field names for `BudgetLPCandidate`, `BudgetLPResult`,
  `BackupCandidate`, `CheckpointBackupDispatch`, `RoutingDecision`, and
  `HedgeDispatch`
- function signatures documented in this file

The top-level facade documented in [API.md](API.md) is the supported `0.1.0`
preview surface. Other modules outside `llm_routewise.core` are repository
implementation details unless a public document says otherwise.
