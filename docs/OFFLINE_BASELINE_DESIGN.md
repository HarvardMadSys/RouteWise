# Offline Baseline Design

> Design document for making the RouteWise simulator offline baseline usable
> for cost-layer concurrency and joint quota+concurrency experiments.
>
> Last updated: 2026-05-07.

## 1. TL;DR

The current `offline` policy in `experiments/simulation/cost_layer.py` is good
enough for smoke tests, but it is not a clean paper baseline for joint
`S_Q + S_C + S_A` experiments.

The fix should be an experiment-local oracle adapter:

```text
experiments/simulation/offline_oracle.py
```

It should expose one public runner used by `cost_layer.py`:

```python
run_offline_oracle_policy(scenario, requests, seed, retain_records=True) -> Run
```

The adapter should support these cost-oracle kinds:

| Mode | Scenario | Baseline semantics |
|---|---|---|
| `stage_q_exact_single_window` | `S_Q + S_A`, one quota window | exact unit-weight quota oracle |
| `stage_q_greedy_value_multi_window` | `S_Q + S_A`, multiple quota windows | deterministic value-greedy quota baseline |
| `stage_c_greedy_interval` | `S_C + S_A` | extraction-compatible interval baseline |
| `stage_c_exact` | `S_C + S_A` | min-cost-flow exact fixed-model concurrency oracle |
| `stage_qc_exact` | `S_Q + S_C + S_A` | bounded exact joint MILP oracle |
| `stage_qc_best_decomposition` | `S_Q + S_C + S_A` | non-oracle scalable fallback, disabled by default |

Do not directly wire the legacy `experiments/offline_stage/strategies/stage2_optimal.py`
into the section runner. It uses the old `rwsim.offline.schemas.Request`
surface, daily quota assumptions, and a separate simulator result schema. Reuse
its formulation as reference, not as the new integration point.

## 2. Problem

The current section-local offline path is embedded in
`experiments/simulation/cost_layer.py`:

```text
_offline_assignments()
_quota_assignments()
_concurrency_assignments()
_offline_record()
```

That implementation has two important limitations:

1. **It composes independent decisions.** It first creates API assignments, then
   overlays quota assignments, then overlays concurrency assignments. In a
   future joint scenario, the concurrency overlay can overwrite quota decisions
   made independently. That is not a joint oracle.
2. **The concurrency path is a greedy first-fit heuristic.** It sorts requests
   by API value and places each request in the first fitting slot. This is a
   useful fast baseline, but it should not be described as the offline optimum
   without validation.

For `1.3.1` concurrency-only, the current greedy baseline is reproducible and
useful, but the paper-grade path is `stage_c_exact`. For `1.3.2` joint, we
need `stage_qc_exact` evidence before any RouteWise-vs-oracle claim. A
decomposition row can be useful for engineering comparisons, but it is not the
paper oracle.

## 3. Objective Semantics

The offline baseline is cost-only. It should match the cost-layer experiment
contract:

- Subscription fixed fees are already accounted for in section summaries.
- Offline assignment minimizes API spend for a fixed purchased capacity count.
- Every request served by `S_Q` or `S_C` has zero marginal cost.
- Every request not served by subscription capacity goes to the cheapest `S_A`
  provider by split input/output token price.
- Latency is reported for sanity, but the offline objective does not optimize
  latency.

Request value is the API cost avoided by serving that request through a
subscription:

```text
value_i = cheapest_api.input_price * input_tokens_i
        + cheapest_api.output_price * output_tokens_i
```

This is the value used by quota ordering, concurrency ordering, and any exact
solver objective.

## 4. Capacity Semantics

### 4.1 Quota

Quota is unit-weight in the current simulator:

```text
one request consumes one quota unit in every active quota window
```

The oracle must support both:

- single-window plans such as Chutes;
- multi-window plans such as MiniMax 5-hour + weekly caps.

Window anchoring should match the simulator cost-layer convention: windows are
anchored at the first request timestamp for simulation sections. Do not mix in
the real-evaluation epoch-aligned anchor in this path.

### 4.2 Concurrency

Concurrency is zero-wait for cost-layer simulation:

```text
request starts at arrival time or falls back to API
```

Do not allow offline queueing in the main baseline. Queueing changes the
problem and would no longer compare against the online simulator execution
semantics.

Use deterministic service intervals for the offline baseline:

```text
start_i = arrival_i
end_i   = arrival_i + p50_ttft(provider) + output_tokens_i / p50_tps(provider)
```

This matches the existing cost-layer offline path and avoids coupling the
oracle to random latency samples from one online run. The result is a lower
bound under deterministic p50 service-time assumptions, not a replay of any one
random execution.

For weighted plans such as Featherless Premium, the current cost-layer model
fixes one model class for the scenario. Therefore all requests in a cell share
one concurrency weight:

```text
effective_slots = floor(capacity_units / model_concurrency_cost)
```

The design can support general weighted requests later, but the first baseline
should optimize the fixed-model case because that is what `1.3.1` and the
planned `1.3.2` grid use.

## 5. Oracle Modes

### 5.1 Stage Q: Quota Oracle Kinds

For each quota provider and each quota-window constraint, choose the
highest-value compatible requests subject to every active window cap.

For the single-window unit-weight case, use:

```text
stage_q_exact_single_window
```

This is exactly:

```text
for each window:
  sort requests by value descending
  assign top Q requests to S_Q
```

For multi-window unit-weight quota, use:

```text
stage_q_greedy_value_multi_window
```

This is a deterministic greedy baseline with feasibility against all windows:

```text
sort all requests by value descending
for request in sorted requests:
  if every quota window has capacity for request:
    assign request to S_Q
    charge every quota window
```

This is exact only for one quota window. The multi-window kind must not be
reported as exact, because overlapping quota windows turn the selection problem
into a multi-constraint packing problem. If we later need a formal
multi-window optimum, add a small ILP validator before making a paper claim
about that specific case.

### 5.2 Stage C: Concurrency Oracle

For `S_C + S_A`, Phase 1 should preserve the current fast baseline but move it
into `offline_oracle.py` and name it explicitly:

```text
stage_c_greedy_interval
```

Algorithm:

```text
assign all requests to cheapest S_A
sort requests by value descending
for request in sorted requests:
  if [arrival, finish) fits in any concurrency slot:
    assign request to S_C
```

This is the current behavior, so moving it first gives us a safe refactor.

Then implement the exact fixed-model solver:

```text
stage_c_exact
```

For fixed-model concurrency, exact zero-wait interval selection is a
polynomial-time min-cost-flow problem:

- one time node per distinct interval endpoint;
- idle arcs between adjacent time nodes with capacity `effective_slots`;
- one request arc from `start_i` to `end_i` with capacity 1 and cost
  `-value_i`;
- the selected interval arcs are the requests served by `S_C`.

This does not require MILP for the Featherless fixed-model setting used in
`1.3.1`. Run the exact min-cost-flow solver on the full 30-day BurstGPT trace;
if it completes in under 30 minutes per cell, make `stage_c_exact` the default
for `S_C + S_A` and keep `stage_c_greedy_interval` only as a compatibility and
runtime fallback. If full-trace exact solving is too slow, the paper must keep
the greedy caveat until the exact path is optimized.

### 5.3 Stage QC: Joint Baseline

For `S_Q + S_C + S_A`, the exact problem is a joint packing problem:

```text
each request can choose at most one of:
  quota slot
  concurrency interval
  API fallback
```

This should not be implemented by independently running Stage Q and Stage C and
overwriting assignments. That double-counts scarce capacity during planning and
can waste one tier on requests served by the other.

The exact bounded joint oracle should be named:

```text
stage_qc_exact
```

This is the paper-semantics oracle for bounded slices and smoke runs. It should
fail fast, not silently degrade, when the trace is too large for the available
solver.

If a scalable non-oracle fallback is needed, name it honestly:

```text
stage_qc_best_decomposition
```

Compute the best of a small set of deterministic decompositions:

1. `quota_first`: Stage Q auto kind, then the configured Stage C kind on
   remaining API requests.
2. `concurrency_first`: the configured Stage C kind, then Stage Q auto kind on
   remaining API requests.
3. `value_partition`: reserve the top-value band for quota and let concurrency
   handle the next band; sweep partition percentiles `[0.25, 0.50, 0.75]`.

The configured Stage C kind should be `stage_c_exact` once Phase 4 promotes it;
before then, decomposition rows must record that they used
`stage_c_greedy_interval` internally.

Return the assignment with the lowest API cost.

The exact MILP formulation is:

```text
stage_qc_exact / stage_qc_exact_slice
```

Variables:

```text
x_q[i] in {0,1}
x_c[i] in {0,1}
x_a[i] in {0,1}
x_q[i] + x_c[i] + x_a[i] = 1
```

Quota constraints:

```text
sum_i x_q[i] <= quota_window_capacity
```

Concurrency constraints:

```text
sum_i active_i(t) * x_c[i] <= effective_slots
```

Objective:

```text
minimize sum_i x_a[i] * value_i
```

Use this first on bounded slices, e.g. 5k-10k requests or one selected busy
hour. The full-trace paper baseline is acceptable only after the same exact
semantics are made scalable enough for the selected `1.3.2` cells, or after an
explicitly reported exact-slice gap justifies a non-oracle fallback.

## 6. API Shape

Add:

```text
experiments/simulation/offline_oracle.py
```

Suggested public surface:

```python
class OfflineOracleKind(str, Enum):
    API_ONLY = "api_only"
    STAGE_Q_EXACT_SINGLE_WINDOW = "stage_q_exact_single_window"
    STAGE_Q_GREEDY_VALUE_MULTI_WINDOW = "stage_q_greedy_value_multi_window"
    STAGE_C_GREEDY_INTERVAL = "stage_c_greedy_interval"
    STAGE_C_EXACT = "stage_c_exact"
    STAGE_QC_EXACT = "stage_qc_exact"
    STAGE_QC_BEST_DECOMPOSITION = "stage_qc_best_decomposition"
    STAGE_QC_EXACT_SLICE = "stage_qc_exact_slice"

@dataclass(frozen=True)
class OfflineAssignment:
    request_id: int
    provider_name: str
    provider_tier: ProviderTier
    avoided_api_cost_usd: float
    oracle_kind: OfflineOracleKind
    start_sec: float | None = None
    finish_sec: float | None = None

def assign_offline(
    scenario: ScenarioConfig,
    requests: list[Request],
    *,
    kind: OfflineOracleKind | str = "auto",
) -> dict[int, OfflineAssignment]:
    ...

def run_offline_oracle_policy(
    scenario: ScenarioConfig,
    requests: list[Request],
    seed: int,
    retain_records: bool = True,
) -> Run:
    ...
```

`kind="auto"` should map by scenario tiers:

| Tiers | Auto kind |
|---|---|
| `S_A` only | `API_ONLY` |
| `S_Q + S_A`, one quota window | `STAGE_Q_EXACT_SINGLE_WINDOW` |
| `S_Q + S_A`, multiple quota windows | `STAGE_Q_GREEDY_VALUE_MULTI_WINDOW` |
| `S_C + S_A` | `STAGE_C_EXACT` |
| `S_Q + S_C + S_A` | `STAGE_QC_EXACT` for bounded exact runs |

`run_offline_oracle_policy()` should emit normal `PerRequestRecord` rows so
that `common.summarize_runs()` remains the single summary path.
The `policy` field stays `"offline"` for grouping and figure compatibility;
the selected oracle kind lives in `metadata["offline_oracle_kind"]`.
The `seed` argument is accepted for interface uniformity with online runners;
the offline oracle itself is deterministic under stable value tie-breaks.

## 7. Integration Plan

### Phase 1: Safe Extraction

Move the current offline helpers out of `cost_layer.py` into
`offline_oracle.py` without changing results.

Acceptance:

- existing tests in `tests/unit/simulation/test_cost_layer.py` pass;
- 1.2 and 1.3 existing `offline` summary rows have `rel_diff < 1e-9` for
  numerical cost fields after rerun;
- per-request metadata includes `offline_oracle_kind`.

### Phase 2: Baseline Naming and Diagnostics

Expose the oracle kind in summary metadata:

```text
offline_oracle_kind
offline_avoided_api_cost_usd
offline_subscription_value_usd
```

For concurrency rows, include:

```text
offline_concurrency_selected_requests
offline_concurrency_capacity_units
offline_concurrency_effective_slots
```

This makes it impossible to confuse a greedy interval baseline with an exact
lower bound.

### Phase 3: Joint Scenario Support

Before running `1.3.2`, add a cost-layer joint scenario builder:

```text
joint__quota_plan=chutes__q=14__concurrency_plan=featherless_premium__c=12__model=sharegpt
```

Provider set:

```text
S_Q: one aggregate quota provider
S_C: one aggregate weighted concurrency provider
S_A: cheap/mid/expensive API fallback ladder
```

Hold latency equal for the first cost-layer joint run. Do not reuse the old
Stage 3 setting where `S_C` is faster than `S_Q`; that experiment confounds
capacity ranking with latency preference.

Blocking code changes:

- update `experiments/subscription_plans.yaml` so Chutes and
  Featherless Premium are both eligible for `cost_layer_joint`;
- make the joint scenario builder accept explicit CLI/grid counts above the
  yaml `subscription_counts` defaults and validate only `>0`, matching the
  current `1.3.1` `--concurrency-counts` behavior;
- update `experiments/simulation/common.py` where joint subscription and
  concurrency summaries currently raise
  `ValueError("joint subscription/concurrency summaries are not supported yet")`;
  the joint summary should add quota fixed fees and concurrency fixed fees into
  one `subscription_fixed_cost_usd` value.

Acceptance:

- `routewise simulator cost-layer` can run one smoke joint cell end to end;
- the smoke row includes both selected quota `q` and selected concurrency `c`;
- explicit grid counts such as `c=12` are accepted even when yaml default
  `subscription_counts` lists only small smoke values;
- summary fixed cost equals quota fixed cost plus concurrency fixed cost;
- latency distributions for `S_Q` and `S_C` are equal in the cost-layer joint
  scenario config.
- offline joint smoke rows record
  `metadata["offline_oracle_kind"] == "stage_qc_exact"`.

### Phase 4: Full-Trace Exact Scaling

Add optional exact solver entrypoints under the same module or a sibling:

```text
experiments/simulation/offline_oracle_exact.py
```

`stage_c_exact` is the default concurrency baseline. The remaining paper gate is
making `stage_qc_exact` scale to the selected full-trace `1.3.2` cells, or
clearly labeling any non-exact fallback.

The exact joint MILP solver is selected with environment variables so local
development can keep the CBC default while gpu2 can use the available Gurobi
license:

```bash
ROUTEWISE_OFFLINE_MILP_SOLVER=gurobi \
ROUTEWISE_OFFLINE_MILP_SEED=42 \
ROUTEWISE_OFFLINE_MILP_TIME_LIMIT_SEC=300 \
ROUTEWISE_OFFLINE_JOINT_MAX_REQUESTS=50000 \
routewise simulator cost-layer --scenario joint --policy offline ...
```

Supported solver names are `cbc`, `gurobi`, and `gurobi_cmd`. The default CBC
path must set `randomCbcSeed`; the Gurobi path must set `Seed` and record
solver, seed, time limit, and request cap in per-request metadata.

Acceptance:

- `stage_c_exact` runs on the full `1.3.1` trace in under 30 minutes per cell,
  or the paper keeps the `stage_c_greedy_interval` caveat;
- on small synthetic cases, exact Stage QC chooses the known optimal assignment;
- bounded joint slices can run with `ROUTEWISE_OFFLINE_MILP_SOLVER=gurobi` on
  gpu2;
- measured exact-vs-online gap is reported before paper figures use the
  full-trace joint baseline.

## 8. Tests

Add focused unit tests before full sweeps:

1. API-only offline sends all requests to cheapest API.
2. Stage Q exact single-window selects highest-value requests within one quota
   window.
3. Stage Q multi-window greedy rejects assignments that violate either window.
4. Stage C greedy interval baseline rejects overlapping requests when capacity is
   full.
5. Stage C exact matches the known optimal interval selection on a small
   overlapping case.
6. Stage C baselines handle `WeightedConcurrencyState.limit`.
7. Stage QC never assigns one request to both quota and concurrency.
8. Stage QC exact chooses the known optimal quota/concurrency split on a small
   overlapping case.
9. Stage QC exact records solver name, solver seed, and joint request cap in
   per-request metadata.
10. Summary rows include `offline_oracle_kind`.

## 9. What Counts As Done

The offline baseline is "done enough" for `1.3.2` when:

- `cost_layer.py` uses `offline_oracle.py` for all `offline` runs;
- `1.3.1` uses `stage_c_exact` as the default concurrency offline row if the
  full-trace min-cost-flow runtime target is met;
- joint scenarios can run through the same `routewise simulator cost-layer`
  surface;
- joint summary supports combined quota + concurrency fixed fees;
- the joint offline row is labeled `stage_qc_exact` only when exact solving is
  used;
- bounded exact slices quantify the RouteWise-vs-oracle gap;
- the result table can report:

```text
RouteWise total cost / offline baseline total cost
greedy total cost / offline baseline total cost
tier mix: S_Q vs S_C vs S_A
selected q, selected c
```

## 10. Non-Goals

- Do not optimize latency in this offline baseline.
- Do not model queueing in the default cost-layer oracle.
- Do not require Gurobi for the default path.
- Do not rename `offline_stage/` in this change.
- Do not change production `RouteWisePolicy` while building the oracle.
- Do not commit to a paper-grade joint `1.3.2` claim before the 2026-05-08
  abstract deadline. The abstract should reference `1.3.1` only unless exact
  joint evidence exists before then.
