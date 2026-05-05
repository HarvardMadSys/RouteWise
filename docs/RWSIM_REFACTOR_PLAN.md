# rwsim Refactor Plan

> Decision document. Defines the target architecture and migration order for
> simplifying `rwsim` around a single simulator, flat policies, explicit policy
> presets, and real-world empirical distributions. `rwsim/offline/` is left
> untouched in this refactor.

Last updated: 2026-05-05.

This replaces the earlier 4-stage pipeline refactor proposal. That proposal
mixed paper notation with implementation structure. The paper can still explain
RouteWise as value estimation, cost routing, latency control, and hedging; the
code does not need to force every policy through those four Protocols.

---

## 1. TL;DR

The final architecture has one central rule:

```text
Policy decides. Simulator executes. World state belongs to the simulator.
Policy learning state belongs to the policy.
```

Refactors land in this order:

1. **Introduce flat Policy interface** - one ABC with `route()` and
   `observe()`. No mandatory 4-stage pipeline.
2. **Replace strategy implementation with policy presets** - update callers,
   configs, docs, and CLI to use policy names. Delete `rwsim/strategies/` in
   the same landed refactor. Do not keep a committed compatibility layer.
3. **Migrate the actual paper methods** - Greedy-cost, Greedy-latency,
   Random, complete RouteWise, and RouteWise ablations.
4. **Add empirical distributions** - raw real-world samples live under
   `experiments/simulation/profiles/`; generic sampling classes live in
   `rwsim/world/`.
5. **Delete dead pipeline-composer code** - remove the 4-stage
   `PolicyPipelineSpec` / `StageSpec` system after policy presets are the only
   path.
6. **Defer offline** - `rwsim/offline/` remains isolated cost-oracle /
   reproducibility code until a separate decision document handles it.

Safety net:

- `tests/golden_capture.py --mode compare` gates behavior-preserving commits.
  When policy names or intentionally dropped historical variants change the
  artifact surface, regenerate signed-off policy goldens instead of pretending
  the old strategy goldens are still authoritative.
- `pytest -m "not slow"` must pass unless a failure is explicitly documented as
  pre-existing and unrelated.
- Refactor commits should not leave `rwsim/strategies/` or `--strategy` as a
  compatibility surface. Update scripts and docs to `--policy`.

---

## 2. First Principles

RouteWise simulates one decision problem:

```text
Given a stream of requests and a set of providers,
choose where to send each request, whether to hedge, and record the outcome.
```

That requires six core concepts:

| Concept | Purpose | Owner |
|---|---|---|
| `Request` | Input item arriving at time `t` | `rwsim/schemas.py` |
| `Provider` | Cost, latency, quota, concurrency metadata | `rwsim/world/` |
| `Distribution` | Sampling and quantile/CDF access | `rwsim/world/` |
| `SimulationState` | Mutable world state during a run | `rwsim/engine/` |
| `Policy` | Decision rule plus learning state | `rwsim/policies/` |
| `Simulator` | Time loop, execution, state mutation, metrics | `rwsim/engine/` |

Everything else is secondary:

- A paper "strategy" may appear in prose, but code should call it a policy
  preset.
- A paper "stage" is explanatory notation, not a required code Protocol.
- An experiment chooses scenario, workload, policy preset, seed, and output
  path. It does not implement routing logic.

---

## 3. Target Shape

```text
rwsim/
  schemas.py                 # Request / RoutingDecision / RoutingOutcome

  engine/
    simulator.py             # one canonical request loop
    state.py                 # SimulationState snapshot only

  world/
    providers.py             # static provider metadata
    capacity.py              # runtime quota + concurrency accounting
    distributions.py         # Uniform / Normal / LogNormal
    empirical.py             # EmpiricalDistribution, optional percentile helper

  data/
    loader.py                # generic CSV/JSONL -> Request trace loaders/cache

  policies/
    base.py                  # Policy ABC
    baselines.py             # Greedy-cost / greedy-latency / random baselines
    routewise.py             # complete RouteWise plus module-local helpers
    __init__.py              # POLICY_CLASSES + build_policy()

  metrics/
    run.py                   # per-request records + Run aggregate (sim + real)

experiments/
  simulation/
    configs/
    presets.yaml             # policy preset name -> policy class + params
    profiles/                # real-world latency samples and workload traces
    suites/

  real_evaluation/
  offline_stage/             # may shrink to harness only, then rename later
  estimator_ablation/

plots/
docs/
```

Gone at the end:

```text
rwsim/strategies/            # implementation layer removed
rwsim/policies/composer.py   # 4-stage dataclass composer removed
rwsim/policies/{value_estimators,cost_routers,latency_routers,hedgers}/
                             # stage-era public modules removed
rwsim/world/shadow_price.py   # RouteWise cost helper, not world state
rwsim/world/workload.py       # synthetic generator removed from target path
rwsim/registry.py            # generic registry helper removed
```

The CLI flag should become `--policy`. Keeping `--strategy` would preserve the
conceptual confusion this refactor is meant to remove.

`policies/routewise.py` starts as one file. LP budget, effective cost,
probability-target hedging, and explorer/probe accounting are RouteWise
internals, so keep them as module-local helpers until real code size gives a
better split. Do not mirror paper sections into files up front.

---

## 4. Policy / Simulator Contract

### 4.1 Policy

`Policy` is the only algorithm interface the simulator knows.

```python
class Policy(Protocol):
    def route(
        self,
        request: Request,
        state: SimulationState,
    ) -> RoutingDecision:
        ...

    def tick(
        self,
        request: Request,
        decision: RoutingDecision,
        elapsed: float,
        state: SimulationState,
    ) -> HedgeDispatch | None:
        ...

    def observe(
        self,
        request: Request,
        decision: RoutingDecision,
        outcome: RoutingOutcome,
    ) -> None:
        ...
```

`route()` decides the primary provider and declares any in-flight checkpoint
schedule via `RoutingDecision.hedge_checkpoints`. It may read a state snapshot,
provider metadata, and policy-owned learning state.

`tick()` is called by the simulator at each elapsed time in
`decision.hedge_checkpoints`, while the request is still in flight. It returns
a `HedgeDispatch` to issue a backup at that moment, or `None` to keep waiting.
The default implementation returns `None`; baselines (Greedy / Random) inherit
the no-op. RouteWise overrides it to recompute $P_{\text{succ}}(t)$ against the
current `state` (queue depth has likely changed since `route()`) and its own
latest latency profiles $\hat{F}_j$.

This is the contract slot for the locked design decision in
`docs/EXPERIMENT_LAYOUT.md` §5: "$P_{\text{succ}}(t)$ is computed at multiple
checkpoints (e.g. P25, P50, P75, P90 of remaining SLO), not once at dispatch
time." The set of checkpoint times is policy-owned (declared on the decision),
not hard-coded by the simulator.

`observe()` updates policy-owned learning state after the simulator has
executed the request, including any hedge that fired. It receives the request
and decision as explicit inputs so `RoutingOutcome` does not become a giant
catch-all object. Examples:

- LP weights.
- SWRR sampler state.
- EMA or histogram estimators.
- Provider latency profiles used by that policy. Explorer behaviour is exactly
  this: feeding the backup's observed TTFT into $\hat{F}_j$ is a `True`/`False`
  toggle inside `RouteWisePolicy.observe()`, not a separate algorithm.

This is deliberately not "pure stages". Online routing learns. Hiding that
learning inside the engine would make the engine know every policy's internals.

`RoutingDecision` carries the per-request hedge schedule:

```python
@dataclass
class RoutingDecision:
    primary: str
    hedge_checkpoints: tuple[float, ...] = ()  # seconds since dispatch
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

For non-hedging policies the tuple is empty and `tick()` is never called.

### 4.2 Simulator

The simulator owns the world and execution:

```python
class Simulator:
    def run(
        self,
        requests: Sequence[Request],
        policy: Policy,
        seed: int,
    ) -> Run:
        ...
```

The simulator is responsible for:

- Request loop and time progression.
- Provider latency/service-time sampling.
- Primary and hedge execution.
- Scheduling each in-flight request's `hedge_checkpoints` and calling
  `policy.tick(request, decision, elapsed, state)` at each one (until either a
  hedge fires or the primary completes).
- Quota and concurrency mutation.
- Per-user prefix-cache hit accounting (look up `state.user_last_provider` to
  decide whether the current dispatch is a same-provider repeat; apply the
  100% hit cost discount per `EXPERIMENT_LAYOUT.md` §5).
- Cost accounting.
- Metrics and per-request logs.
- Calling `policy.observe(request, decision, outcome)` after each completed
  request.

The simulator is not responsible for:

- LP solving.
- SWRR weights.
- EMA/histogram learning.
- Choosing primary or backup providers.
- Paper-specific ablation logic.

### 4.3 SimulationState

First version should be small:

```python
@dataclass
class SimulationState:
    now: float
    providers: Mapping[str, Provider]
    capacity: CapacityState
    user_last_provider: Mapping[str, str]  # user_id -> last dispatched provider
```

`user_last_provider` is a world fact (deterministic from routing history) that
the prefix-cache cost discount in `EXPERIMENT_LAYOUT.md` §5 reads from. It
lives in `SimulationState` rather than in any policy because it is shared,
identical for every policy, and would otherwise be re-derived by each policy
from its own observation log. Per-provider hit-rate variance is intentionally
not modelled (`EXPERIMENT_LAYOUT.md` §5 acknowledged limitation), so this map
is sufficient.

Rules:

- Engine mutates `SimulationState`.
- Policies may read it during `route()`.
- Policies do not mutate capacity or time.
- Policy-specific learning state does not go into `SimulationState`.
- Static provider metadata lives in `world/providers.py`: prices, limits,
  names, and configured distributions.
- Runtime usage lives in `world/capacity.py`: quota used, concurrency in
  flight, and derived raw utilization facts.
- World state exposes raw capacity facts such as quota fraction used and
  concurrency utilization. Shadow prices are RouteWise cost-model helpers:
  compute them inside `policies/routewise.py`, not in `world/`.
- Execution randomness belongs to the simulator. Policy randomness belongs to
  the policy instance. Do not share one mutable RNG through `SimulationState`.

Do not add `cost_profiles`, prefix-cache state, or estimator history until a
specific policy needs them and ownership is clear.

### 4.4 Per-request record and Run aggregate

`rwsim/metrics/` carries the canonical per-request record schema for both the
simulator AND real-evaluation experiments, not just the simulator's output.
This is the layer paper figures consume; sim-vs-real comparisons (5/4 meeting:
"real on left, simulation on right") require the two sides to emit the same
core fields.

```python
# rwsim/metrics/record.py

class Status(str, Enum):
    SUCCESS = "SUCCESS"
    REJECTED = "REJECTED"          # capacity refusal (sim)
    RATE_LIMITED = "RATE_LIMITED"  # real only
    ERROR = "ERROR"                # real only (network, timeout, 4xx/5xx)


@dataclass
class PerRequestRecord:
    """One request as observed by either the simulator or live evaluation.

    Core fields are universal: same name AND same physical meaning on both
    sides. Per-side extensions live in `metadata`, prefixed with `sim_*` or
    `real_*` so a merged record's source is unambiguous.

    Time/latency convention: every time-like core field is measured from the
    primary's dispatch instant (t=0). Wall-clock for real-eval rows goes into
    `metadata["real_wall_clock_ts"]`.
    """

    # Identity
    request_id: str
    elapsed_sec: float                  # seconds since run start; same physical meaning sim and real
    policy: str                         # paper-name preset, e.g. "routewise"

    # Workload
    prompt_tokens: int
    completion_tokens_budget: int | float | None  # output cap/estimate visible at routing time; used by LP cost estimate
    completion_tokens_actual: int | None  # observed after generation; what billing uses; None on rejection or pre-completion error

    # Routing decision and outcome
    primary_provider: str
    primary_tier: str                   # tier at routing time
    final_provider: str                 # whose response the user got: == primary if no hedge or primary won
    final_tier: str                     # tier of final_provider
    backup_provider: str | None         # only set when hedge dispatched
    backup_tier: str | None             # tier of backup_provider when hedge dispatched

    # Latency — all times are user-visible, measured from primary's dispatch (t=0)
    ttft_ms: float                      # USER-VISIBLE TTFT: time from primary dispatch to first response token of `final_provider`. For hedged backup-wins case = hedge_delay_ms + dispatch_overhead_ms + backup_local_ttft_ms. inf on rejection.
    e2e_ms: float | None                # USER-VISIBLE end-to-end: primary dispatch to last response token. None if sim does not model decode time (boundary decision below).
    primary_local_ttft_ms: float | None # primary's own TTFT, measured from primary dispatch; None if no provider request was admitted
    backup_local_ttft_ms: float | None  # backup's own TTFT, measured from backup dispatch; None if no hedge
    slo_ms: float | None                # SLO threshold for this request; legacy/test rows may be None until Run-level backfill
    slo_violated: bool                  # canonical SLO result when slo_ms is known; status != SUCCESS or user-visible ttft_ms > slo_ms

    # Cost
    total_cost_usd: float               # primary_cost + backup_cost (if hedged)
    primary_cost_usd: float
    backup_cost_usd: float | None       # None when no hedge

    # Hedging
    hedge_triggered: bool
    hedge_delay_ms: float | None        # primary-dispatch -> backup-dispatch; None when no hedge
    hedge_winner: str | None            # "primary" | "backup" | None when no hedge

    # Status
    status: Status
    error_class: str | None = None      # populated for ERROR / RATE_LIMITED

    # Per-side extensions (sim_* keys vs real_* keys)
    metadata: dict[str, Any] = field(default_factory=dict)
```

**Why every time-like field is primary-dispatch-relative.** SLO is a user
deadline. P99 is a user-experience metric. They are only sim/real-comparable
if both sides emit the same physical quantity. Local-to-each-provider TTFT
(`primary_local_ttft_ms`, `backup_local_ttft_ms`) stay as separate diagnostic
fields — they are useful for understanding *why* a hedge won/lost, but the
canonical SLO/P99 numbers all flow from `ttft_ms`.

**Why three tier fields, not one.** Cost can split across tiers when a hedge
fires across tiers (e.g. primary on `quota` with zero marginal cost, backup
on `api` with billed cost). `cost_by_tier()` needs `primary_tier` for
`primary_cost_usd` and `backup_tier` for `backup_cost_usd`. `final_tier` is
which provider the user actually saw. All three are cheap; resolving them
post-hoc from a `Run` would require an out-of-band provider→tier catalog
that defeats the point of a self-contained record.

**Why split `completion_tokens` into budget vs actual.** Routing-time cost
estimates use `completion_tokens_budget` (the cap visible to the policy when
it picks a provider). Billing uses `completion_tokens_actual` (what was
generated). They are physically different quantities and the gap matters for
cost-layer ablations (estimator-vs-oracle) and for real billing reconciliation.

**Sim-only metadata keys** (engine populates):

- `sim_lp_weights: dict[str, float]` — LP solution at routing time
- `sim_lp_budget: float` — `B_p(t)` at routing time
- `sim_c_eff: dict[str, float]` — effective cost vector at routing time
- `sim_true_p99_ms: float` — ground-truth provider P99 at decision time
- `sim_quota_fraction_used: dict[str, float]` — per-provider snapshot
- `sim_concurrency_utilization: dict[str, float]` — per-provider snapshot

**Real-only metadata keys** (recorder populates):

- `real_retry_count: int`
- `real_rate_limit_count: int`
- `real_http_status: int`
- `real_wall_clock_ts: float` — original wall-clock timestamp for the row
- `real_transport: str | None` — transport used for the request
- `real_retry_sleep_ms: float`
- `real_status: str` — raw transport status before canonicalization
- `real_lp_status: str | None`
- `real_lp_weights: dict[str, float] | None`
- `real_budget_usd: float | None`
- `real_reference_cost_usd: float | None`
- `real_tier_mix: dict[str, float] | None`

The `Run` aggregate is the figure-facing surface. Both sim and real produce
this type:

```python
# rwsim/metrics/run.py

@dataclass
class Run:
    records: list[PerRequestRecord]
    policy: str
    scenario_name: str
    source: Literal["simulation", "real"]

    # SLO / status
    def slo_violation_rate(self, slo_ms: float | None = None) -> float: ...
    def status_breakdown(self) -> dict[str, int]: ...

    # User-visible TTFT (the canonical paper number)
    def mean_ttft_ms(self) -> float: ...
    def p50_ms(self) -> float: ...
    def p90_ms(self) -> float: ...
    def p95_ms(self) -> float: ...
    def p99_ms(self) -> float: ...

    # End-to-end latency (when populated; sim may emit None per boundary decision below)
    def mean_e2e_ms(self) -> float: ...     # NaN when no rows have e2e_ms
    def p50_e2e_ms(self) -> float: ...      # NaN when no rows have e2e_ms
    def p90_e2e_ms(self) -> float: ...      # NaN when no rows have e2e_ms
    def p99_e2e_ms(self) -> float: ...      # NaN when no rows have e2e_ms

    # Cost
    def mean_cost_usd(self) -> float: ...
    def total_cost_usd(self) -> float: ...
    def cost_by_tier(self) -> dict[str, float]: ...      # uses primary_tier and backup_tier per record
    def cost_by_provider(self) -> dict[str, float]: ...  # same — splits across primary/backup providers

    # Hedging
    def hedge_rate(self) -> float: ...
    def hedge_winner_rate(self) -> dict[str, float]: ...  # primary/backup win fractions among triggered hedges

    # Routing mix
    def provider_fractions(self) -> dict[str, float]: ...    # by final_provider
    def tier_fractions(self) -> dict[str, float]: ...        # by final_tier

    # Vectorised column views, computed lazily for plot code
    @property
    def ttft_ms(self) -> np.ndarray: ...                 # user-visible TTFT
    @property
    def e2e_ms(self) -> np.ndarray: ...                  # user-visible E2E (NaN where None)
    @property
    def cost_usd(self) -> np.ndarray: ...                # total per request
    @property
    def hedge_triggered(self) -> np.ndarray: ...
    @property
    def elapsed_sec(self) -> np.ndarray: ...             # for rolling windows
```

`Run` is row-oriented internally (`list[PerRequestRecord]`) but exposes
column-view properties for vectorised aggregation. The two representations
are O(n) inter-convertible; do not maintain two separate dataclasses for
"row" and "column" runs.

Temporary migration surface: `Run.__init__` may accept legacy column kwargs
(`ttft_ms=`, `cost_usd=`, `provider=`, `timestamp=`, `hedge_triggered=`,
etc.) while plots/golden/suites are migrated. That path can only compute
per-record `slo_violated` when the caller also passes `slo_ms`; otherwise
callers must use `slo_violation_rate(slo_ms=...)` for SLO metrics. Delete
the column-kwargs constructor after downstream code constructs
`PerRequestRecord` rows directly.

Plot code (`plots/<section>/*.py`) consumes `Run` and core fields only.
Per-side extensions in `metadata` are read by diagnostic / appendix figures,
never by main paper figures, so cross-source plots stay shape-stable.

**Migration note.** The current `SimulationRun` is column-oriented and only
covers the simulator. Phase 0 of the schema migration:

1. Rename `SimulationRun` → `Run`; keep column-view properties.
2. Internal storage moves to `list[PerRequestRecord]`.
3. `experiments/real_evaluation/recorder.py` stops emitting its own per-request
   CSV schema and writes `PerRequestRecord` rows.
4. Existing CSV columns in real eval map to core fields; `retry_count` /
   `rate_limit_count` / `http_status` move into `metadata` with `real_*` prefix.

**Open boundary decisions** (need Murphy + Juncheng review before Phase 0
implementation):

| Decision | Question |
|---|---|
| `e2e_ms` in simulator | Sim today only models TTFT. Either (a) compute `e2e_ms = ttft_ms + completion_tokens_actual / tps_dist.sample()` and populate, or (b) leave `None` for sim and only populate for real. Affects whether E2E paper figures can include simulator points. |
| `hedge_delay_ms` zero-point | Confirm: measured from primary `route()` return (= primary dispatch instant in sim) → backup dispatch instant. Sim rounds to nearest ms; real records with submillisecond resolution but stored as float ms. |
| `backup_cost_usd` parity | Sim uses `provider.marginal_cost(prompt_tokens + completion_tokens_actual, now)` — i.e. the existing `marginal_cost(total_tokens, now)` formula matching primary. Real uses provider-billed amount. Confirm both are USD per request and match for the same provider given identical token counts. |
| `Status` sim coverage | Sim emits only `SUCCESS` and `REJECTED`. `RATE_LIMITED` and `ERROR` are real-only. Confirm sim's "no-capacity fallback succeeded" path maps to `SUCCESS` (not REJECTED). |
| `metadata` namespacing | Lock `sim_*` / `real_*` prefix. Any other prefix is a contract violation; readers can dispatch on prefix when merging cross-source data. |
| `completion_tokens_budget` source in sim | For S0–S3 synthetic scenarios, where does the budget come from? Options: (a) the scenario's known `response_tokens` (oracle), (b) a histogram-predictor estimate, (c) `None` (don't model the budget for sim). Affects what cost-layer ablations the simulator can express. |

---

## 5. Policy Presets

Named paper variants live as data, not implementations.

```yaml
# experiments/simulation/presets.yaml

greedy_cost:
  policy: BaselinePolicy
  params:
    mode: greedy_cost

greedy_latency:
  policy: BaselinePolicy
  params:
    mode: greedy_latency

random:
  policy: BaselinePolicy
  params:
    mode: random

ablation_lp_only:
  policy: RouteWisePolicy
  params:
    hedging: false
    explorer: false

ablation_lp_hedging:
  policy: RouteWisePolicy
  params:
    hedging: probability_target
    explorer: false

routewise:
  policy: RouteWisePolicy
  params:
    hedging: probability_target
    explorer: true
```

`rwsim/policies/__init__.py` carries the small class table and the loader.
There is no root registry, separate factory module, or separate class-registry
module:

```python
# rwsim/policies/__init__.py

POLICY_CLASSES = {
    "BaselinePolicy": BaselinePolicy,
    "RouteWisePolicy": RouteWisePolicy,
}


def build_policy(name: str, presets: Mapping[str, Any]) -> Policy:
    preset = presets[name]
    cls = POLICY_CLASSES[preset["policy"]]
    return cls(**preset.get("params", {}))
```

No `StageSpec`, no "none" stages, no factory registry for four artificial
slots. A complex policy can still use helper classes internally, but the public
architecture stays flat.

Existing code names are migration inputs, not target API. They should be
translated to paper/Notion names in configs, golden files, and plots during the
refactor rather than preserved as runtime aliases.

The example above uses one logical `RouteWisePolicy` family. Start with one
class and module-local helpers in `policies/routewise.py`. Split only after the
real implementation shows a clear cohesion boundary. The public preset names
stay flat either way.

The cost-oracle / offline lower-bound baseline remains in `offline_stage` for
this refactor. It is not a RouteWise online policy preset until the separate
offline decision is made.

---

## 6. Phase 0 - Lock the Contract

Deliverables:

1. Add or update the `Policy` ABC in `rwsim/policies/base.py`.
2. Add the minimal `SimulationState` dataclass in `rwsim/engine/state.py`.
3. Add `RoutingDecision` and `RoutingOutcome` fields needed by the simulator
   in `rwsim/schemas.py`.
4. **Lock the metrics schema contract from §4.4**:
   - Add `Status`, `PerRequestRecord` to `rwsim/metrics/record.py`.
   - Replace `SimulationRun` (or `StrategyRun` in pre-368e56d code) with
     `Run` in `rwsim/metrics/run.py`, wrapping `list[PerRequestRecord]`
     and exposing the §4.4 aggregation methods + column-view properties.
   - Resolve the 6 boundary decisions at the end of §4.4 (e2e_ms,
     hedge_delay_ms zero, backup_cost parity, sim Status coverage,
     metadata namespacing, completion_tokens_budget source in sim).
   - This deliverable is a contract change but no behaviour change yet:
     plot/golden/real-eval recorder still operate on the legacy schema
     until Phase 1.
5. Add architecture tests:
   - Policies expose `route()`, `tick()`, `observe()`.
   - Simulator owns capacity mutation.
   - Experiments import policy presets, not strategy implementations.
   - `Run.records` is `list[PerRequestRecord]` and each record has the
     §4.4 core fields.
6. Remove target-architecture `Policy`, `Executor`, and `MetricsRecorder`
   Protocols from `rwsim/engine/`; the engine should import `Policy` from
   `rwsim/policies/base.py` and emit `Run` objects from `rwsim/metrics/`
   (see §4.4 for the per-request schema and aggregate methods).
7. Update `docs/ALGORITHMS.md` to describe the flat policy contract and
   the §4.4 metrics schema.

The metrics schema lands in Phase 0 specifically because it is an
interface layer: plot code, golden capture, and the real-eval recorder
all consume it downstream. Letting it slip past Phase 0 means every
subsequent phase emits or consumes the wrong schema and has to be
revisited.

No behaviour changes in Phase 0.

---

## 7. Phase 1 - Replace `strategies/` With Policies

This phase is the actual cleanup. It is not a compatibility shim.

Landed target:

```text
rwsim/strategies/ does not exist.
routewise run ... --policy routewise
experiments/simulation/presets.yaml owns policy names.
```

The target policy names are exactly the paper/Notion names in snake case:

- `greedy_cost`
- `greedy_latency`
- `random`
- `routewise`
- `ablation_lp_only`
- `ablation_lp_hedging`

Historical code names are not target policy names. If an old figure depends on
one, either rename it to a paper-level ablation name or remove that dependency.

Historical implementation names to drop in this refactor:

- `oracle_per_window`
- `lp_mix`
- `lp_hedge`
- `lp_explorer`
- `lp_explorer_no_probe`
- `v2_only`
- `v2_p50_hedge`
- `v2_explorer`
- `v2_explorer_no_probe`
- `two_layer`
- `joint_nohedge`
- `joint_hedge`
- `joint_p50band_nohedge`
- `joint_p50band_hedge`

Their configs, golden entries, and plot palette/label entries are removed in
the same refactor unless they are renamed into an explicit paper-level
ablation. Do not keep names such as "legacy" or "compat" to avoid deciding.

File-level plan:

| Current | Target |
|---|---|
| `rwsim/strategies/baseline.py` | `rwsim/policies/baselines.py` |
| `rwsim/strategies/lp.py`, `latency_impl.py` | `rwsim/policies/routewise.py` one file; private helpers stay module-local |
| other files under `rwsim/strategies/` | fold useful logic into paper-named policies or delete |
| `rwsim/strategies/registry.py` | delete |
| `rwsim/policies/composer.py` | delete after preset loader exists |
| `rwsim/policies/{value_estimators,cost_routers,latency_routers,hedgers}/` | delete after RouteWise/Baseline policies own the useful logic |
| `rwsim/world/shadow_price.py` | move useful formulas into `rwsim/policies/routewise.py` |
| `rwsim/metrics/run.py::SimulationRun` | rename to `Run` (cross-source aggregate per §4.4); rewrite to wrap `list[PerRequestRecord]` with column-view properties. (Already renamed from legacy `StrategyRun` in commit 374f50a.) |
| `rwsim/world/workload.py` | delete after callers use trace loader/cache |
| `rwsim/registry.py` | delete |

Call-site updates happen in the same refactor:

- CLI: `--strategy` becomes `--policy`.
- Experiment configs: `strategy` keys become `policy` keys.
- Golden JSON: result labels use target policy names.
- Plots: labels use target policy names.
- Docs: "strategy implementation" language is removed.

Comparison against old strategy output is a development-time sanity check, not
the merge gate. The merge gate is that the new policy goldens reproduce
byte-identically across two captures, and key metrics align with the locked
paper decisions: cost, mean/P50/P95/P99 TTFT, SLO violation, hedge rate, and
provider mix. Do not land an environment flag such as `RWSIM_POLICY_ENGINE`,
and do not land a `rwsim/strategies` compatibility package.

Phase 1 lands when:

- `rg "rwsim\\.strategies|--strategy|STRATEGY_REGISTRY" rwsim experiments tests docs`
  has no production-path hits.
- New policy goldens are stable across repeated captures.
- `uv run pytest -m "not slow"` passes or unrelated failures are documented.

---

## 8. Phase 2 - Policy Set

Target paper-facing policy presets:

- `greedy_cost`
- `greedy_latency`
- `random`
- `routewise`
- `ablation_lp_only`
- `ablation_lp_hedging`

Optional appendix presets must be named by paper purpose, not by historical code
family. If the paper does not name the ablation, the code should not invent a
fancy public name for it.

Per-policy recipe:

1. Add or update a preset entry.
2. Implement one flat `Policy` class or extend `BaselinePolicy` /
   `RouteWisePolicy`.
3. Run old-vs-new comparison locally as a sanity check when useful.
4. Commit only the final policy path, with `rwsim/strategies/` already gone.

Rules:

- Do not create a new simulator loop for a policy.
- Do not add a new public Protocol unless two real policies need it.
- Private helper classes are fine inside a policy module.
- A paper ablation should normally be a preset param change, not a new loop.

---

## 9. Phase 3 - Real-World Empirical Distributions

Generic code:

```text
rwsim/world/empirical.py
```

```python
class EmpiricalDistribution:
    def __init__(self, samples: np.ndarray): ...
    def sample(self, rng: np.random.Generator, size: int = 1) -> np.ndarray: ...
    def quantile(self, q: float) -> float: ...
    def p50(self) -> float: ...
    def p95(self) -> float: ...
    def p99(self) -> float: ...
    def mean(self) -> float: ...
    def std(self) -> float: ...
    def cdf(self, value: float) -> float: ...
```

Experiment data:

```text
experiments/simulation/profiles/
  qwen3_24h.parquet
  sharegpt_trace.parquet
  pools.yaml
```

`pools.yaml`:

```yaml
rw3:
  samples: experiments/simulation/profiles/qwen3_24h.parquet
  providers: [WandB, DeepInfra, Novita]

rw8:
  samples: experiments/simulation/profiles/qwen3_24h.parquet
  providers: [WandB, DeepInfra, Google, Alibaba, Novita, Cerebras,
              SiliconFlow, AtlasCloud]
```

Rules:

- `rwsim/world/` contains generic distribution classes only.
- There is no target `rwsim/world/workload.py`. Request streams come from
  real traces loaded through `rwsim/data/loader.py` and experiment cache code.
- OpenRouter, RedNote, or model-specific sample files live under
  `experiments/`.
- Routers and hedgers do not know which concrete sample file produced an
  empirical distribution. They call the same distribution methods.
- `PercentileDistribution` is allowed as a low-priority utility, but it is not
  used for main latency-layer figures while raw samples exist.

---

## 10. What We Are Not Doing

| Not doing now | Reason |
|---|---|
| Mandatory 4-stage Protocols | Paper notation is useful, but forcing every policy through value/cost/latency/hedge slots creates no-op stages and config noise. |
| `PolicyPipelineSpec` / `StageSpec` composer factory | A flat policy class plus YAML preset is simpler and closer to the actual decision object. |
| Pure stages only | Online policies need learning state. `observe()` is part of the interface. |
| A god simulator that owns LP/SWRR/EMA state | That state belongs to policies. The simulator owns world execution state only. |
| Pre-splitting `routewise.py` into per-paper-section helper files | Paper notation is useful for explanation, not a file-boundary rule. Keep RouteWise in one module until real code shows a concrete split. |
| Engine recorder module | Per-request records and run aggregates belong in `rwsim/metrics/run.py`; the simulator can append records directly until another sink exists. |
| Generic registry infrastructure | A two-class policy table plus preset builder does not justify a root `rwsim/registry.py`. |
| A committed strategy compatibility layer | It would preserve the ambiguity this refactor removes. Temporary comparison helpers may exist locally but not in the landed code. |
| Offline merge | `rwsim/offline/` is isolated lower-bound / reproducibility code. Merge it only in a separate refactor. |
| Synthetic workload generation | Main experiments are trace-driven. Keep synthetic request streams out of the target architecture to avoid a second workload source. |
| Rename `distributions.py` during this refactor | The name/docstring can be cleaned separately. Do not mix naming churn into behavior migration. |
| Rename `real_evaluation/` or `offline_stage/` here | Naming-only. Keep it separate and easy to revert. |
| Plot structural rewrites | Plots consume artifacts. Label updates land with policy renames, but no new figure scaffolding belongs in this refactor. |
| Percentile-only main experiments | Raw samples exist for the important real-world pools; use them first. |

---

## 11. Commit Sequence

| # | Type | Subject |
|---|---|---|
| 1 | docs | replace 4-stage refactor plan with flat policy contract |
| 2 | test | lock current golden outputs for migration comparison |
| 3a | refactor | add flat policy code and Run/PerRequestRecord with no production callers |
| 3b | refactor | switch callers to `--policy` presets and delete `rwsim/strategies/` |
| 4 | test | assert no `rwsim.strategies`, `STRATEGY_REGISTRY`, or `--strategy` surface remains |
| 5 | refactor | wire simulator engine to emit `PerRequestRecord` rows; replace `SimulationRun` with `Run` (§4.4) |
| 6 | refactor | wire `experiments/real_evaluation/recorder.py` to emit `PerRequestRecord` rows; map legacy CSV columns to core fields, push retry/rate_limit/http_status to `metadata["real_*"]` |
| 7 | refactor | switch `plots/`, golden capture, and suite metadata to consume `Run` (delete legacy `mean_cost_usd` array fields, etc.) |
| 8 | feat | add EmpiricalDistribution and real-world profile pools |
| 9 | docs | update architecture and algorithm docs |

Commit 3a may add new policy files while old strategies still exist, but no
production caller may use the new path yet. Commit 3b is the switch: callers
move to policy presets and the strategy implementation layer is deleted. There
is no committed state where both engines are selectable at runtime.

Commits 5–7 land the §4.4 metrics schema in three steps: simulator emits the
new schema first (5), real-evaluation catches up next (6), downstream
consumers (plots, goldens, suite metadata) flip last (7). Between 5 and 6 the
real-eval CSV is still the old shape; that is acceptable because no
production code path joins sim and real records during this window. Commit 7
is the byte-equivalent gate: golden compare must pass on the new schema
before merging.

Each behavior-preserving commit must pass:

```bash
uv run pytest -m "not slow"
uv run python tests/golden_capture.py --mode compare
```

If an existing unrelated test fails, document it in the commit notes instead of
mixing a fix into the refactor commit.

---

## 12. Open Items

| Item | Owner | Blocks |
|---|---|---|
| Exact `RoutingDecision` and `RoutingOutcome` fields | Murphy | Phase 0 |
| `Run` and `PerRequestRecord` field list (§4.4) sign-off | Murphy + Juncheng | Phase 0 |
| `Policy.tick()` checkpoint schedule semantics | Resolved: policy-declared per request through `RoutingDecision.hedge_checkpoints` | Phase 0 |
| `HedgeDispatch` shape | Resolved: backup provider plus metadata; one hedge per request | Phase 0 |
| `user_id` field on `Request` — already present in trace data, or a synthetic per-trace assignment? Affects prefix-cache accounting reproducibility | Murphy | Phase 0 |
| Final paper-facing policy names | Resolved for simulator: `greedy_cost`, `greedy_latency`, `random`, `ablation_lp_only`, `ablation_lp_hedging`, `routewise` | Phase 1 |
| Preset file ownership | Resolved: shared defaults live in `rwsim.policies`; `experiments/simulation/presets.yaml` mirrors experiment-facing config | Phase 1 |
| CLI migration scope | Resolved: CLI uses `--policy` | Phase 1 |
| Golden parity tolerance for floating metrics | Murphy | Phase 1 |
| Provider name canonicalization for empirical samples | Defer | Phase 3 |
| Fate of `offline_stage`: merge fully or keep as `cost_oracle` harness | Murphy + Juncheng | Separate future refactor |

---

## 13. Sign-Off

If we agree with these three points, this document is the source of truth:

- `Policy` is flat: `route()` plus `tick()` plus `observe()`.
- Simulator owns execution/world state; policy owns learning state.
- Policy presets replace strategies; no runtime strategy compatibility layer.

Disagreements should name the section and propose the smallest alternative
that preserves the single-simulator invariant.
