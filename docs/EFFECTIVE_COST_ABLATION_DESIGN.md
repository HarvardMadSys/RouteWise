# Effective Cost Ablation Design

> Design doc for the RouteWise simulator ablation that tests the shadow-price
> formula behind effective cost. This document assumes cost-layer 1.1, 1.2,
> and 1.3 exist as plan-backed simulator sections, and defines the next
> experiment harness without changing the main cost-layer figure path.

Last updated: 2026-05-07.

---

## 1. TL;DR

We need this ablation to answer two separate questions:

1. **Quota curve choice.** For consumable subscription quota, should the
   shadow price be the paper's exponential curve or a linear curve?
2. **Unified scarcity formula.** Can quota and concurrency use one common
   scarcity-price function, or does reusable concurrency need a separate
   formula?

The implementation should live in a centralized ablation package that reuses
the existing simulator orchestration without changing the production
`RouteWisePolicy` surface. Use **Method A**: the ablation owns a small
LP-only policy that implements the cost-router decision needed for formula
sweeps.

```text
experiments/
  simulation/
    cost_layer.py                  # main paper cost-layer path, unchanged

  ablations/
    effective_cost/
      curves.py                    # ablation candidate formulas
      policy.py                    # LPOnlyAblationPolicy, no hedging/explorer
      presets.py                   # curve/p sweep -> section-local presets
      harness.py                   # scenario/policy listing, CLI, run_cell
      oracle.py                    # deferred Stage Q / Stage QC adapter
      README.md
```

Do not reintroduce additive effective cost. The tier semantics remain
piecewise:

```text
S_A: c_eff = real API marginal cost
S_Q: c_eff = quota scarcity price
S_C: c_eff = concurrency scarcity price
```

The first implementation target is **Phase A quota-only** using the existing
1.2 configuration:

```text
q* = 16
latency_family = heavy_tail
workload = burstgpt
seed = 42
```

The first run should compare formulas and inspect routing behavior. The Stage
Q / Stage QC oracle remains important for final regret numbers, but it should
not block the first formula sweep.

---

## 2. Research Questions

### Q1. What is the right quota scarcity curve?

Quota is a consumable resource. If the online router spends quota early on
low-value requests, it can lose future savings. The paper formula currently
uses an exponential shadow price:

```text
psi_exp(z; L, U) = L * (U / L)^z
```

where:

- `z` is the fraction of quota consumed in the active quota window.
- `L` is the lower API-equivalent request-cost envelope.
- `U` is the upper API-equivalent request-cost envelope.

The ablation should compare this against at least one linear alternative:

```text
psi_linear_lu(z; L, U) = L + z * (U - L)
```

The decision criterion is not which curve is prettier. For the first sweep,
the criterion is which curve produces the most reasonable cost, latency, and
quota-allocation behavior under the fixed 1.2 configuration. For the final
paper result, this should be backed by lower regret against the quota oracle
while keeping the same latency distribution and the same purchased
subscription plan.

### Q2. Can quota and concurrency share one formula?

Quota and concurrency are not the same problem class:

- Quota is consumable across a window.
- Concurrency is reusable after each request completes.

So a unified formula is a hypothesis, not a prior. The ablation needs a
joint test because isolated quota-only and concurrency-only runs cannot
answer whether their rankings remain stable when both resources are present.

The joint question is:

```text
scarcity_price(curve, x, L, U)
```

with:

```text
S_Q: x = z, quota fraction used
S_C: x = u, weighted concurrency utilization
```

If the same `curve` works for both tiers in joint runs, we can claim a
unified effective-cost rule. If not, the paper should report separate
quota and concurrency curves.

---

## 3. Current Readiness

### Ready

The current simulator already has the core pieces needed to build the
ablation:

- Cost-layer 1.1 has API-only scenarios with workload-level cost envelopes.
- Cost-layer 1.2 has plan-backed quota scenarios, including Chutes and
  MiniMax-style multi-window quota.
- Cost-layer 1.3 has plan-backed Featherless weighted concurrency design, but
  the final concurrency configuration/results are still in progress.
- `RouteWisePolicy` now uses explicit workload-level `(L, U)`, not
  per-request calibration.
- `effective_cost()` is piecewise by provider tier and follows the paper
  structure.
- Summary rows already cover the main cost/latency outputs needed for the
  first quota-only sweep.

### Not Ready For Final Ablation

Three gaps matter before headline ablation results:

1. **Oracle gap.** `experiments/simulation/cost_layer.py` has a local
   `offline` runner that performs greedy quota selection and first-fit
   concurrency packing. That is acceptable for smoke tests, but it is not
   the Stage Q / Stage QC lower bound.
2. **Concurrency gap.** The 1.3 concurrency configuration has not fully
   settled, so Phase B and Phase C should wait until the selected concurrency
   setup is reproducible.
3. **Preset gap.** Global `rwsim.policies.DEFAULT_PRESETS` still expose
   RouteWise presets without a `cost_envelope`. Section-local simulator
   harnesses inject the envelope, but generic `build_policy("routewise")`
   currently fails. This is not a blocker for the ablation harness if the
   harness uses section-local presets, but it should be fixed before relying
   on the generic runner.

---

## 4. Experimental Design

### Phase 0. Method A Harness Sanity

Before formula comparisons, build the smallest Method A harness that can run a
single quota-only scenario with a curve-specific LP-only policy.

Use:

- `experiments/ablations/effective_cost/policy.py`
- `experiments/ablations/effective_cost/presets.py`
- `experiments/ablations/effective_cost/harness.py`

The ablation policy should not subclass or modify `RouteWisePolicy`. It should
implement only:

1. compute per-provider `c_eff` using `scarcity_price()`;
2. compute `B_p = c_min + p * (c_max - c_min)`;
3. solve the same LP-only routing decision;
4. sample a primary provider with the provided seed.

Solver implementation is locked: copy the current hand-written LP enumerator
from `RouteWisePolicy` rather than using `scipy.optimize.linprog`. This is
acceptable duplication for the ablation because it preserves the production
cost-router tie-break, normalization, and sampling semantics. The goal is to
measure curve changes, not solver differences.

Sanity test the Method A policy against the production formula: when the
ablation curve is `exp_lu`, S_Q effective cost should match the current
RouteWise quota formula. This prevents the sweep from accidentally measuring a
different router rather than a different curve.

### Phase A. Quota-Only Curve Ablation

Goal: answer Q1.

Setup:

- Scenario family: cost-layer quota only, reusing the mature 1.2 builder.
- Primary plan: `chutes`.
- Optional sensitivity: `minimax_subscription_plus`.
- Headline count: `q* = 16`.
- Latency: `heavy_tail`.
- Dataset: `burstgpt` (the BurstGPT 30-day trace used by the §1.2 Chutes main
  run).
- Seed: `42`.
- Development smoke: same config with `--max-requests`.

Curves:

| Curve id | Formula | Purpose |
|---|---|---|
| `quota_exp_lu` | `L * (U / L)^z` | Paper formula |
| `quota_linear_lu` | `L + z * (U - L)` | Main alternative |
| `quota_constant_l` | `L` | Sanity baseline: treats quota as always cheap |
| `quota_constant_u` | `U` | Sanity baseline: treats quota as always expensive |

Primary metrics:

- `total_cost_usd_per_run`
- `api_cost_usd_per_run`
- `subscription_fixed_cost_usd_per_run`
- `tier_mix`
- `quota_fits_in_trace`
- quota/API routing split
- quota exhaustion time, if exhausted
- quota utilization over time
- API fallback concentration before/after quota scarcity
- mean / p95 / p99 latency

Derived analysis:

- average API-equivalent value of requests routed to S_Q
- average API-equivalent value of requests routed to S_A
- percentile rank of S_Q-routed requests in the request value distribution
- high-value requests forced to S_A after quota is depleted

Implementation decision: choose option (b). These derived analyses are a
post-run script or notebook over the normal simulator outputs and per-request
records, not new `summary.csv` fields in the first harness. Do not expand the
shared summary schema or touch `experiments/simulation/common.py` just to
support this first sweep. If records are insufficient for one derived view,
report that view qualitatively and keep the first paper-facing comparison on
the primary metrics above.

Success criterion:

The exponential curve is justified for the first quota-only result if it lowers
API fallback / total cost relative to linear and constant baselines while
showing a sensible quota trajectory: it should not win by leaving quota unused,
nor by burning quota early on low-value requests and pushing later high-value
requests to S_A. Tail latency must not degrade enough to erase the cost
argument.

### Phase B. Concurrency-Only Curve Ablation

Goal: understand whether the existing concurrency formula is enough before
testing a unified formula.

Status: wait until the 1.3 concurrency configuration is reproducible.

Setup:

- Scenario family: cost-layer concurrency only.
- Primary plan: `featherless_premium`.
- Counts: configured `subscription_counts`.
- Primary model: `sharegpt` mapped to `ge_70b`, because it fully occupies one
  Premium account's weighted capacity.
- Optional sensitivity: `qwen3-coder-30b` mapped to `24_34b`.
- Latency: hold equal with the existing cost-layer `heavy_tail` default.

Curves:

| Curve id | Formula | Purpose |
|---|---|---|
| `conc_legacy_linear_u` | `U * u` | Current implementation |
| `conc_linear_lu` | `L + u * (U - L)` | Same linear shape as quota |
| `conc_exp_lu` | `L * (U / L)^u` | Exponential unified candidate |
| `conc_constant_l` | `L` | Sanity baseline: overuses concurrency |

Primary metrics:

- `total_cost_usd_per_run`
- `api_cost_usd_per_run`
- `oracle_gap_pct`
- `tier_mix`
- `peak_used_concurrency_cost`
- `mean_concurrency_utilization`
- `concurrency_saturated_in_trace`
- selected concurrency count under each curve

Success criterion:

The selected curve must reduce API fallback cost without pushing the router
into obviously saturated concurrency behavior. Because concurrency is
reusable, Phase B should be interpreted as evidence, not the final answer to
Q2.

### Phase C. Joint Quota + Concurrency Ablation

Goal: answer Q2.

Status: wait until Phase A and Phase B both have stable configurations.

This phase is mandatory. A unified formula cannot be validated in isolated
quota-only or concurrency-only runs because the real question is whether the
router ranks S_Q, S_C, and S_A correctly when all are feasible.

Setup:

- One S_Q provider from the Phase A candidate set.
- One S_C provider from the Phase B candidate set.
- The same fixed cheap/mid/expensive S_A fallback ladder.
- Hold latency equal across all tiers.
- Sweep the selected quota counts and concurrency counts around the
  independently best settings from Phase A and Phase B.

Candidate policies:

| Policy id | Quota curve | Concurrency curve | Question |
|---|---|---|---|
| `separate_best` | best Phase A curve | best Phase B curve | Upper bar for online formulas |
| `unified_exp_lu` | exponential | exponential | Can one exponential curve work? |
| `unified_linear_lu` | linear LU | linear LU | Can one linear curve work? |
| `current_paper` | exponential | legacy linear U | Current RouteWise behavior |

Primary metrics:

- `oracle_gap_pct` against Stage QC.
- `total_cost_usd_per_run`.
- tier mix: S_Q vs S_C vs S_A.
- selected `(quota_count, concurrency_count)`.
- utilization diagnostics for both scarce resources.

Success criterion:

The unified formula is acceptable only if its joint oracle gap is close to
`separate_best` and it preserves the same selected capacity region. If it
changes the selected plan/count in a way that increases regret, we should not
claim a unified formula.

---

## 5. Code Design

### 5.1 Experiment-scoped curve helpers

Add a small module:

```text
experiments/ablations/effective_cost/curves.py
```

Suggested surface:

```python
ScarcityCurve = Literal[
    "exp_lu",
    "linear_lu",
    "legacy_linear_u",
    "constant_l",
    "constant_u",
]

def scarcity_price(curve: ScarcityCurve, x: float, *, L: float, U: float) -> float:
    ...
```

This module should be deterministic and easy to unit test. It must not import
provider, policy, or simulator engine types. Candidate curves stay in the
ablation package until the experiment justifies changing the stable RouteWise
formula.

### 5.2 Method A policy boundary

Do not add every candidate curve to `rwsim.policies` as a core policy surface.
The stable `rwsim` implementation should keep the paper-current formula:

```text
S_Q: exp_lu
S_C: legacy_linear_u
```

Do not add an `effective_cost_fn` field, subclass hook, or ablation-specific
branch to `RouteWisePolicy`. The ablation policy is a separate cost-layer-only
tool and should stay inside `experiments/ablations/effective_cost/`.

Add:

```text
experiments/ablations/effective_cost/policy.py
```

Suggested surface:

```python
@dataclass
class LPOnlyAblationPolicy:
    quota_curve: ScarcityCurve
    concurrency_curve: ScarcityCurve
    p: float
    cost_envelope: tuple[float, float]
    seed: int = 0
    profile_window_sec: float = 15 * 60

    def route(self, request, state):
        ...
```

This policy intentionally omits hedging and explorer feedback. It should keep
the same rolling latency-profile objective as production LP-only RouteWise so
the `p` sweep remains meaningful under equal configured provider latency
distributions. It should duplicate only the small LP-only cost-router and
profile logic needed for a clean formula ablation.

Policy construction is ablation-local. `presets.py` may emit curve/p metadata,
but `harness.py` should instantiate `LPOnlyAblationPolicy` through a small
local builder after the workload cost envelope is materialized. Do not register
`LPOnlyAblationPolicy` in `rwsim.policies.DEFAULT_PRESETS`, and do not route it
through the generic `build_policy()` path.

Do not add ablation-specific branching to `cost_layer.py`.

### 5.3 Ablation harness

Add two small modules plus the harness:

```text
experiments/ablations/effective_cost/presets.py
experiments/ablations/effective_cost/harness.py
```

Responsibilities:

- Build Phase A quota-only scenarios from existing plan-backed builders.
- Prefer public `cost_layer.make_scenarios()` / `make_scenario()` APIs. If the
  private quota builder is unavoidable, wrap it once in the harness rather than
  spreading private imports.
- Build curve-specific LP-only ablation presets.
- Call the shared `run_section()` helper.
- Write `metadata.json`, `summary.csv`, `summary.json`, and histograms using
  the same shape as other simulator sections.
- Leave Stage Q / Stage QC oracle attachment for the later `oracle.py` step.
- The implementation may add a thin `ablation` subcommand group to
  `routewise_cli/main.py` that delegates to this harness. This is a user-facing
  CLI entry point, not a change to the production `rwsim` policy path.

Proposed CLI:

```bash
routewise ablation effective-cost \
  --phase quota \
  --curve exp_lu \
  --curve linear_lu \
  --qstar 16 \
  --latency-family heavy_tail \
  --workload burstgpt \
  --p 0.5 \
  --seed 42 \
  --max-requests 1000

routewise ablation effective-cost \
  --phase joint \
  --quota-plan chutes \
  --concurrency-plan featherless_premium \
  --model sharegpt
```

### 5.4 Tests

Add unit tests before running full traces:

```text
tests/unit/ablations/test_effective_cost_curves.py
tests/unit/ablations/test_effective_cost_policy.py
tests/unit/ablations/test_effective_cost_harness.py
```

Minimum coverage:

- `scarcity_price()` returns expected values at `x = 0`, `x = 0.5`, and
  near exhaustion.
- Current-curve ablation S_Q effective cost matches production
  `quota_shadow_price()`.
- `p` changes the LP budget and not the workload cost envelope.
- Curve-specific presets pass explicit workload cost envelopes.
- Phase A scenario construction uses only S_Q + S_A for the first harness.
- The headline Phase A config is fixed to `q*=16`, `heavy_tail`,
  `burstgpt`, `seed=42`.
- No test requires modifying `rwsim/policies/routewise.py` or
  `experiments/simulation/common.py`.

---

## 6. Output Layout

Use a separate output directory:

```text
outputs/ablations/effective_cost/
  quota/
  concurrency/
  joint/
```

Each phase should emit:

```text
metadata.json
summary.csv
summary.json
ttft_histograms.json
ttft_histograms_by_seed.json
```

Plot code should live under:

```text
plots/ablations/effective_cost/
```

Do not mix these figures into `plots/cost_layer/simulator/` until the
ablation result is stable and we know which paper figure or appendix figure
it feeds.

---

## 7. Yangsun Branch Handling

Yangsun's c1-c4 concurrency comparison can be kept as Phase B context, but
it should not be the implementation source of truth.

Keep:

- scenario intuition
- any generated results that expose the "linear looks good at c4 because of
  implicit load balancing" observation
- credit for the first-pass concurrency comparison

Retire:

- formula code that bypasses the paper's piecewise effective-cost structure
- additive effective cost
- section-local one-off curve logic embedded in `cost_layer.py`

If useful, recreate his c1-c4 sweep inside the new ablation harness as a
legacy sensitivity run.

---

## 8. Out Of Scope

This ablation should not include:

- hedging
- explorer feedback
- live OpenRouter calls
- service-time-aware concurrency pricing
- queueing policies for S_C
- real invoice reconciliation

Service-time-aware concurrency is a good later idea, but it requires
predicted duration and therefore touches the latency/value-estimation axis.
That would confound the current effective-cost formula ablation.

---

## 9. Open Questions

1. Should the oracle objective be cost-only, or value-aware with predicted
   request savings? Default for this ablation should be cost-only unless the
   paper narrative explicitly moves to value-aware routing.
2. Which Phase A quota plans are in the headline grid: Chutes only, or
   Chutes plus MiniMax Plus?
3. Should Phase B include both `sharegpt` and `qwen3-coder-30b` in the main
   table, or keep the second model as sensitivity?
4. What tolerance counts as "close to separate_best" for accepting a unified
   formula: absolute cost delta, relative oracle gap, or selected count
   stability?
5. Should global `DEFAULT_PRESETS` receive a safe default `cost_envelope`, or
   should generic runners be forced to pass one explicitly?
6. Should Phase A include a square/root-style curve mentioned in the May 5
   discussion, or keep the first paper-facing grid to exp/linear/constants?

---

## 10. Suggested Implementation Order

1. Keep `experiments/ablations/effective_cost/curves.py` plus unit tests as the
   formula surface.
2. Add `policy.py` with the self-contained Method A LP-only ablation policy.
3. Add `presets.py` for curve/p sweep preset generation.
4. Add `harness.py` with Phase A scenario/preset generation for `q*=16`,
   `heavy_tail`, `burstgpt`, `seed=42`.
5. Run Phase A smoke with `--max-requests`, then the full Phase A trace.
6. Analyze cost, quota trajectory, request-value allocation, and latency tails.
7. Add `oracle.py` with the Stage Q adapter once the formula sweep is
   reproducible.
8. After 1.3 settles, add Phase B scenario/preset generation.
9. Add the concurrency-only oracle adapter.
10. Run Phase B smoke and full trace.
11. Add Phase C joint scenario generation and Stage QC oracle adapter.
12. Run Phase C smoke.
13. Only then run the full joint grid and produce paper/appx plots.
