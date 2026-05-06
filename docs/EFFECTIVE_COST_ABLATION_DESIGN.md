# Effective Cost Ablation Design

> Design doc for the RouteWise simulator ablation that tests the shadow-price
> formula behind effective cost. This document assumes cost-layer 1.1, 1.2,
> and 1.3 exist as plan-backed simulator sections, and defines the next
> experiment harness without changing the main cost-layer figure path.

Last updated: 2026-05-06.

---

## 1. TL;DR

We need this ablation to answer two separate questions:

1. **Quota curve choice.** For consumable subscription quota, should the
   shadow price be the paper's exponential curve or a linear curve?
2. **Unified scarcity formula.** Can quota and concurrency use one common
   scarcity-price function, or does reusable concurrency need a separate
   formula?

The implementation should live in a centralized ablation package that reuses
the existing simulation and oracle systems without turning every ablation into
a top-level experiment subsystem:

```text
experiments/
  simulation/
    cost_layer.py                  # main paper cost-layer path, unchanged

  ablations/
    effective_cost/
      curves.py                    # ablation candidate formulas
      experiment.py                # Phase A/B/C harness
      oracle_adapter.py            # Stage Q / Stage QC adapter
      README.md
```

Do not reintroduce additive effective cost. The tier semantics remain
piecewise:

```text
S_A: c_eff = real API marginal cost
S_Q: c_eff = quota scarcity price
S_C: c_eff = concurrency scarcity price
```

The current 1.1 / 1.2 / 1.3 simulator code is ready enough to start this
ablation implementation. The main missing piece is the oracle baseline: the
cost-layer section-local `offline` runner is a smoke baseline, not the
ablation oracle. The ablation should use or adapt the existing
`experiments/offline_stage/` Stage Q and Stage QC logic.

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

The decision criterion is not which curve is prettier. The criterion is
which curve yields lower cost regret against the quota oracle while keeping
the same latency distribution and the same purchased subscription plan.

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
- Cost-layer 1.3 has plan-backed Featherless weighted concurrency.
- `RouteWisePolicy` now uses explicit workload-level `(L, U)`, not
  per-request calibration.
- `effective_cost()` is piecewise by provider tier and follows the paper
  structure.
- Summary rows include fixed subscription fees, quota-fit flags, and
  weighted concurrency metrics.

### Not Ready For Final Ablation

Two gaps matter before headline ablation results:

1. **Oracle gap.** `experiments/simulation/cost_layer.py` has a local
   `offline` runner that performs greedy quota selection and first-fit
   concurrency packing. That is acceptable for smoke tests, but it is not
   the Stage Q / Stage QC lower bound.
2. **Preset gap.** Global `rwsim.policies.DEFAULT_PRESETS` still expose
   RouteWise presets without a `cost_envelope`. Section-local simulator
   harnesses inject the envelope, but generic `build_policy("routewise")`
   currently fails. This is not a blocker for the ablation harness if the
   harness uses section-local presets, but it should be fixed before relying
   on the generic runner.

---

## 4. Experimental Design

### Phase 0. Oracle Adapter Sanity

Before formula comparisons, wire the ablation to a trustworthy lower bound.

Use:

- **Stage Q** for quota-only scenarios.
- **Stage QC** for quota + concurrency scenarios.
- Stage QC with quota disabled, or a small adapter around the same MILP, for
  concurrency-only scenarios.

The adapter should produce the same run-level summary shape as simulator
policies:

```text
scenario
policy
seed
n_requests
api_cost_usd
subscription_fixed_cost_usd
total_cost_usd
provider_mix
tier_mix
oracle_gap_usd
oracle_gap_pct
```

Do not compare formula curves until this adapter has a small deterministic
test case showing that the oracle uses high-cost API-equivalent requests for
scarce quota/concurrency before low-cost requests.

### Phase A. Quota-Only Curve Ablation

Goal: answer Q1.

Setup:

- Scenario family: cost-layer quota only.
- Primary plan: `chutes`.
- Optional sensitivity: `minimax_subscription_plus`.
- Counts: use each plan's configured `subscription_counts`.
- Latency: hold equal with the existing cost-layer `heavy_tail` default.
- Dataset: ShareGPT one-month trace for paper results; `--max-requests`
  smoke runs for development.

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
- `oracle_gap_pct`
- `tier_mix`
- `quota_fits_in_trace`
- selected subscription count under each curve

Success criterion:

The exponential curve is justified if it has lower oracle regret than the
linear curve across the binding quota regimes and does not simply win by
avoiding quota usage entirely.

### Phase B. Concurrency-Only Curve Ablation

Goal: understand whether the existing concurrency formula is enough before
testing a unified formula.

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

### 5.2 RouteWise integration boundary

Do not add every candidate curve to `rwsim.policies` as a core policy surface.
The stable `rwsim` implementation should keep the paper-current formula:

```text
S_Q: exp_lu
S_C: legacy_linear_u
```

If the ablation policy needs to reuse RouteWise's LP and sampling logic, add
only a minimal hook to `RouteWisePolicy` such as:

```python
def effective_cost_for_provider(...):
    return effective_cost(...)
```

Then the ablation package can subclass and override that hook. The hook keeps
default behavior unchanged and avoids copying the whole RouteWise route body.

Do not add ablation-specific branching to `cost_layer.py`.

### 5.3 Ablation harness

Add:

```text
experiments/ablations/effective_cost/experiment.py
```

Responsibilities:

- Build Phase A, B, and C scenarios from existing plan-backed builders.
- Build curve-specific RouteWise presets.
- Call the shared `run_section()` helper.
- Attach oracle rows through the Stage Q / Stage QC adapter.
- Write `metadata.json`, `summary.csv`, `summary.json`, and histograms using
  the same shape as other simulator sections.

Proposed CLI:

```bash
routewise ablation effective-cost \
  --phase quota \
  --curve exp_lu \
  --curve linear_lu \
  --p 0.5 \
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
tests/unit/ablations/test_effective_cost_ablation.py
```

Minimum coverage:

- `scarcity_price()` returns expected values at `x = 0`, `x = 0.5`, and
  near exhaustion.
- RouteWise defaults still match current behavior.
- Curve-specific presets pass explicit workload cost envelopes.
- Phase A scenario construction uses only S_Q + S_A.
- Phase B scenario construction uses only S_C + S_A.
- Phase C scenario construction uses S_Q + S_C + S_A.
- Oracle adapter is not the local cost-layer greedy `offline` runner.

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
- latency-profile learning
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

---

## 10. Suggested Implementation Order

1. Add `experiments/ablations/effective_cost/curves.py` plus unit tests.
2. Add only the minimal RouteWise hook needed by the ablation policy, if needed.
3. Fix or explicitly guard the global preset `cost_envelope` issue.
4. Add the ablation harness skeleton and Phase A scenario/preset generation.
5. Add the Stage Q oracle adapter.
6. Run Phase A smoke and full trace.
7. Add Phase B scenario/preset generation.
8. Add the concurrency-only oracle adapter.
9. Run Phase B smoke and full trace.
10. Add Phase C joint scenario generation and Stage QC oracle adapter.
11. Run Phase C smoke.
12. Only then run the full joint grid and produce paper/appx plots.
