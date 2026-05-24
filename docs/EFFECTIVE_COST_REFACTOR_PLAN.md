# Effective-Cost Shared Module Refactor

> Decision document. Defines the target structure for a single source of
> truth on effective-cost / shadow-price formulas, and the migration order
> to get there without changing experiment results.

Last updated: 2026-05-08.

---

## 1. TL;DR

The same scarcity-price math is implemented in three places today:

- `rwsim/policies/routewise.py` — production simulator policy
- `experiments/real_evaluation/shadow_price.py` — real-eval mirror
- `experiments/ablations/effective_cost/curves.py` — ablation curve sweep

Switching the concurrency formula from `U*u` to `constant_l` required edits
in all three. This is fragile and asymmetric: the ablation already exposes a
clean pure function `scarcity_price(curve, x, L=L, U=U)`, but the simulator
and real-eval re-implement the math inline.

The refactor moves `scarcity_price` to a shared module and rewrites the
simulator/real-eval shadow-price wrappers as thin adapters that extract the
scarcity signal (`z` for quota, `u` for concurrency) and delegate the math
to the shared module.

The refactor is **behavior-preserving**: every wrapper produces the same
numbers as before for the formulas currently in use (`exp_lu` for quota,
`constant_l` for concurrency).

---

## 2. Motivation

The current state has two concrete problems:

1. **Synchronization debt.** When we changed concurrency from
   `util_linear_u` to `constant_l` in commit `1b8b03b`, four files needed
   coordinated edits, plus tests and docs. The three production formula
   sites don't share code, so a future change still requires the same
   coordination — and divergence is silent (no compile-time signal that
   they should match).

2. **Documentation drift.** The simulator and real-eval files each carry a
   long docstring saying "this mirrors the formulas in the other module —
   keep them in lock-step." That comment is the only enforcement we have
   today. After §1.3 lands, end-to-end and hedging will likely surface
   another formula iteration; this is the right moment to remove the
   manual lock-step.

Refactoring now is cheaper than doing it after end-to-end results land,
because we have no in-flight experiment changes touching these files.

---

## 3. Current State

### 3.1 The pure math (already correct shape)

`experiments/ablations/effective_cost/curves.py` defines a single dispatch:

```python
def scarcity_price(curve: ScarcityCurve, x: float, *, L: float, U: float) -> float:
    if curve == "exp_lu":          return L * (U / L) ** x   # current quota
    if curve == "linear_lu":       return L + x * (U - L)
    if curve == "util_linear_u": return U * x              # former concurrency
    if curve == "constant_l":      return L                  # current concurrency
    if curve == "constant_u":      return U
```

This function is intentionally pure: no `Provider` import and no tier check.
It owns curve-local clamping and validation, while callers only extract the
scarcity signal from their own state objects. It is the right shape to become
the shared kernel.

### 3.2 The two adapter layers

Both `rwsim/policies/routewise.py` and
`experiments/real_evaluation/shadow_price.py` define identical-shaped
adapter pairs:

```python
def quota_shadow_price(provider_or_state, now, *, U, L) -> float:
    # 1. tier check
    # 2. extract z = quota.fraction_used(now), clamp to [0, 0.9999]
    # 3. validate 0 < L < U
    # 4. return L * (U / L) ** z          ← inline math, duplicated

def concurrency_shadow_price(provider_or_state, now, *, U, L, alpha=1.0) -> float:
    # 1. tier check
    # 2. validate L > 0
    # 3. return L                          ← inline math, duplicated
```

The only thing that genuinely differs between the two is **how to extract
the scarcity scalar from the input object**:

- simulator: `provider.quota.fraction_used(now)` on `rwsim.world.providers.Provider`
- real-eval: `state.quota.fraction_used(now)` on `experiments.real_evaluation.inventory.ProviderState`

The math is the same; the inputs are different types from different layers.

### 3.3 The ablation policy

`experiments/ablations/effective_cost/policy.py` already does it the right
way: it imports `scarcity_price` and feeds it the extracted `z` or `u`. No
duplication. This is the model we want to replicate everywhere.

---

## 4. Target Architecture

```text
rwsim/core/cost.py   ← shared pure math and piecewise scalar API
    ScarcityCurve
    EffectiveCostTier
    SCARCITY_CURVES
    scarcity_price(curve, x, *, L, U)
    effective_cost(tier, *, request_cost_usd, quota_fraction_used,
                   concurrency_utilization, L, U, ...)

rwsim/policies/routewise.py
    quota_shadow_price(provider, now, *, U, L) -> float:
        if provider.tier != ProviderTier.S_Q: return 0.0
        if provider.quota is None:            return 0.0
        z = clamp(provider.quota.fraction_used(now), 0, 0.9999)
        return scarcity_price("exp_lu", z, L=L, U=U)

    concurrency_shadow_price(provider, now, *, U, L, alpha=1.0) -> float:
        if provider.tier != ProviderTier.S_C: return 0.0
        if provider.concurrency is None:      return 0.0
        return effective_cost("concurrency", concurrency_utilization=None, L=L, U=U)

experiments/real_evaluation/shadow_price.py
    # Same shape, but extracts from ProviderState instead of Provider.
    # Calls the same scarcity_price() kernel.

experiments/ablations/effective_cost/{policy,presets,harness}.py
    # Import ScarcityCurve / scarcity_price directly from the kernel.
```

### 4.1 Naming

Use `rwsim/core/cost.py` for the public core API. The module is allowed to
export a pure scalar `effective_cost(...)` because it lives outside
`rwsim/policies/routewise.py`, where the simulator adapter keeps its
provider-shaped wrapper of the same name.

### 4.2 Why the kernel lives under `rwsim/core/`

It is core simulator math. Real-eval and ablation depend on `rwsim`; the
reverse is not true. Keeping the shared scalar API in `rwsim.core` also matches
the LP and hedging extraction boundary:

```text
rwsim.core.cost   (no provider/harness deps)
    ├── rwsim.policies.routewise
    ├── experiments.real_evaluation.shadow_price
    └── experiments.ablations.effective_cost.{policy,presets,harness}
```

---

## 5. Migration Plan

Five small commits, each independently testable. Steps 1–3 are
behavior-preserving; step 4 adds a guardrail; step 5 removes the old
ablation-local formula module.

### Step 1 — Introduce the kernel module (new file only)

Create `rwsim/core/cost.py` containing the scarcity curves plus a pure scalar
`effective_cost(...)` that accepts normalized tier names and read-only scalar
snapshots.

Leave the ablation imports unchanged until step 5 so steps 2 and 3 can land
independently.

**Verification:**
- `uv run pytest -q tests/unit/ablations/test_effective_cost_curves.py`
- `uv run pytest -q tests/unit/ablations/test_effective_cost_policy.py`

### Step 2 — Rewire `rwsim/policies/routewise.py`

Replace the inline math in `quota_shadow_price` and
`concurrency_shadow_price` with calls to `scarcity_price`. Behavior must
not change: `exp_lu` for quota, `constant_l` for concurrency. The wrappers
handle tier/null checks and signal extraction; the kernel owns curve-local
validation and clamping.

**Verification:**
- `uv run pytest -q tests/unit/policies/test_flat_policies.py`
- `uv run pytest -q tests/unit/simulation/test_cost_layer.py`
- Spot-check: re-run a §1.2 q=16 scenario for one seed; total cost must
  match the previous summary CSV bit-for-bit.

### Step 3 — Rewire `experiments/real_evaluation/shadow_price.py`

Same change, against `ProviderState`. Drop the long "lock-step" docstring
preamble; replace with a one-liner pointing to the kernel.

**Verification:**
- `uv run pytest -q tests/unit/real_evaluation/test_policies.py`
- `uv run pytest -q tests/unit/real_evaluation/`

### Step 4 — Add cross-implementation consistency test

New file `tests/unit/policies/test_effective_cost_kernel_consistency.py`.
For a fixed `(L, U)` and a sweep of `z, u ∈ {0.0, 0.25, 0.5, 0.75, 1.0}`,
build a minimal simulator `Provider` and a minimal real-eval
`ProviderState` with matching scarcity signals, then assert:

```python
sim_value      = rwsim_quota_shadow_price(provider, now, L=L, U=U)
real_value     = real_eval_quota_shadow_price(state, now, L=L, U=U)
kernel_value   = scarcity_price("exp_lu", z, L=L, U=U)
assert sim_value == real_value == kernel_value
```

This is the safety net for any future formula change. After this commit,
divergence between the three sites becomes a CI failure rather than a
silent bug.

### Step 5 — Delete the ablation-local `curves.py`

Remove `experiments/ablations/effective_cost/curves.py` and update all callers
to import directly from `rwsim.core.cost`. This avoids keeping two formula entry
points after the migration.

---

## 6. Testing Strategy

Three layers:

1. **Existing unit tests stay green.** No test file should need to change
   in steps 1–3 except trivial import path updates if any test reaches
   into `curves.py` for module-private symbols (none currently do).

2. **New cross-consistency test** (step 4) locks the three sites to the
   same kernel.

3. **End-to-end smoke check** before declaring the refactor done:
   re-run one cost-layer scenario (§1.1 normal, single seed, no quota) and
   one §1.2 scenario (q=16, single seed) and `diff` the resulting
   `summary.csv` against the pre-refactor versions. Both must be
   identical.

The §1.3 rerun with multi-seed `constant_l` is **out of scope** for this
refactor and should land in a separate commit chain after the refactor
merges.

---

## 7. Backward Compatibility

- All public function names and signatures stay the same in
  `rwsim/policies/routewise.py` and
  `experiments/real_evaluation/shadow_price.py`.
- Existing in-repo imports of `scarcity_price` move directly to
  `rwsim.core.cost`.
- The `util_linear_u` curve stays available in the kernel — it is still
  the named ablation comparison curve in `presets.py`.

External callers that import the old ablation-local `curves.py` must update
to `rwsim.core.cost`.

---

## 8. Risks and Rollback

**Risk 1: silent numeric divergence.** The pure-function kernel uses
`math.pow(U/L, x)`; the legacy inline simulator code did the same. Confirm
identical floating-point output before and after step 2 by running:

```bash
uv run pytest -q tests/unit/policies/test_flat_policies.py \
                tests/unit/simulation/test_cost_layer.py \
                tests/unit/real_evaluation/test_policies.py \
                tests/unit/ablations/
```

If any test changes by even 1 ULP, treat it as a bug, not a tolerance
issue.

**Risk 2: import cycle.** The kernel must not import anything from
`rwsim.world`, `rwsim.engine`, or `experiments`. Enforce by keeping
imports in the new file limited to `math` and `typing`.

**Rollback:** each step is a single commit. If step 2 or 3 introduces a
regression, revert that commit; if step 5 breaks an external script, restore
`curves.py` temporarily as a re-export.

---

## 9. Out of Scope

- Changing any formula. The current production formulas (`exp_lu` for
  quota, `constant_l` for concurrency) are locked.
- Touching `rwsim/policies/routewise.py`'s `effective_cost(...)` function
  beyond the two shadow-price helpers it calls.
- Reorganizing `experiments/ablations/effective_cost/` beyond deleting the
  old `curves.py` entry point. The harness, presets, and policy stay where
  they are.
- §1.3 multi-seed rerun, §2.2 hedging, §3 end-to-end. These are
  experiment-scope tasks, not refactor-scope.

---

## 10. Acceptance

The refactor is done when:

- A single file under `rwsim/core/` defines `scarcity_price` and the scalar
  `effective_cost(...)` API.
- `rwsim/policies/routewise.py` and
  `experiments/real_evaluation/shadow_price.py` no longer contain the
  inline `L * (U/L)**z` or `return L` math.
- `tests/unit/policies/test_effective_cost_kernel_consistency.py` exists
  and passes.
- All previously green tests are still green.
- Re-running §1.1 normal and §1.2 q=16 yields byte-identical summary CSVs
  to the pre-refactor versions.
