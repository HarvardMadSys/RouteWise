# LP-Budget Synthetic Simulator Plan

**Status**: Proposed implementation plan  
**Date**: 2026-04-23  
**Scope**: Synthetic simulator validation only  
**Out of scope**: Production evaluation, paper writing, new safety/UCB variants

## 1. Goal

This document specifies how we will validate Juncheng's new latency-layer LP idea in the merged synthetic simulator.

The goal is not to redesign the router. The goal is to verify whether the following separation works in simulation:

- The LP handles the **body** of the latency distribution.
- Hedging handles the **tail**.
- A per-request budget `tau * v_hat_i` provides a clean and tunable cost-latency tradeoff.

## 2. Core Hypothesis

The old LP formulation is:

```text
minimize expected effective cost
subject to expected SLO-attainment >= target
```

The new LP formulation is:

```text
minimize expected body latency
subject to expected effective cost <= budget
```

The budget is controlled by a per-request Pareto knob:

```text
B_i = tau * v_hat_i
tau in {0.25, 0.50, 0.75}
```

Where:

- `v_hat_i` is the request-side estimated API price anchor for request `i`
- in the synthetic simulator, `v_hat_i` is implemented as the per-request billed cost of the cheapest `S_A` provider for that request
- `tau` is an operator-facing willingness-to-pay knob

The intended story is:

- LP should no longer chase tail-CDF noise.
- LP should optimize the body using stable latency signals.
- Hedging should remain the mechanism that protects P99 and SLO violations.

## 3. What We Will and Will Not Change

### 3.1 Fixed

The following parts must remain unchanged in the main experiment:

- The merged simulator world model
- Provider definitions and scenario construction
- Shadow-price definitions
- Warm-up logic
- Dedicated probing cadence
- Backup-provider selection logic
- Existing workload generation

### 3.2 Changed

We change two things in the mainline synthetic experiment:

- the **body selector / LP objective**
- the **hedge trigger rule**
- the **hedge feedback path** (Explorer / hedge-as-probe)

### 3.3 Explicitly not included

These are not part of the main plan:

- UCB safety variants
- New safety-prefilter variants
- Backup randomization / redundant-backup exploration policies
- Shadow-price formula changes
- Backup-provider selection redesign
- Production-port changes

If any of the above is explored later, it must be clearly labeled as a follow-up ablation, not part of the main claim.

### 3.4 Interpretation boundary: what each phase means

Phase A still isolates the LP change because hedging is disabled there.

Phase B and Phase C evaluate the **new full method**:

- request-driven budget LP
- probability-only hedge trigger
- explorer-style hedge feedback

Interpretation rule:

- Phase A is the evidence for whether the LP improves the body.
- Phase B/C are the evidence for whether the combined system story
  "LP handles body, hedging handles tail" is supported.

### 3.5 Implemented hedge redesign

The mainline sidecar simulator now replaces the old probability-versus-cost
hedge trigger with a probability-only SLO-feasibility rule.

At wait time `t`, let:

- `L` be the SLO budget
- `delta` be the dispatch overhead
- `F_p` be the primary TTFT CDF
- `F_b` be the backup TTFT CDF

Conditioned on the primary still being unfinished at time `t`, the combined
success probability after dispatching a backup is:

```text
P(not violate | t) + P(violate | t) * P(backup succeeds)
```

with:

```text
P(not violate | t) = (F_p(L) - F_p(t)) / (1 - F_p(t))
P(violate | t) = (1 - F_p(L)) / (1 - F_p(t))
P(backup succeeds) = F_b(L - t - delta)
```

The hedge target is:

```text
P(not violate | t) + P(violate | t) * P(backup succeeds) >= 0.99
```

Operational interpretation:

- We do **not** trigger at the earliest `t` satisfying the condition.
- We search for the **latest safe dispatch time** `t*` that still satisfies
  the condition.
- If the sampled primary TTFT exceeds `t*`, dispatch the backup at `t*`.
- If no safe wait time exists but immediate hedging improves success over
  no-hedge, dispatch immediately at `t = 0`.

This interpretation avoids the degenerate behavior where an earliest-trigger
search collapses into immediate replication.

### 3.6 Implemented Explorer semantics

The mainline hedged variants also enable **hedge-as-probe**:

- when a hedge fires and the backup is different from the primary,
  the backup TTFT sample is fed back into the rolling latency profile
- the existing 5% dedicated probing path is still kept

This follows the earlier `lp_explorer` / `v2_explorer` simulator semantics:

- Explorer is a complement to dedicated probing, not a replacement
- We do not disable background probing in the mainline setting
- We also support a pure-explorer ablation by setting `probe_rate = 0.0`

### 3.7 Implemented adaptive backup-selection mix

The new hedge path no longer fixes the backup provider to the deterministic
fastest cross-tier provider.

Instead, backup selection is split into two branches:

1. **SLO-safe cheapest branch**
   - among currently available non-primary providers, keep those whose backup
     TTFT CDF satisfies the target success requirement under the full SLO
     budget minus dispatch overhead
   - choose the cheapest such provider, breaking ties by lower P50

2. **Random explorer branch**
   - choose a random currently available non-primary provider

The random branch probability starts at 0.5 and shrinks when the recent final
SLO violation rate rises. This is the minimal operationalization of the
meeting idea:

- when violations are low, spend more hedges on exploration coverage
- when violations rise, bias backup selection back toward SLO-safe providers

Legacy `*_oldhedge` ablation variants retain the old deterministic
fastest-backup rule.

Operationally, the profile-update order is:

1. add the primary sample
2. if hedged, add the backup sample as an explorer update
3. run the usual background probe path

## 4. Canonical Simulator Surface

The experiment must be implemented on top of the merged simulator:

- `routewise-simulator/`

Canonical package surface:

- `rwsim/`
- `rwsim/world/`
- `rwsim/runner.py`
- `rwsim/strategies/`

Important compatibility detail:

- Golden-baseline capture still enumerates the default tiered strategies through the legacy tiered path.
- Therefore the new LP-budget experiment must not modify the default tiered strategy list used by golden tests.

## 5. Implementation Strategy

The experiment should be implemented as a **sidecar evaluation harness**, not as a default built-in strategy family.

Recommended files:

- `routewise-simulator/experiment/scripts/simulate/synthetic/tiered/lp_budget_eval.py`
- `routewise-simulator/run_joint_lp_budget_eval.py`

Recommended output directory:

- `routewise-simulator/results/lp_budget/`

This keeps the research experiment isolated from:

- golden-baseline verification
- default strategy registry behavior
- legacy scripts used for merge validation

The sidecar harness should also support freezing an accepted result snapshot under:

- `routewise-simulator/results/lp_budget/golden/`

This is for experiment-side regression checking only. It is not part of the main simulator golden-capture path.

## 6. Variants to Run

### 6.1 Main variants

These are the only variants in the main experiment matrix:

- `old_body`
- `old_body_hedge`
- `budget_vhat_t25`
- `budget_vhat_t50`
- `budget_vhat_t75`
- `budget_vhat_t25_hedge`
- `budget_vhat_t50_hedge`
- `budget_vhat_t75_hedge`

### 6.2 Provider-percentile comparator

The older provider-percentile budget family is still useful as a comparator,
but it is no longer the mainline formulation:

- `budget_body_p25`
- `budget_body_p50`
- `budget_body_p75`
- `budget_body_p25_hedge`
- `budget_body_p50_hedge`
- `budget_body_p75_hedge`

### 6.3 Recommended control baselines

These are recommended sanity controls, but they are not part of the primary hypothesis matrix:

- `cheapest_available`
- `fastest_available`

These controls should be implemented in the sidecar harness with tier-aware availability semantics, i.e. they choose among providers satisfying `p.is_available(now)` at the current decision. We should not directly reuse the legacy latency-only fixed baselines because those ignore tier-capacity availability.

### 6.4 Optional hedge-ablation variants

To isolate the hedge redesign from the LP redesign, the sidecar harness may
also run a minimal 2x2 hedge ablation with the old hedge trigger:

- `old_body_oldhedge`
- `budget_vhat_t75_oldhedge`

These are not part of the default mainline matrix. They are only used to
separate:

- LP contribution under a fixed hedge rule
- hedge-trigger contribution under a fixed body selector

### 6.5 Interpretation of variants

- `old_body`: old LP body selector, hedge disabled
- `old_body_hedge`: old LP body selector, probability-only hedge enabled
- `old_body_oldhedge`: old LP body selector, legacy hedge enabled
- `budget_vhat_tXX`: new budgeted LP selector with per-request budget `tau * v_hat_i`, hedge disabled
- `budget_vhat_tXX_hedge`: same selector, probability-only hedge enabled
- `budget_vhat_t75_oldhedge`: `tau = 0.75` selector with legacy hedge enabled, used in the 2x2 ablation
- `budget_body_pXX`: provider-percentile comparator, hedge disabled
- `budget_body_pXX_hedge`: same comparator, probability-only hedge enabled

## 7. Exact Selector Definitions

### 7.1 Old selector: `old_body`

Use the old LP semantics as the baseline:

```text
minimize   sum_j pi_j * c_eff_j
subject to sum_j pi_j * F_j(SLO) >= target
           sum_j pi_j = 1
           pi_j >= 0
```

Notes:

- `c_eff_j` is the existing cross-tier effective cost
- `F_j(SLO)` is the provider's estimated SLO attainment
- `target` is fixed at `0.99`
- Existing relaxation / fallback behavior should be preserved in the baseline implementation

### 7.2 New selector: `budget_vhat_tXX`

For request `i`, define the request-side API price anchor:

```text
v_hat_i = estimated API price anchor for request i
```

In the synthetic simulator implementation:

```text
v_hat_i = request.total_tokens * min_{p in S_A}(api_price_per_token(p))
```

Then solve:

```text
minimize   sum_j pi_j * Tbar_j
subject to sum_j pi_j * c_eff_j <= tau * v_hat_i
           sum_j pi_j = 1
           pi_j >= 0
```

Where:

- `c_eff_j` is the existing effective cost from the tiered world model
- `Tbar_j` is the body-latency proxy
- `tau` is one of `{0.25, 0.50, 0.75}`

### 7.3 Provider-percentile comparator: `budget_body_pXX`

This is kept only as a comparator / ablation:

```text
feasible(now) = {p : p.is_available(now)}
B_tau = percentile_tau({c_eff_j over feasible(now)})
```

Then solve:

```text
minimize   sum_j pi_j * Tbar_j
subject to sum_j pi_j * c_eff_j <= B_tau
           sum_j pi_j = 1
           pi_j >= 0
```

### 7.4 New-selector fallback contract

The new LP must have an explicit fallback contract so the experiment compares formulations rather than accidental implementation behavior.

Fallback order for `budget_vhat_tXX` and `budget_body_pXX`:

1. **Normal LP solution**
   - The LP solver returns success and a normalized non-empty weight vector.
   - Status label: `optimal`

2. **Degenerate feasible fallback**
   - Trigger: solver succeeds numerically but returns an all-zero or unusable weight vector after clipping / normalization.
   - Action: restrict to providers with `c_eff_j <= B_i + eps` and choose the one with smallest `Tbar_j`, breaking ties by smallest `c_eff_j`.
   - Status label: `degenerate_feasible_fallback`

3. **Solver failure fallback**
   - Trigger: LP solve reports infeasible, unbounded, numerical failure, or timeout.
   - Action: restrict to providers with `c_eff_j <= B_i + eps` and choose the one with smallest `Tbar_j`, breaking ties by smallest `c_eff_j`.
   - Status label: `solver_failure_fallback`

4. **No-in-budget fallback**
   - Trigger: no provider satisfies `c_eff_j <= B_i + eps` because of budget tightness or floating-point edge cases.
   - Action: choose the minimum-`c_eff_j` provider.
   - Status label: `no_in_budget_fallback`

5. **No-capacity fallback**
   - Trigger: there are no currently feasible providers.
   - Action: fall back to the existing no-capacity behavior already used by the tiered runner.
   - Status label: `no_capacity_fallback`

Requirements:

- `eps` must be fixed and documented in the implementation.
- Diagnostics must count fallback events by reason, not just in aggregate.
- Diagnostics must also count decisions that did not end with the intended LP solution path, i.e. every status other than `optimal`.

### 7.5 Body-latency proxy

The proxy order should be:

1. Rolling mean TTFT if the provider has at least 5 in-window TTFT samples
2. Rolling median TTFT if the provider has 1-4 in-window TTFT samples
3. Analytical `true_p50_ms(...)` only as a cold-start fallback when the in-window sample count is 0

The rolling window must match the current tiered profile window, i.e. 900 seconds.

The experiment must count how often the analytical fallback is used.

### 7.6 Hedge trigger contract

All hedged variants use the same hedge trigger:

```text
P(not violate | t) + P(violate | t) * P(backup succeeds) >= 0.99
```

Implementation contract:

1. Backup selection remains unchanged from the current tiered runner:
   choose the fastest non-primary cross-tier backup.
2. `delta` is fixed at 50 ms.
3. `F_p` and `F_b` are evaluated from the synthetic providers' active TTFT
   distributions in the simulator.
4. The policy computes the latest safe dispatch time `t*` on a fixed search grid.
5. If `T_primary > t*`, the backup is dispatched at `t*`.
6. Final TTFT is:

```text
min(T_primary, t* + delta + T_backup)
```

7. If no safe wait time exists but immediate hedging strictly improves
   success probability over no-hedge, dispatch immediately at `t = 0`.

## 8. Why the New LP Is Different

The old LP tries to optimize cost while enforcing a tail-driven hard constraint.

The new LP does not try to solve the tail directly. Instead it asks:

> Given a cost budget, what is the best body latency we can get?

This matches the intended division of labor:

- LP optimizes the body
- hedging protects the tail

## 9. Scenarios

### 9.1 First batch: required

Run all of the following first:

- `S6` from `tiered/scenarios.py`
- `S7` from `tiered/scenarios.py`
- `S8` from `tiered/scenarios.py`
- `s8m_multi_sq_hierarchy` from `tiered/scenarios_mm25.py`
- `s7m_quota_depletion` from `tiered/scenarios_mm25.py`

This first batch is the minimum set required before drawing any conclusion about the body-selector story. The MM25 multi-provider case is intentionally included here so we do not over-interpret two-point spillover behavior from `S6/S7/S8` alone. The MM25 quota-depletion case is also included here because dynamic shadow-price drift is a central mechanism for the budget-LP story.

### 9.2 Deferred

These can be run later if the first batch is promising:

- `s6m_featherless_saturation`

## 10. Experimental Phases

### Phase A: body selector only

Run:

- `old_body`
- `budget_vhat_t25`
- `budget_vhat_t50`
- `budget_vhat_t75`

Goal:

- Verify whether the `tau * v_hat_i` LP creates a clean body-level cost-latency frontier by itself

Primary questions:

- Do `P50` and `P90` improve as budget increases?
- Does mean TTFT move consistently as a secondary supporting metric?
- Does traffic mix move monotonically toward faster providers?
- Does S6 reveal a clear failure boundary for the pure budget formulation?

### Phase B: add hedging

Run:

- `old_body_hedge`
- `budget_vhat_t25_hedge`
- `budget_vhat_t50_hedge`
- `budget_vhat_t75_hedge`

Goal:

- Verify whether hedging mainly improves tail metrics on top of the body selector

Primary questions:

- Does hedging mostly improve `P99` and `SLO violation`?
- Does hedging leave `mean TTFT` and `P50` mostly unchanged?
- Does hedge rate stay moderate and interpretable?

### Phase C: final comparison

Compare:

- `old_body_hedge`
- `budget_vhat_t25_hedge`
- `budget_vhat_t50_hedge`
- `budget_vhat_t75_hedge`

Goal:

- Decide whether the new full method is more compelling than the old full method

## 11. Metrics

### 11.1 Main outcome metrics

Every run must report:

- mean TTFT
- P50 TTFT
- P90 TTFT
- P99 TTFT
- SLO violation rate
- average billed cost
- hedge rate
- provider traffic mix
- tier traffic mix

### 11.2 Budget-faithfulness diagnostics

Every run must also report:

- mean budget `B_i`
- mean `E_pi[c_eff]`
- budget utilization: `E_pi[c_eff] / B_i`
- budget slack: `B_i - E_pi[c_eff]`
- per-decision utilization quantiles: `p10`, `p50`, `p90`
- per-decision slack quantiles: `p10`, `p50`, `p90`
- solver-status frequency
- fallback frequency by reason
- count of decisions that did not end with the intended LP solution path
- count of decisions with exactly one feasible provider
- count of trivially single-provider LP outcomes
- analytical `true_p50_ms` fallback count

These diagnostics are mandatory. Without them, we cannot verify that the new LP is actually behaving as a budget-constrained effective-cost optimizer.

## 12. Plots and Tables

### 12.1 Required tables

For each scenario, produce a summary table with:

- variant
- mean TTFT
- P50
- P90
- P99
- SLO violation
- average billed cost
- hedge rate
- provider mix
- tier mix

Also produce a diagnostics table with:

- variant
- mean budget `B_i`
- mean `E_pi[c_eff]`
- utilization
- slack
- utilization quantiles
- slack quantiles
- solver-status counts
- fallback counts by reason
- non-optimal decision count
- single-feasible-provider decision count
- trivial-single-provider outcome count

### 12.2 Required plots

For each scenario:

- cost vs mean TTFT
- cost vs P99

For hedged variants, also produce delta summaries:

- `delta P50`
- `delta P90`
- `delta P99`
- `delta SLO violation`
- `delta cost`

where delta is measured from the corresponding no-hedge variant.

## 13. Decision Rules

### 13.1 What counts as a good result

The idea is working if the following pattern appears:

- `budget_vhat_t25/t50/t75` form a clean and interpretable cost-body frontier
- provider/tier mix changes monotonically with the budget knob
- `budget_vhat` improves the body in `S7`, `S8`, and `s8m_multi_sq_hierarchy`
- `P50` and `P90` are the primary body evidence; mean TTFT is secondary supporting evidence
- hedging mainly improves `P99` and `SLO violation`, not the body metrics

### 13.2 What counts as a boundary result

If `budget_vhat` fails badly in `S6`, that should not be hidden.

The correct interpretation is:

> Pure budget LP has a clear failure regime in slow-free scenarios.

This is still a useful result. It defines the boundary of the idea.

### 13.3 What would invalidate the story

The story does not hold if:

- the `tau * v_hat_i` budget knob does not produce a meaningful frontier
- `P50` and `P90` do not improve in the budgeted variants
- hedging is doing most of the work while the body selector itself adds little value

### 13.4 Adoption rule for the new full method

The new full method should be considered compelling only if there exists at least one `budget_vhat_tXX_hedge` variant that, on at least 2 mandatory scenarios, simultaneously satisfies all of the following against `old_body_hedge`:

- lower mean TTFT
- lower or equal average billed cost
- no more than 10% relative regression in `P99`
- no more than 10% relative regression in SLO violation rate

## 14. Deliverables

The implementation should produce:

1. A short summary of the exact simulator code path used
2. A precise statement of the old selector as implemented
3. A precise statement of the new selector as implemented
4. Per-scenario results tables for all variants
5. Budget-diagnostics tables
6. Cost-vs-mean and cost-vs-P99 plots
7. A short interpretation for each scenario

## 15. Recommended Execution Order

1. Implement the sidecar harness on top of the merged simulator
2. Implement `old_body` and `budget_vhat_t25/t50/t75`
3. Add sidecar control baselines `cheapest_available` and `fastest_available`
4. Run Phase A on `S6/S7/S8`, `s8m_multi_sq_hierarchy`, and `s7m_quota_depletion`
5. Freeze an accepted sidecar result snapshot under `results/lp_budget/golden/`
6. If Phase A is promising, add hedged variants
7. Run Phase B on `S6/S7/S8`, `s8m_multi_sq_hierarchy`, and `s7m_quota_depletion`
8. Optionally run `s6m_featherless_saturation`
9. Optionally run the provider-percentile comparator family as an ablation
10. Decide whether the new full method is worth carrying into the paper narrative

## 16. Bottom Line

This plan treats the algorithm as already defined and focuses on validating one idea:

> Replace the old tail-constrained LP with a per-request `tau * v_hat_i` body-latency LP, while keeping hedging as the tail mechanism.

The synthetic simulator should answer whether this idea works, where it helps, and where it fails.
