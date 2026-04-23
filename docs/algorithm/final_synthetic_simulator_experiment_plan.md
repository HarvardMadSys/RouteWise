# Final Synthetic Simulator Experiment Plan

## Purpose

This document defines the final synthetic simulator design for the NSDI
submission.

The simulator is **not** the paper's main evaluation. The main evaluation
lives on real OpenRouter traffic and production traces. The simulator is
used for three narrow purposes:

1. **Sanity check**: verify that the new budgeted LP and smart hedging
   behave correctly on a controlled ground truth where the optimal policy
   is analytically defensible.
2. **Fast design iteration**: exercise formula or parameter changes
   (budget formulation, `tau` sweep, hedging rule, beta in concurrency
   shadow price) in minutes rather than paying for 24-hour live runs.
3. **Robustness evidence**: demonstrate that the routing decisions remain
   sensible under latency drift and provider trap conditions.

Everything else is a secondary concern. In particular we do not try to
convince the reader of system generality through the simulator alone; the
live OpenRouter experiments carry that load.

## Core Principle

The simulator models:

- a single **synthetic provider pool** with transparent parameters, and
- **three real workload datasets** driving the arrival process.

## Design Overview

### Provider pool: unified 5-slot pool across three tiers

All three tiers are instantiated from the **same five latency profiles**.
This is a deliberate control: method-to-method differences therefore
isolate tier-constraint handling (quota shadow price, concurrency shadow
price, pay-per-token cost) from per-provider latency variability.

Pool layout:

| Slot | P50 (ms) | P99 (ms) | Intent                    |
|------|----------|----------|---------------------------|
| 0    | 200      | 500      | fast-normal               |
| 1    | 250      | 2500     | tail-heavy (large sigma)  |
| 2    | 700      | 1800     | normal                    |
| 3    | 1000     | 2600     | trap (mid-run degrades)   |
| 4    | 1500     | 4000     | slow-deep (mid-run improves) |

Each slot is replicated in all three tiers:

- **S_Q** (quota subscription): free, quota scales with latency so that
  fast slots are scarce and slow slots are abundant (200, 400, 800, 1200,
  2000 requests respectively).
- **S_C** (concurrency subscription): free, concurrency limit scales with
  latency (1, 2, 3, 4, 5 slots), fixed 2 s service time per slot.
- **S_A** (pay-per-token): cost is deliberately non-monotone with
  latency; slot 2 is cheap and fast (dominant, reproducing real WandB /
  Alibaba observations on Qwen3-235B), slot 3 is expensive and slow
  (trap), slot 4 is the cheapest but slowest.

S_A cost per token: `[2e-6, 2e-6, 1e-6, 3e-6, 0.5e-6]`.

Pool size: 15 providers (5 per tier).

### Drift

Two S_A providers shift distribution mid-run to simulate the 5 x latency
drift reported in the paper introduction:

- `SA_s3_trap` degrades at 50 % of the simulation duration: P50 1000 ms ->
  1800 ms, P99 2600 ms -> 5000 ms.
- `SA_s4_slowdeep` improves at 30 % of the simulation duration: P50
  1500 ms -> 900 ms, P99 4000 ms -> 2400 ms.

S_Q and S_C are kept static because subscription tiers are typically more
stable in practice, and because including drift on every tier makes the
results harder to interpret.

### Per-request noise

Every TTFT sample is drawn from a LogNormal distribution parameterised by
the slot's (P50, P99). There is always per-request noise; "static" refers
only to the distribution parameters, not to the realised values.

## Workloads

Three real datasets drive the arrival process, matching the paper's
offline and online stage evaluation:

- `freeinference`
- `rednote`
- `sharegpt`

Trace loading preserves local burstiness and token counts and rescales
each trace into the scenario duration.

The scenario also has a synthetic fallback (`n_requests=5000`,
`duration=1500 s`, Poisson arrivals) retained for quick smoke tests and
for experiments that need controlled arrival processes.

## Methods (8 variants)

### Anchor baselines

1. `cheapest_available` - always pick the cheapest currently feasible
   provider.
2. `fastest_available` - always pick the lowest-P50 currently feasible
   provider.

### Heuristic baselines

3. `quota_first` - prioritise S_Q, spill to S_C then S_A when unavailable.
4. `concurrency_first` - prioritise S_C, spill to S_Q then S_A.

### Prior-method baselines

5. `original_lp` - the previous LP formulation:
   `min sum_j pi_j * c_eff_j s.t. sum_j pi_j * F_j(SLO) >= 0.99`
   with relaxation tiers and a best-effort fallback.
6. `original_lp_hedge` - (5) augmented with the current smart hedger.

### Proposed methods

7. `budget_vhat_t75` - the new budgeted LP:
   `min sum_j pi_j * Tbar_j s.t. sum_j pi_j * c_eff_j <= tau * v_hat_i`
   with `tau = 0.75` as the headline operating point.
8. `budget_vhat_t75_hedge` - (7) augmented with the same smart hedger.

The headline comparison the paper should defend is `(6)` vs `(8)`.

## Tau Sweep (appendix)

The paper main table reports only `tau = 0.75`. To support the
cost / latency tradeoff claim, the simulator also runs `tau in {0.25,
0.50, 0.75}` (the existing variants `budget_vhat_t25`,
`budget_vhat_t50`, `budget_vhat_t75`) and their hedged counterparts, as
an appendix figure.

## Metrics

Primary:

- mean TTFT, P50 TTFT, P99 TTFT
- SLO violation rate (reported at multiple thresholds: 500, 1000, 2000,
  3000, 5000 ms; 2000 ms is the primary SLO)
- realised economic cost (USD per request)

Secondary (diagnostic):

- hedge rate
- provider mix and tier mix
- budget utilisation distribution (`sum_j pi_j * c_eff_j` vs
  `tau * v_hat_i`)
- fallback counts

The main table reports realised economic cost. The effective-cost
quantity `c_eff` is treated as an internal routing signal and only
surfaces in diagnostics.

## Seeds

Each cell runs with seeds `{42, 43, 44}`. Means across seeds are reported
in the main numbers; confidence bands on Pareto plots.

## Runtime

Total main grid: `3 workloads x 8 methods x 3 seeds = 72 runs`.
Wall-clock on this machine is well under 30 minutes.

## Paper-facing Outputs

The simulator contributes at most a small section in the evaluation,
structured as:

1. A compact main table: 3 workloads x 8 methods, reporting cost, P50,
   P99, SLO violation, hedge rate.
2. One Pareto-tradeoff plot: cost vs P99 TTFT, averaged across workloads,
   with method markers.
3. One diagnostic plot: provider-mix or tier-mix under the headline
   method variants.

Everything else (tau sweep, shadow-price diagnostics, per-dataset
breakouts) goes to the appendix.

## Scenarios Retained for Regression / Appendix Only

The following single-failure-mode scenarios remain in the codebase for
regression testing and appendix material. They are **not** part of the
headline grid:

- `s6_slow_q_trap` - slow-S_Q trap, isolates SLO-anchored filter.
- `s7_quota_depletion` - quota depletion ramp, isolates `psi(z)`.
- `s8_concurrency_saturation` - concurrency saturation, isolates
  `lambda(u)`.
- `s7m_quota_depletion`, `s8m_multi_sq_hierarchy` - multi-provider
  variants of the above.

These are useful when debugging a specific mechanism in isolation but do
not, by themselves, defend the full method against reviewer scrutiny.

## Non-goals

To keep the scope honest, the simulator explicitly does not attempt:

- to reproduce the full 13-provider OpenRouter marketplace;
- to serve as the paper's generality claim;
- to sweep provider pool structure (multiple pools, multiple trap
  topologies, etc.) in the main text;
- to model cross-provider correlation or shared-latent-event outages.

If any of these become reviewer-blocking concerns, they should be
addressed by the live experiments or by additional appendix material,
not by expanding the main simulator grid.

## Immediate Next Steps

1. Run the full grid: `3 workloads x 8 methods x 3 seeds` via
   `run_joint_lp_budget_eval.py --scenario unified_pool --dataset
   freeinference --dataset rednote --dataset sharegpt`.
2. Run the tau sweep with the same workloads, appendix-only.
3. Produce the main table + Pareto plot + diagnostic plot described
   above.
4. Move to the live OpenRouter experiments for the main evaluation.

## One-sentence Summary

The simulator is a single 5-slot provider pool shared across S_Q, S_C,
and S_A, driven by three real workloads, evaluated with eight methods, and
used as a fast sanity + robustness check for the budgeted LP and smart
hedger before committing to live provider experiments.
