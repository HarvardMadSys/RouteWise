# Final Synthetic Simulator Experiment Plan

## Purpose

This document defines the **final paper-facing synthetic simulator design** for
the ICML submission.

The goal is not to build a general-purpose simulator framework. The goal is to
produce a **small, coherent, paper-ready experimental suite** that can answer
the following three questions:

1. Does the new budgeted LP improve the **body-latency vs cost** tradeoff over
   the original LP?
2. Does hedging primarily improve the **tail** rather than compensate for a bad
   body selector?
3. Does the final method outperform the original full method in a realistic
   joint market setting?

This document intentionally simplifies earlier design branches and removes
process-oriented experimental scaffolding.

## Core Principle

The simulator should model:

- a **synthetic but calibrated provider world**
- driven by **real workload traces**

In other words:

- **Provider side is synthetic**:
  - provider tiers
  - provider latency distributions
  - provider costs
  - quota limits
  - concurrency limits
  - shadow-price-driven effective costs
- **Workload side is real**:
  - arrival timestamps
  - request token counts
  - response token counts

This is the correct abstraction boundary for the paper.

We do **not** want:

- synthetic provider world + synthetic workload as the main paper result
- many small toy worlds with missing tiers as the main story

Those can still be useful for internal debugging, but not as the final paper
presentation.

## Workloads

The final synthetic simulator should use **three real workload datasets**:

1. `freeinference`
2. `rednote`
3. `sharegpt`

These datasets provide:

- realistic burstiness
- realistic request-size heterogeneity
- realistic long-tail token distributions

This keeps the simulator aligned with the offline stage-1 evaluation, which
also uses three datasets.

## Workload Normalization

Because `freeinference`, `rednote`, and `sharegpt` have very different burst
profiles and token scales, each scenario must define a **workload scaling
rule** so that its intended regime remains stable across datasets.

For every scenario, we should specify:

- a target quota-utilization range
- a target concurrency-utilization range
- a load scale factor applied to timestamps
- whether the scenario is intended to be tail-light or tail-heavy

The purpose is not to erase workload differences. The purpose is to ensure that
`quota-dominant`, `concurrency-dominant`, and `boundary` remain meaningful
labels across all three datasets.

In the final parameter table, each scenario must include:

- target `z` regime for quota usage
- target `u` regime for concurrency usage
- load scaling rule
- expected overflow or spillover tendency

## Synthetic Provider World

Each simulator scenario defines a synthetic joint provider market with all three
tiers present:

- `S_Q`: quota-based providers
- `S_C`: concurrency-based providers
- `S_A`: API-based providers

Each scenario should include **multiple providers per tier**. As a default
target:

- `S_Q`: 2 to 3 providers
- `S_C`: 2 providers
- `S_A`: 2 to 3 providers

Each provider should have:

- a latency distribution
- a pricing model
- a quota or concurrency budget when applicable
- an effective cost contribution used by the LP

The final paper should present the simulator as a **unified joint market**, not
as a collection of tier-missing micro-worlds.

## Methods

The final method set contains **8 methods total**.

### Anchor baselines

These are not the main scientific baselines. They define the rough frontier
endpoints.

1. `Cheapest`
2. `Fastest`

### Heuristic baselines

These test whether simple tier-priority rules are sufficient.

3. `Quota-first`
4. `Concurrency-first`

### Main baselines

These represent the previous LP-based method family.

5. `Original LP`
6. `Original LP + Hedge`

### Proposed methods

These are the methods we want to defend in the paper.

7. `Budgeted LP (tau = 0.75)`
8. `Budgeted LP (tau = 0.75) + Hedge`

## Tau Sweep Policy

The paper's **main operating point** is still:

- `tau = 0.75`

However, to support the body-latency vs cost tradeoff claim, the simulator must
also run a **small tau sweep** outside the main table.

The recommended sweep is:

- `tau = 0.25`
- `tau = 0.50`
- `tau = 0.75`
- `tau = 1.00`

Presentation policy:

- the main table reports only `tau = 0.75`
- the appendix or a small side plot reports the tau sweep

This keeps the main story simple while still supporting the tradeoff claim.

## Main Table vs Plot Policy

To keep the paper clear:

- the **main table** should report:
  - `Quota-first`
  - `Concurrency-first`
  - `Original LP`
  - `Original LP + Hedge`
  - `Budgeted LP (tau = 0.75)`
  - `Budgeted LP (tau = 0.75) + Hedge`

- `Cheapest` and `Fastest` should still be run, but mainly appear as
  **anchor points in tradeoff plots**

This separation is important:

- `Original LP` is the main old-method baseline
- `Quota-first` and `Concurrency-first` are simple heuristics
- `Cheapest` and `Fastest` are frontier anchors, not serious competitors

## Method Definitions

### Original LP

The original LP keeps the previous formulation:

- minimize effective cost
- subject to a target SLO-attainment constraint

This is the old body-selector baseline.

### Budgeted LP

The new LP uses the request-driven budget formulation:

```text
min   sum_j pi_j * mean_TTFT_j
s.t.  sum_j pi_j * c_eff_j <= tau * v_hat_i
```

where:

- `mean_TTFT_j` is the body-latency estimate for provider `j`
- `c_eff_j` is the effective cost for provider `j`
- `v_hat_i` is the request-specific API cost anchor
- `tau = 0.75` is the paper's main operating point

### Hedging

The paper-facing hedging definition should remain simple:

- a probability-based hedge trigger
- one backup provider
- hedging mainly presented as the tail-improvement layer

The paper should avoid introducing too many hedging sub-variants in the main
text.

## Final Scenario Family

The final paper should use **4 unified joint scenarios**.

All four scenarios must contain all three tiers:

- multiple `S_Q`
- multiple `S_C`
- multiple `S_A`

### Scenario 1: Quota-dominant joint

Purpose:

- test whether the new LP improves the cost-body tradeoff when quota-side
  decisions dominate

Desired behavior:

- `Original LP` tends to over-favor cheap quota-side decisions
- `Budgeted LP` should better navigate body latency within the budget
- quota utilization should be meaningfully active, but not instantly saturate
- the scenario should be primarily tail-light

### Scenario 2: Concurrency-dominant joint

Purpose:

- test whether the new LP remains effective under concurrency saturation

Desired behavior:

- `Concurrency-first` should look appealing but become myopic
- `Budgeted LP` should spill to better alternatives more gracefully
- concurrency utilization should frequently enter the stressed regime
- the scenario should be at least moderately tail-heavy

### Scenario 3: Rich mixed joint market

Purpose:

- test the method in a truly general many-provider-per-tier market

Desired behavior:

- no simple heuristic should dominate
- the LP should show value beyond two-point switching
- both quota and concurrency should be meaningfully exercised
- the scenario may be tail-light or moderately tail-heavy, but must not collapse
  into a single obvious bottleneck

This should be the paper's strongest **generality scenario**.

### Scenario 4: Trap / boundary joint

Purpose:

- provide an honest negative or boundary case

Desired behavior:

- a very cheap but poor provider should exist
- low-quality selections should fail clearly
- hedging should still matter
- the scenario should be explicitly tail-heavy
- hedging should have visible P99 / SLO impact here

This scenario is important because the paper should not imply unconditional
dominance.

## Metrics

The final simulator should report the following **primary metrics**:

- mean TTFT
- P50 TTFT
- P99 TTFT
- SLO violation rate
- average realized economic cost

The final simulator should also report the following **secondary metrics**:

- hedge rate
- provider mix
- tier mix

Useful diagnostics:

- budget utilization
- fallback count
- average effective-cost utilization

Cost reporting policy:

- the main table should report **realized economic cost**
- `c_eff` should be treated as an internal routing quantity and only appear in
  diagnostics or appendix material

The paper should not drown the main story in low-level simulator telemetry.

## Main Comparisons

The simulator should answer the three core questions using these comparisons.

### Body-value comparison

Compare:

- `Original LP`
- `Budgeted LP (tau = 0.75)`

Purpose:

- show whether the new LP improves the body-latency vs cost tradeoff

Support material:

- a small tau sweep figure or appendix table

### Tail-value comparison

Compare:

- `Budgeted LP (tau = 0.75)`
- `Budgeted LP (tau = 0.75) + Hedge`

Purpose:

- show whether hedging mainly improves the tail

### Full-system comparison

Compare:

- `Original LP + Hedge`
- `Budgeted LP (tau = 0.75) + Hedge`

Purpose:

- show whether the new full method replaces the old full method

### Heuristic comparison

Compare:

- `Quota-first`
- `Concurrency-first`
- `Budgeted LP (tau = 0.75)`

Purpose:

- show that fixed tier-priority rules are insufficient in a unified market

## Final Experimental Grid

The final paper-facing grid should be:

- `3 workloads`
- `4 scenarios`
- `8 methods`

This is:

```text
3 x 4 x 8 = 96 runs
```

This is large enough to be convincing, but still small enough to remain
coherent.

## Randomness and Uncertainty

The final simulator results should not rely on a single run per cell.

Minimum requirement:

- run each cell with `3` to `5` random seeds

Recommended presentation:

- report means across seeds
- show confidence intervals or error bars in plots

If runtime becomes a concern, seeds should still be retained for:

- the main table methods
- the tau sweep
- at least one representative figure per scenario family

## What We Should Not Do

To keep the paper clean, we should avoid the following in the main text:

- many synthetic workloads
- many tau values in the main table
- many hedging variants
- many tiny mechanism-isolation worlds with missing tiers
- many diagnostic-only baselines

These can still be useful internally or in the appendix, but they should not
define the main experimental story.

## Immediate Next Steps

The next simulator tasks should be:

1. Finalize the **4 unified joint scenarios**
2. Finalize the **workload normalization rules** for all four scenarios
3. Run the **3 x 4 x 8** grid with multiple seeds
4. Run the **small tau sweep**
5. Produce:
   - one main results table
   - cost-vs-latency plots
   - one tau sweep figure
   - a short diagnostics summary

Only after that should we move on to the real-system experiments.

## One-sentence Summary

The final simulator should be:

> a unified synthetic joint provider market, driven by three real workload
> datasets, evaluated with eight methods, and presented through four
> paper-facing joint scenarios.
